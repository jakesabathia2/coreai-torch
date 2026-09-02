# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Externalizing graphs into a separate artifact.

`ExternalizeSpec` already puts a matched submodule in its own ``coreai.graph`` and
calls it with ``coreai.invoke``. These tests cover the experimental additions that let
those graphs be **shipped separately** from the program that calls them:

* ``_graph_externalize`` marks a graph with the ``externalize`` attribute;
* ``_namespace`` places it in nested symbol tables, reached by a qualified reference;
* ``_externalize_graphs()`` splits the marked graphs into their own
  :class:`AIProgram`, leaving the source program intact.

The target shape, for a submodule externalized with
``_namespace="group_a.stage_one"``::

    module {
      module @group_a {
        module @stage_one {
          coreai.graph externalize noinline @<path>_spec_<in-types>(
              %arg0: tensor<...>, %arg1: tensor<...>) -> ... { ... }
      }}
      coreai.graph @main(...) {
        %n = coreai.invoke @group_a::@stage_one::@<path>_spec_<in-types>(%a, %b)
      }
    }

Two properties of that shape drive the tests:

1. **A submodule may take more than one tensor.** The motivating case is a side branch
   combining a layer input with that layer's output, so the boundary must survive
   externalization without being flattened to a single argument.
2. **Names must be reproducible.** A separately-exported base program and side
   artifact have to agree on symbol names, so a random suffix will not do.

Groups:

* **Group 1** — the multi-input submodule boundary, and the failure mode of an
  untyped ``*args`` forward.
* **Group 2** — the experimental feature itself. Everything in it is
  underscore-prefixed deliberately: no backwards-compatibility guarantee, and
  removable once something first-class exists.
* **Group 3** — the underlying ``coreai`` pass, pinned directly, so a compiler
  regression is distinguishable from a regression here.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
import torch.nn as nn

from coreai_torch import ExternalizeSpec, TorchConverter, get_decomp_table

from .utils import filecheck_pattern

# Conv2d with a 1x1 kernel over a [1, C, 1, S] layout: a channel projection, which is
# what a linear layer becomes in this layout.
IN_DIM, MID_DIM, OUT_DIM, SEQ = 16, 4, 8, 2


class SideBranch(nn.Module):
    """A submodule taking two tensors: a layer input and that layer's output.

    Computes ``base_out + up(down(x) * scale)`` — a low-rank branch summed onto the
    base path, so the combination happens *inside* the externalized graph.
    """

    def __init__(
        self, in_dim: int = IN_DIM, mid: int = MID_DIM, out_dim: int = OUT_DIM
    ):
        super().__init__()
        self.down = nn.Conv2d(in_dim, mid, kernel_size=1, bias=False)
        self.up = nn.Conv2d(mid, out_dim, kernel_size=1, bias=False)
        self.scale = 0.5

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        return base_out + self.up(self.down(x) * self.scale)


class VarArgsBranch(nn.Module):
    """The same computation behind an untyped ``*args/**kwargs`` forward.

    A forward like this cannot carry a tensor schema, which is a real situation when a
    module's ``forward`` serves several calling conventions.
    """

    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(IN_DIM, MID_DIM, kernel_size=1, bias=False)
        self.up = nn.Conv2d(MID_DIM, OUT_DIM, kernel_size=1, bias=False)

    def forward(self, *args, **kwargs) -> torch.Tensor:  # type: ignore[no-untyped-def]
        x, base_out = args[0], args[1]
        return base_out + self.up(self.down(x) * 0.5)


class OneLayer(nn.Module):
    """A base projection plus one side branch."""

    def __init__(self, branch_cls: type = SideBranch):
        super().__init__()
        self.base = nn.Conv2d(IN_DIM, OUT_DIM, kernel_size=1, bias=False)
        self.branch = branch_cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x, self.base(x))


class ThreeBranches(nn.Module):
    """Three call sites of the same class: each needs its own graph."""

    def __init__(self):
        super().__init__()
        self.q = OneLayer()
        self.k = OneLayer()
        self.v = OneLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q(x) + self.k(x) + self.v(x)


def _sample() -> tuple[torch.Tensor, ...]:
    return (torch.randn(1, IN_DIM, 1, SEQ),)


def _convert(model: nn.Module, specs: list, sample=None):
    sample = sample or _sample()
    return (
        TorchConverter()
        .add_pytorch_module(
            model,
            export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
                get_decomp_table()
            ),
            externalize_modules=specs,
        )
        .to_coreai()
    )


# ---------------------------------------------------------------------------
# Group 1 -- the multi-input submodule boundary.
# ---------------------------------------------------------------------------
@pytest.mark.ir
def test_two_tensor_boundary_ir() -> None:
    """A `(x, base_out) -> out` submodule externalizes with both tensors as args."""
    torch.manual_seed(42)
    program = _convert(OneLayer().eval(), [SideBranch])

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @branch_{{[0-9a-f]+}}(%arg0: tensor<1x16x1x2xf32>
        // CHECK-SAME:  %arg1: tensor<1x8x1x2xf32>
        // CHECK:     coreai.conv2d
        // CHECK:     coreai.conv2d
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @branch_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(program), check_file=check_file)


def test_varargs_forward_is_rejected_with_actionable_error() -> None:
    """An untyped forward cannot be externalized -- and the error must say why.

    `_prepare_module` builds a `torch.library.custom_op` from the forward, and schema
    inference rejects varargs. Pinning this keeps the failure mode explicit: the fix is
    to give the module a typed signature, not to fall back to a single-tensor schema,
    which would silently break any submodule taking more than one tensor.
    """
    torch.manual_seed(42)
    with pytest.raises((ValueError, TypeError, RuntimeError)) as exc:
        _convert(OneLayer(branch_cls=VarArgsBranch).eval(), [VarArgsBranch])
    message = str(exc.value)
    assert "varargs" in message or "positional-only" in message, (
        f"expected a schema-inference error naming varargs, got: {message[:400]}"
    )


@pytest.mark.ir
def test_three_call_sites_each_get_their_own_graph_ir() -> None:
    """Same class, three instances: three graphs and three invokes, no dedup."""
    torch.manual_seed(42)
    program = _convert(ThreeBranches().eval(), [SideBranch])
    ir = str(program)

    assert ir.count("coreai.graph noinline @") == 3, (
        f"expected 3 externalized graphs, got {ir.count('coreai.graph noinline @')}"
    )
    assert ir.count("coreai.invoke @") == 3, (
        f"expected 3 invokes, got {ir.count('coreai.invoke @')}"
    )


async def test_numerics_match_eager() -> None:
    """Externalizing a submodule must not change the numbers."""
    from coreai.runtime import NDArray

    from .utils import compare_outputs

    torch.manual_seed(42)
    model = OneLayer().eval()
    sample = _sample()
    program = _convert(model, [SideBranch], sample=sample)

    with TemporaryDirectory(suffix=".aimodel") as tmp:
        asset = program.save_asset(Path(tmp))
        async with asset.executable() as ai_model:
            fn = ai_model.load_function("main")
            out = await fn(inputs={"x": NDArray(sample[0])})
            got = {k: v.numpy() for k, v in out.items()}
            key = next(iter(got))
            assert compare_outputs({key: model(*sample)}, got)


# ---------------------------------------------------------------------------
# Group 2 -- the experimental externalize-to-separate-artifact feature.
# ---------------------------------------------------------------------------
@pytest.mark.ir
def test_graph_externalize_marks_the_graph_ir() -> None:
    """`_graph_externalize=True` must put `externalize` on the emitted graph.

    That attribute is the only thing `ISOLATE_EXTERNALIZED_GRAPHS` keys on, so it is
    the hinge of the whole feature.

    Note `externalize` and `private` are mutually exclusive per the op definition, so
    this cannot be combined with `composite_op_name`.
    """
    torch.manual_seed(42)
    program = _convert(
        OneLayer().eval(),
        [ExternalizeSpec(target_class=SideBranch, _graph_externalize=True)],
    )
    ir = str(program)
    assert "coreai.graph externalize" in ir, (
        f"no graph carries the externalize attribute:\n{ir[:600]}"
    )


def test_externalize_graphs_splits_and_leaves_source_intact() -> None:
    """Extraction must yield a marked-graphs-only program *without* consuming the source.

    `ISOLATE_EXTERNALIZED_GRAPHS` mutates its module in place and `AIProgram` holds
    `_mlir_module` by reference, so a naive call destroys the program you still need to
    save. The helper clones first, so callers are order-independent and can inspect
    both artifacts.
    """
    from coreai_torch import _externalize_graphs

    torch.manual_seed(42)
    program = _convert(
        OneLayer().eval(),
        [ExternalizeSpec(target_class=SideBranch, _graph_externalize=True)],
    )
    before = str(program)

    extracted = _externalize_graphs(program)
    extracted_ir, source_ir = str(extracted), str(program)

    assert "@main" not in extracted_ir, (
        f"extracted program should not contain the entrypoint:\n{extracted_ir[:400]}"
    )
    assert "coreai.conv2d" in extracted_ir, "extracted program lost the graph body"
    assert source_ir == before, (
        "extraction mutated the source program; it must clone (see docstring)"
    )
    # The op contract: survivors do not keep the attribute once externalized.
    assert "coreai.graph externalize" not in extracted_ir, (
        "externalize should be stripped from graphs in the extracted program"
    )


@pytest.mark.ir
def test_graphs_are_nested_under_a_namespace_ir() -> None:
    """A dotted `_namespace` must nest the graph and qualify the call site.

    `"group_a.stage_one"` becomes `module @group_a { module @stage_one { ... } }`, and
    the invoke reaches it as `@group_a::@stage_one::@<name>`. Nesting uses
    `builtin.module` — a symbol table, which is what makes the qualified reference
    resolve — because the `coreai` dialect has no module op of its own.
    """
    torch.manual_seed(42)
    program = _convert(
        OneLayer().eval(),
        [
            ExternalizeSpec(
                target_class=SideBranch,
                _graph_externalize=True,
                _namespace="group_a.stage_one",
            )
        ],
    )
    check_file = """
        // CHECK:   udml.namespace @group_a {
        // CHECK:     namespace @stage_one {
        // CHECK:       coreai.graph externalize noinline @
        // CHECK:   coreai.invoke @group_a::@stage_one::@
    """
    filecheck_pattern(str(program), check_file=check_file)


def test_nested_graphs_survive_asset_serialization() -> None:
    """A namespaced program must serialize, not just verify in memory.

    Nesting is emitted with `udml.namespace` ops the converter creates directly. Two
    things this catches that an IR check cannot: a nested `builtin.module` (the obvious
    choice) is unserializable, and an op created without an explicit location inherits
    whatever is ambient, which can be a location the bytecode writer cannot represent -- the module then verifies and
    prints fine but `save_asset` fails with

        Failed to serialize module to Bytecode: at #aicode.debuginfo.location_v1<
            src = <file = <filename = "-", directory = "", sha256Sum = "">, ...

    reported against the nested module op itself. An IR-only check cannot see this, so
    this test writes the asset and reads it back.
    """
    torch.manual_seed(42)
    program = _convert(
        OneLayer().eval(),
        [
            ExternalizeSpec(
                target_class=SideBranch,
                _graph_externalize=True,
                _namespace="group_a.stage_one",
            )
        ],
    )

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested.aimodel"
        program.save_asset(path)
        assert (path / "main.mlirb").is_file()

        from coreai.authoring import AIProgram

        reloaded = str(
            AIProgram._load_bytecode(path / "main.mlirb")._mlir_module  # noqa: SLF001
        )
        assert "udml.namespace @group_a" in reloaded, "namespace lost in round-trip"
        assert "namespace @stage_one" in reloaded, "inner namespace lost in round-trip"
        assert "@group_a::@stage_one::@" in reloaded, "qualified callee lost"


def test_externalized_graph_names_are_deterministic() -> None:
    """Two identical conversions must produce identical symbol names.

    Names are otherwise `f"{module_path}_{uuid4().hex[:8]}"`, i.e. different on every
    run, so a separately-exported base program and side artifact would disagree on
    symbols. With `_graph_externalize` the suffix is derived from the call site's input
    type signature, which is stable and also disambiguates shape specialisations.
    """
    import re

    def symbols() -> list[str]:
        torch.manual_seed(42)
        ir = str(
            _convert(
                OneLayer().eval(),
                [ExternalizeSpec(target_class=SideBranch, _graph_externalize=True)],
            )
        )
        return sorted(
            re.findall(r"coreai\.graph (?:noinline |externalize )*@(\S+?)\(", ir)
        )

    first, second = symbols(), symbols()
    assert first == second, (
        f"symbol names are not reproducible across runs:\n{first}\n{second}"
    )


# ---------------------------------------------------------------------------
# Group 3 -- the underlying coreai pass, pinned directly.
# ---------------------------------------------------------------------------
def test_isolate_pass_keeps_only_marked_graphs_and_strips_the_attribute() -> None:
    """`ISOLATE_EXTERNALIZED_GRAPHS` semantics the feature relies on."""
    import numpy as np
    from coreai._compiler._transforms.passes import (
        CorePasses,
        GlobalOptions,
        PassEntry,
        apply_passes_sync,
    )
    from coreai._compiler.context import Context
    from coreai._compiler.dialects import coreai as c
    from coreai._compiler.ir import (
        F16Type,
        InsertionPoint,
        Location,
        Module,
        RankedTensorType,
    )
    from coreai.authoring import AIProgram

    ctx = Context()
    with ctx._mlir_context, Location.unknown():  # noqa: SLF001
        ty = RankedTensorType.get([1, 4], F16Type.get())
        module = Module.create()

        def graph(name: str, externalize: bool) -> None:
            with InsertionPoint(module.body):
                g = c.GraphOp(name, [ty], [ty], ["x"], ["y"], externalize=externalize)
            block = g.regions[0].blocks[0]
            with InsertionPoint(block):
                one = c.constant(np.ones((1, 4), dtype=np.float16))
                c.output([c.add(block.arguments[0], one)])

        graph("base_entry", False)
        graph("marked", True)

        with TemporaryDirectory() as tmp:
            apply_passes_sync(
                module,
                [PassEntry.get(CorePasses.ISOLATE_EXTERNALIZED_GRAPHS)],
                GlobalOptions(output_directory=Path(tmp)),
            )

        names = [
            op.attributes["sym_name"].value
            for op in module.body.operations
            if "sym_name" in op.attributes
        ]
        assert names == ["marked"], f"expected only the marked graph, got {names}"
        assert "externalize" not in str(module), (
            "the pass should strip `externalize` from graphs it keeps"
        )
        assert module.operation.verify()
        # The pass is destructive: nothing of the source program survives.
        assert AIProgram._from_mlir_module(module) is not None  # noqa: SLF001


def test_externalize_attribute_is_settable_after_the_graph_is_built() -> None:
    """`graph_op.externalize = True` works post-hoc, and the module still verifies.

    This is what makes `_graph_externalize` cheap to implement: the converter can mark
    the graph after `_get_graph_op` rather than threading the flag into construction.
    """
    import numpy as np
    from coreai._compiler.context import Context
    from coreai._compiler.dialects import coreai as c
    from coreai._compiler.ir import (
        F16Type,
        InsertionPoint,
        Location,
        Module,
        RankedTensorType,
    )

    ctx = Context()
    with ctx._mlir_context, Location.unknown():  # noqa: SLF001
        ty = RankedTensorType.get([1, 4], F16Type.get())
        module = Module.create()
        with InsertionPoint(module.body):
            g = c.GraphOp("g", [ty], [ty], ["x"], ["y"])
        block = g.regions[0].blocks[0]
        with InsertionPoint(block):
            one = c.constant(np.ones((1, 4), dtype=np.float16))
            c.output([c.add(block.arguments[0], one)])

        assert g.externalize is False
        g.externalize = True
        assert g.externalize is True
        assert "coreai.graph externalize" in str(module)
        assert module.operation.verify()
