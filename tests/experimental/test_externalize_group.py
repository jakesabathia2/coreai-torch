# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Grouping a module's converted ops, and shipping the result separately.

``TorchConverter._set_externalize_group`` names a module class and outlines every op each
instance contributed into its own ``coreai.graph``, marked ``externalize`` and reached by a
``coreai.invoke``. Matching is by attribution -- what ``nn_module_stack`` says produced each
op -- so it asks nothing of the module: no typed boundary, no schema, no particular call
convention. ``Protocol`` below is the shape that forces the difference: its ``forward``
dispatches on Python containers, so there is nothing a custom op could stand in for.

The rest covers what happens to such graphs afterwards -- ``_externalize_graphs()`` splitting
them into their own :class:`AIProgram`, the namespaces surviving a real asset round-trip, and
the ``coreai`` pass underneath pinned directly.
"""

import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
import torch.nn as nn
from coreai.runtime import NDArray

from coreai_torch import TorchConverter, get_decomp_table

from ..utils import compare_outputs

DIM = 4
RANK = 2


class Protocol(nn.Module):
    """``forward`` takes a protocol, not tensors, and owns children that emit ops."""

    def __init__(self) -> None:
        super().__init__()
        self.a_transpose = nn.Linear(DIM, RANK, bias=False)
        self.b_transpose = nn.Linear(RANK, DIM, bias=False)
        self.scale = 0.5

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        x, base_out = kwargs["transformed_args"][0], kwargs["outputs"]
        return base_out + self.b_transpose(self.a_transpose(x) * self.scale)


class AdaptedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(DIM, DIM, bias=False)
        self.branch = Protocol()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x, transformed_args=(x,), outputs=self.base(x))


class Net(nn.Module):
    def __init__(self, layers: int = 1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(AdaptedLinear() for _ in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x * 2.0


class Scale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.factor = 3.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


class CalledTwice(nn.Module):
    """One instance invoked twice, with the parent's own op in between.

    A parent and its children are contiguous by construction, since attribution collects
    the whole subtree -- so this is what a broken span actually looks like: two runs of
    ops for the same instance with unrelated work between them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shared = Scale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(self.shared(x) + 1.0)


X = torch.randn(2, DIM)


def _convert(
    model: nn.Module,
    groups: dict[str, Any],
    *,
    entrypoints: tuple[str, ...] = ("main",),
    args: tuple = (X,),
    name_map: dict[str, str] | None = None,
    externalize: bool = True,
):
    """Export + convert with externalize groups set. Returns the AIProgram."""
    ep = torch.export.export(model, args).run_decompositions(get_decomp_table())
    converter = TorchConverter()
    for cls, namespace in groups.items():
        converter._set_externalize_group(cls, namespace, externalize=externalize)
    if name_map:
        converter._set_externalize_name_map(name_map)
    for entrypoint in entrypoints:
        converter.add_exported_program(ep, entrypoint_name=entrypoint)
    program = converter.to_coreai()
    program._mlir_module.operation.verify()  # noqa: SLF001
    return program


def _outlined(program) -> list[str]:  # type: ignore[no-untyped-def]
    """Headers of the outlined graphs (everything but the entrypoints)."""
    return [
        h
        for h in re.findall(r"coreai\.graph[^\n]*", str(program))
        if "noinline" in h or "externalize" in h
    ]


def _body_of(program, symbol: str) -> str:  # type: ignore[no-untyped-def]
    """The text of the outlined graph named ``symbol``, up to its terminator."""
    ir = str(program)
    start = ir.index(f"@{symbol}(")
    end = ir.index("coreai.output", start)
    return ir[start:end]


# --- the boundary attribution produces ---------------------------------------------


def test_matched_module_is_outlined_into_its_own_graph() -> None:
    program = _convert(Net(), {"Protocol": "lora.extend"})
    assert len(_outlined(program)) == 1
    assert "coreai.invoke @lora::@extend::@layers.0.branch" in str(program)


def test_no_group_set_emits_no_outlined_graph() -> None:
    """Opt-in: without the call, conversion is untouched."""
    ir = str(_convert(Net(), {}))
    assert "udml.namespace" not in ir
    assert "coreai.invoke" not in ir


def test_children_of_the_matched_module_come_with_it() -> None:
    """The projections belong to their own ``Linear``, but to the side branch's graph.

    Keying on the innermost module would leave the side branch only the ops it performed
    *between* its children's, which is not a boundary at all.
    """
    program = _convert(Net(), {"Protocol": "lora.extend"})
    body = _body_of(program, "layers.0.branch")
    assert body.count("batch_matmul") == 2, body  # both projections
    assert "broadcasting_add" in body  # and the residual add


def test_base_layer_stays_outside() -> None:
    program = _convert(Net(), {"Protocol": "lora.extend"})
    body = _body_of(program, "layers.0.branch")
    # three matmuls exist in total: base + two projections
    assert str(program).count("batch_matmul") == 3
    assert body.count("batch_matmul") == 2


def test_arguments_are_unnamed() -> None:
    """Attribution has no parameter names to draw on."""
    header = _outlined(_convert(Net(), {"Protocol": "lora.extend"}))[0]
    assert "coreai.name" not in header, header


def test_boundary_is_the_values_that_cross_it() -> None:
    """``(x, base_out) -> out``: two operands in, one result out."""
    header = _outlined(_convert(Net(), {"Protocol": "lora.extend"}))[0]
    assert header.count("%arg") == 2, header
    invoke = re.search(
        r"coreai\.invoke @lora::@extend::@\S+\(([^)]*)\)",
        str(_convert(Net(), {"Protocol": "lora.extend"})),
    )
    assert invoke is not None and invoke.group(1).count(",") == 1


def test_weights_are_pulled_into_the_outlined_graph() -> None:
    """A constant used only inside travels with the ops rather than becoming an operand."""
    model = Net()
    with torch.no_grad():
        model.layers[0].branch.a_transpose.weight.fill_(0.25)
    program = _convert(model, {"Protocol": "lora.extend"})
    assert "2.500000e-01" in _body_of(program, "layers.0.branch")


# --- naming and placement ---------------------------------------------------------


def test_graph_is_named_for_the_module_path() -> None:
    program = _convert(Net(layers=2), {"Protocol": "lora.extend"})
    symbols = {h.split("@")[1].split("(")[0] for h in _outlined(program)}
    assert symbols == {"layers.0.branch", "layers.1.branch"}


def test_one_graph_per_instance() -> None:
    program = _convert(Net(layers=3), {"Protocol": "lora.extend"})
    assert len(_outlined(program)) == 3


def test_namespace_nesting_and_qualified_call() -> None:
    ir = str(_convert(Net(), {"Protocol": "lora_32.extend"}))
    assert "udml.namespace @lora_32" in ir
    assert "namespace @extend" in ir
    assert "coreai.invoke @lora_32::@extend::@" in ir


def test_graph_is_marked_externalize() -> None:
    assert "externalize" in _outlined(_convert(Net(), {"Protocol": "lora.extend"}))[0]


def test_externalize_false_leaves_a_plain_noinline_graph() -> None:
    header = _outlined(_convert(Net(), {"Protocol": "lora.extend"}, externalize=False))[
        0
    ]
    assert "noinline" in header and "externalize" not in header


def test_callable_namespace_is_given_the_entrypoint() -> None:
    """One converter, several procedures, one namespace each.

    Conversion is deferred to ``to_coreai``, so a plain string cannot be re-set between
    ``add_exported_program`` calls -- the last value would win for every procedure.
    """
    program = _convert(
        Net(),
        {"Protocol": lambda ep: f"lora_32.{ep}"},
        entrypoints=("extend", "prompt_opt"),
    )
    ir = str(program)
    assert "coreai.invoke @lora_32::@extend::@" in ir
    assert "coreai.invoke @lora_32::@prompt_opt::@" in ir


def test_same_name_in_two_namespaces_does_not_collide() -> None:
    """Every procedure has a side branch at the same module path; only the namespace differs."""
    program = _convert(
        Net(),
        {"Protocol": lambda ep: f"lora_32.{ep}"},
        entrypoints=("extend", "prompt_opt"),
    )
    assert len(_outlined(program)) == 2
    # twice as a definition, twice as a qualified call -- the two definitions are
    # distinguished by their namespace, not by their symbol
    assert str(program).count("@layers.0.branch(") == 4
    assert (
        str(program).count("coreai.graph externalize noinline @layers.0.branch(") == 2
    )


def test_name_map_pins_the_emitted_symbol() -> None:
    """For a tree restructured between procedures -- see ``_set_externalize_name_map``."""
    program = _convert(
        Net(),
        {"Protocol": "lora.extend"},
        name_map={"layers.0.branch": "pinned.path.branch"},
    )
    ir = str(program)
    assert "@pinned.path.branch(" in ir
    assert "coreai.invoke @lora::@extend::@pinned.path.branch" in ir


def test_name_map_entries_that_match_nothing_are_ignored() -> None:
    """One map can cover every procedure, including those needing no pinning."""
    program = _convert(
        Net(), {"Protocol": "lora.extend"}, name_map={"not.a.module": "x"}
    )
    assert "@layers.0.branch(" in str(program)


# --- what cannot be grouped ------------------------------------------------------


def test_unmatched_class_name_warns_rather_than_silently_doing_nothing() -> None:
    """Matching goes through ``nn_module_stack``, so a no-match has to be loud.

    A module invoked as ``module.forward(...)`` never enters the stack, so registering it
    produces no graphs while conversion succeeds -- the missing side branchs would otherwise
    surface only much later, if at all.
    """
    with pytest.warns(UserWarning, match="no ops were attributed"):
        program = _convert(Net(), {"NoSuchModule": "lora.extend"})
    assert _outlined(program) == []


def test_matched_class_does_not_warn() -> None:
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        _convert(Net(), {"Protocol": "lora.extend"})


def test_non_contiguous_module_is_skipped_with_warning() -> None:
    """Ops in between belong to another module; outlining them would move its work."""
    with pytest.warns(UserWarning, match="not contiguous"):
        program = _convert(CalledTwice(), {"Scale": "g.main"})
    assert _outlined(program) == []


def test_set_externalize_group_returns_self_for_chaining() -> None:
    converter = TorchConverter()
    assert converter._set_externalize_group("Protocol", "lora.extend") is converter
    assert converter._set_externalize_name_map({}) is converter


# --- interaction with delegation --------------------------------------------------


def test_outlining_composes_with_delegate_ids() -> None:
    """Both groupings read op spans recorded during conversion.

    The spans are *indices*, and each grouping rewrites the block -- so one must not
    invalidate the other's. Delegating `Scale` and outlining the side branch exercises that.
    """

    class Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = AdaptedLinear()
            self.scale = Scale()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.scale(self.layer(x))

    ep = torch.export.export(Mixed(), (X,)).run_decompositions(get_decomp_table())
    converter = TorchConverter()
    converter._set_delegate_id("Scale", "Interpreter")
    converter._set_externalize_group("Protocol", "lora.extend")
    converter.add_exported_program(ep, entrypoint_name="main")
    program = converter.to_coreai()
    program._mlir_module.operation.verify()  # noqa: SLF001
    ir = str(program)
    assert 'coreai.isolated_group<"Interpreter">' in ir
    assert "coreai.invoke @lora::@extend::@layer.branch" in ir
    # the delegated region must hold Scale's op, not the side branch's
    body = _body_of(program, "layer.branch")
    assert "isolated_group" not in body


# --- numerics --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outlining_preserves_numerics() -> None:
    """Outlining is a structural move: the answer must be unchanged.

    Run with ``externalize=False`` so the call resolves in one asset -- a graph marked
    ``externalize`` is part of a separate module's public API by definition.
    """
    model = Net(layers=2)
    expected = model(X).detach()
    program = _convert(model, {"Protocol": "lora.extend"}, externalize=False)
    with TemporaryDirectory(suffix=".aimodel") as tmp:
        asset = program.save_asset(Path(tmp))
        async with asset.executable() as ai_model:
            out = await ai_model.load_function("main")(inputs={"x": NDArray(X)})
            got = {k: v.numpy() for k, v in out.items()}
            assert compare_outputs({list(got)[0]: expected}, got)


class TwoTensorBranch(nn.Module):
    """Combines a layer's input with its output -- a two-tensor boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(DIM, DIM, 1, bias=False)

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        return base_out + self.proj(x)


class LayerWithBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Conv2d(DIM, DIM, 1, bias=False)
        self.branch = TwoTensorBranch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x, self.base(x)) * 2.0


def _convert_branch(model: nn.Module, namespace: str = "group_a.stage_one"):
    """Convert `model` with `SideBranch` outlined into `namespace`."""
    args = (torch.randn(1, DIM, 1, DIM),)
    ep = torch.export.export(model, args).run_decompositions(get_decomp_table())
    converter = TorchConverter()
    converter._set_externalize_group(TwoTensorBranch.__name__, namespace)
    converter.add_exported_program(ep, entrypoint_name="main")
    return converter.to_coreai()


def test_externalize_graphs_splits_and_leaves_source_intact() -> None:
    """Extraction must yield a marked-graphs-only program *without* consuming the source.

    ``ISOLATE_EXTERNALIZED_GRAPHS`` mutates its module in place and ``AIProgram`` holds
    ``_mlir_module`` by reference, so a naive call destroys the program you still need to
    save. The helper clones first, so callers are order-independent and can inspect both
    artifacts -- which a caller's graph extraction relies on.
    """
    from coreai_torch import _externalize_graphs

    torch.manual_seed(42)
    program = _convert_branch(LayerWithBranch().eval())
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


def test_nested_graphs_survive_asset_serialization() -> None:
    """A namespaced program must serialize, not just verify in memory.

    Two things this catches that an IR check cannot: a nested ``builtin.module`` (the
    obvious choice for nesting) is unserializable, and an op created without an explicit
    location inherits whatever is ambient, which can be a location the bytecode writer
    cannot represent -- the module then verifies and prints fine but ``save_asset`` fails
    with

        Failed to serialize module to Bytecode: at #aicode.debuginfo.location_v1<
            src = <file = <filename = "-", directory = "", sha256Sum = "">, ...

    reported against the nested module op itself. So this test writes the asset and reads
    it back.
    """
    torch.manual_seed(42)
    program = _convert_branch(LayerWithBranch().eval())

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
