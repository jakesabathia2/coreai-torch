# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for ``TorchConverter._set_delegate_id``.

Pin every instance of a module class to a backend, by wrapping the ops it contributed in
``coreai.isolated_group<target>``.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
import torch.nn as nn
from coreai.runtime import NDArray

from coreai_torch import TorchConverter, get_decomp_table

from ..utils import compare_outputs

TABLE_ROWS = 16
TABLE_COLS = 4


class Lookup(nn.Module):
    """A gather off a constant table -- the shape of the RoPE/embedding lookups."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "table",
            torch.arange(TABLE_ROWS * TABLE_COLS, dtype=torch.float32).reshape(
                TABLE_ROWS, TABLE_COLS
            ),
            persistent=False,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.table[idx.to(torch.int32)]


class Scale(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


class Net(nn.Module):
    """`lookup` feeds the tail, so its ops are contiguous and its result is consumed."""

    def __init__(self) -> None:
        super().__init__()
        self.lookup = Lookup()

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return x + self.lookup(idx).sum(-1, keepdim=True)


class TwoLookups(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = Lookup()
        self.second = Lookup()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.first(idx).sum(-1) + self.second(idx).sum(-1)


class LookupOnly(nn.Module):
    """The delegated module produces the graph output directly."""

    def __init__(self) -> None:
        super().__init__()
        self.lookup = Lookup()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.lookup(idx)


class Nested(nn.Module):
    """A delegated module containing another delegated module."""

    def __init__(self) -> None:
        super().__init__()
        self.lookup = Lookup()
        self.scale = Scale(2.0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.scale(self.lookup(idx)) + 1.0


class Straddle(nn.Module):
    """Own ops on both sides of a submodule's, so its span is broken."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = Lookup()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.inner(idx + 1) * 2.0


class OuterStraddle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.straddle = Straddle()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.straddle(idx)


IDX = torch.zeros(2, dtype=torch.int64)
X = torch.ones(2, 1)


def _convert(model: nn.Module, args: tuple, delegates: dict[str, str] | None = None):
    """Export + convert, optionally with delegate ids set. Returns the AIProgram."""
    ep = torch.export.export(model, args).run_decompositions(get_decomp_table())
    converter = TorchConverter()
    for cls, target in (delegates or {}).items():
        converter._set_delegate_id(cls, target)
    converter.add_exported_program(ep, entrypoint_name="main")
    return converter.to_coreai()


def _verify(program) -> None:
    """The invariant every case must hold: the module still verifies."""
    program._mlir_module.operation.verify()


def _region_of(ir: str, target: str) -> str:
    """The text of the first isolated_group region for `target`."""
    marker = f'isolated_group<"{target}">'
    assert marker in ir, f"no {marker} in:\n{ir}"
    start = ir.index(marker)
    depth, i = 0, ir.index("{", start)
    for j in range(i, len(ir)):
        if ir[j] == "{":
            depth += 1
        elif ir[j] == "}":
            depth -= 1
            if depth == 0:
                return ir[i : j + 1]
    raise AssertionError("unbalanced region")


def test_wraps_matched_module_in_isolated_group() -> None:
    program = _convert(Net(), (X, IDX), {"Lookup": "Interpreter"})
    ir = str(program)
    assert ir.count('coreai.isolated_group<"Interpreter">') == 1
    _verify(program)


def test_no_delegate_id_emits_no_region() -> None:
    """Opt-in: without the call, conversion is untouched."""
    ir = str(_convert(Net(), (X, IDX)))
    assert "isolated_group" not in ir


def test_delegated_ops_move_into_the_region() -> None:
    """The gather belongs to Lookup, the reduce/add to Net."""
    ir = str(_convert(Net(), (X, IDX), {"Lookup": "Interpreter"}))
    region = _region_of(ir, "Interpreter")
    assert "coreai.gather_nd" in region
    assert "reduce_sum" not in region
    assert "broadcasting_add" not in region


def test_constant_used_only_inside_is_pulled_in() -> None:
    """A region carries its own tables rather than taking them as arguments."""
    ir = str(_convert(Net(), (X, IDX), {"Lookup": "Interpreter"}))
    region = _region_of(ir, "Interpreter")
    # the table constant lives inside...
    assert f"tensor<{TABLE_ROWS}x{TABLE_COLS}xf32>" in region
    # ...and the group therefore takes a single operand: the index
    header = ir[ir.index('isolated_group<"Interpreter">') :].split("{")[0]
    assert header.count("%arg") == 1, header


def test_only_externally_used_results_are_yielded() -> None:
    ir = str(_convert(Net(), (X, IDX), {"Lookup": "Interpreter"}))
    region = _region_of(ir, "Interpreter")
    yields = [ln for ln in region.splitlines() if "coreai.yield" in ln]
    assert len(yields) == 1
    assert yields[0].count(",") == 0, f"expected one yielded value: {yields[0]}"


def test_one_region_per_instance_not_per_class() -> None:
    """Two instances of the class must give two regions, not one per class."""
    program = _convert(TwoLookups(), (IDX,), {"Lookup": "Interpreter"})
    assert str(program).count('coreai.isolated_group<"Interpreter">') == 2
    _verify(program)


def test_delegated_module_producing_the_graph_output() -> None:
    """The output spec must follow the region's result, not the value now inside it."""
    program = _convert(LookupOnly(), (IDX,), {"Lookup": "Interpreter"})
    ir = str(program)
    assert ir.count('coreai.isolated_group<"Interpreter">') == 1
    _verify(program)
    # the graph output is the group's result
    output_line = next(ln for ln in ir.splitlines() if "coreai.output" in ln)
    group_result = next(
        ln.split("=")[0].strip()
        for ln in ir.splitlines()
        if 'isolated_group<"Interpreter">' in ln
    )
    assert group_result in output_line, f"{group_result!r} not in {output_line!r}"


def test_distinct_targets_for_distinct_classes() -> None:
    program = _convert(Nested(), (IDX,), {"Lookup": "Interpreter", "Scale": "CPU"})
    ir = str(program)
    assert ir.count('coreai.isolated_group<"Interpreter">') == 1
    assert ir.count('coreai.isolated_group<"CPU">') == 1
    _verify(program)


def test_unmatched_class_name_is_a_noop() -> None:
    program = _convert(Net(), (X, IDX), {"NoSuchModule": "Interpreter"})
    assert "isolated_group" not in str(program)
    _verify(program)


def test_non_contiguous_module_is_skipped_with_warning() -> None:
    """A module whose ops straddle a submodule's cannot be grouped.

    Pulling the far side into the region would move the submodule's computation with it,
    so the instance is skipped -- loudly -- and the inner module is still delegated.
    """
    with pytest.warns(UserWarning, match="not contiguous"):
        program = _convert(
            OuterStraddle(), (IDX,), {"Straddle": "CPU", "Lookup": "Interpreter"}
        )
    ir = str(program)
    assert ir.count('coreai.isolated_group<"Interpreter">') == 1
    assert 'isolated_group<"CPU">' not in ir
    _verify(program)


def test_set_delegate_id_returns_self_for_chaining() -> None:
    converter = TorchConverter()
    assert converter._set_delegate_id("Lookup", "Interpreter") is converter


def test_delegate_ids_persist_across_staged_programs() -> None:
    """One converter, two entrypoints: both get the region."""
    ep = torch.export.export(Net(), (X, IDX)).run_decompositions(get_decomp_table())
    converter = TorchConverter()
    converter._set_delegate_id("Lookup", "Interpreter")
    converter.add_exported_program(ep, entrypoint_name="first")
    converter.add_exported_program(ep, entrypoint_name="second")
    program = converter.to_coreai()
    assert str(program).count('coreai.isolated_group<"Interpreter">') == 2
    _verify(program)


def test_region_is_isolated_from_above() -> None:
    """Nothing inside may name an outer SSA value -- that is what makes it delegatable."""
    ir = str(_convert(Net(), (X, IDX), {"Lookup": "Interpreter"}))
    region = _region_of(ir, "Interpreter")
    block_args = {
        tok.strip().rstrip(":")
        for tok in region.splitlines()[1].replace("(", " ").replace(")", " ").split()
        if tok.startswith("%arg")
    }
    defined = {
        ln.split("=")[0].strip() for ln in region.splitlines() if "=" in ln
    } | block_args
    for line in region.splitlines()[2:]:
        if "=" not in line and "yield" not in line:
            continue
        rhs = line.split("=", 1)[-1]
        for tok in rhs.replace(",", " ").replace("(", " ").replace(")", " ").split():
            if tok.startswith("%"):
                name = tok.split(":")[0]
                assert name in defined, f"{name} escapes the region:\n{region}"


@pytest.mark.asyncio
async def test_delegation_preserves_numerics() -> None:
    """A delegate is a placement hint: the result must be unchanged."""
    model, args = LookupOnly(), (IDX,)
    program = _convert(model, args, {"Lookup": "Interpreter"})
    with TemporaryDirectory(suffix=".aimodel") as tmp:
        asset = program.save_asset(Path(tmp))
        async with asset.executable() as ai_model:
            fn = ai_model.load_function("main")
            out = await fn(inputs={"idx": NDArray(args[0].to(torch.int32))})
            got = {k: v.numpy() for k, v in out.items()}
            expected = {list(got)[0]: model(*args)}
            assert compare_outputs(expected, got)


# ---------------------------------------------------------------------------
# _set_entrypoint_delegate_id: the whole procedure, not one module's ops
# ---------------------------------------------------------------------------


class ConstantOnly(nn.Module):
    """A procedure whose body is a single constant, so nothing is attributed."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("table", torch.arange(TABLE_ROWS, dtype=torch.uint8))

    def forward(self) -> torch.Tensor:
        return self.table


def _convert_entrypoint_delegate(
    model: nn.Module, args: tuple, target: str | None, entrypoint: str = "main"
):
    ep = torch.export.export(model, args).run_decompositions(get_decomp_table())
    converter = TorchConverter()
    if target is not None:
        converter._set_entrypoint_delegate_id(entrypoint, target)
    converter.add_exported_program(ep, entrypoint_name=entrypoint)
    return converter.to_coreai()


def test_whole_procedure_is_wrapped_in_one_group() -> None:
    """Every op in the body moves into the region, and the graph still verifies."""
    program = _convert_entrypoint_delegate(Net(), (X, IDX), "Interpreter")
    ir = str(program)

    assert ir.count('coreai.isolated_group<"Interpreter">') == 1
    region = _region_of(ir, "Interpreter")
    # Ops from the delegated module and from its parent alike.
    assert "coreai.gather_nd" in region
    assert "reduce_sum" in region
    # The terminator stays in the graph body; the region yields to it.
    assert "coreai.output" not in region
    assert "coreai.yield" in region
    _verify(program)


def test_constant_only_procedure_is_wrapped() -> None:
    """The case attribution cannot express: no op belongs to any module.

    ``_set_delegate_id`` has nothing to match here -- a constant is materialised during
    graph setup, not attributed to a lowered node -- so it produces no region at all.
    """
    delegated = _convert_entrypoint_delegate(ConstantOnly(), (), "Interpreter")
    assert 'coreai.isolated_group<"Interpreter">' in str(delegated)
    _verify(delegated)

    by_class = _convert(ConstantOnly(), (), {"ConstantOnly": "Interpreter"})
    assert "isolated_group" not in str(by_class)


def test_no_entrypoint_delegate_emits_no_region() -> None:
    """Opt-in, and keyed by entrypoint name: another name leaves the graph untouched."""
    assert "isolated_group" not in str(
        _convert_entrypoint_delegate(Net(), (X, IDX), None)
    )

    program = _convert_entrypoint_delegate(
        Net(), (X, IDX), "Interpreter", entrypoint="main"
    )
    assert "isolated_group" in str(program)

    converter = TorchConverter()
    converter._set_entrypoint_delegate_id("some_other_procedure", "Interpreter")
    ep = torch.export.export(Net(), (X, IDX)).run_decompositions(get_decomp_table())
    converter.add_exported_program(ep, entrypoint_name="main")
    assert "isolated_group" not in str(converter.to_coreai())


@pytest.mark.asyncio
async def test_entrypoint_delegate_preserves_numerics() -> None:
    """Placement only: wrapping the whole procedure must not change the result."""
    model, args = LookupOnly(), (IDX,)
    program = _convert_entrypoint_delegate(model, args, "Interpreter")
    with TemporaryDirectory(suffix=".aimodel") as tmp:
        asset = program.save_asset(Path(tmp))
        async with asset.executable() as ai_model:
            fn = ai_model.load_function("main")
            out = await fn(inputs={"idx": NDArray(args[0].to(torch.int32))})
            got = {k: v.numpy() for k, v in out.items()}
            expected = {list(got)[0]: model(*args)}
            assert compare_outputs(expected, got)
