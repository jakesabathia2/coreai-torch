# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for sub-byte lut indices reached through a pass-through op.

A palettized model may not hand its indices to ``lut_to_dense`` directly: IFP routes them
through a placeholder op that carries slot metadata. ``_inject_subbyte_in_lut`` packs the
indices in the state dict, which makes the *placeholder* convert to ``ui<nbits>``, so any
op passing them through has to be re-annotated to agree -- a custom op's fake kernel can
only say ``uint8``, and the converter's result type check rejects the mismatch.
"""

import pytest
import torch
import torch.nn as nn

import coreai_torch._compression.custom_layers  # noqa: F401  (registers coreai::lut_to_dense)
from coreai_torch._compression.utils import _inject_subbyte_in_lut


@torch.library.custom_op("test_subbyte_lut::ifp_constant_hint", mutates_args=())
def ifp_constant_hint(c_input: torch.Tensor) -> torch.Tensor:
    """Pass the input through, standing in for IFP's metadata-carrying placeholder."""
    return c_input.clone()


@ifp_constant_hint.register_fake  # type: ignore[misc]
def _(c_input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(c_input)


@torch.library.custom_op("test_subbyte_lut::stand_in_lut_to_dense", mutates_args=())
def stand_in_lut_to_dense(lut: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """``(lut, indices)`` order, as ``tamm_export::lut_to_dense`` declares it."""
    return torch.zeros(indices.shape, dtype=lut.dtype)


@stand_in_lut_to_dense.register_fake  # type: ignore[misc]
def _(lut: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.zeros(indices.shape, dtype=lut.dtype)


class _IndirectPalettizedModel(nn.Module):
    """Both lut and indices reach the lut op through a pass-through placeholder."""

    def __init__(self, nbits: int) -> None:
        super().__init__()
        # The palette count sits at lut.shape[-2], and the layout is recognised by rank:
        # lut.ndim == indices.ndim + 2.
        self.register_buffer(
            "palettized_lut",
            torch.ones((1, 1, 2**nbits, 2), dtype=torch.float16),
        )
        self.register_buffer(
            "palettized_indices",
            torch.randint(0, 2**nbits, (8, 4), dtype=torch.uint8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lut = torch.ops.test_subbyte_lut.ifp_constant_hint(self.palettized_lut)
        indices = torch.ops.test_subbyte_lut.ifp_constant_hint(self.palettized_indices)
        dense = torch.ops.test_subbyte_lut.stand_in_lut_to_dense(lut, indices)
        return dense + x


def _nodes_by_op(program: torch.export.ExportedProgram, op_name: str) -> list:
    return [
        node
        for node in program.graph.nodes
        if node.op == "call_function"
        and hasattr(node.target, "name")
        and node.target.name().endswith(op_name)
    ]


@pytest.mark.parametrize("nbits", [2, 4])
def test_passthrough_indices_are_reannotated_to_the_packed_dtype(nbits: int) -> None:
    """The op passing the indices through reports ``ui<nbits>``, matching the placeholder.

    Without this the pass-through node keeps torch's ``uint8`` while the placeholder it
    reads converts to ``ui<nbits>``, and lowering the pass-through fails with
    ``dtype ui2 vs ui8``.
    """
    model = _IndirectPalettizedModel(nbits).eval()
    program = torch.export.export(model, (torch.zeros((8, 4), dtype=torch.float16),))

    hints = _nodes_by_op(program, "ifp_constant_hint")
    assert len(hints) == 2, (
        "expected one pass-through for the lut and one for the indices"
    )
    lut_hint, indices_hint = hints
    assert indices_hint.meta["val"].dtype == torch.uint8, (
        "precondition: torch says uint8"
    )

    _inject_subbyte_in_lut(program)

    expected = getattr(torch, f"uint{nbits}")
    assert indices_hint.meta["val"].dtype == expected
    # Shape is preserved: only the element type is restated.
    assert tuple(indices_hint.meta["val"].shape) == (8, 4)
    # The lut is not indices; re-annotating it would corrupt the palette table.
    assert lut_hint.meta["val"].dtype == torch.float16


@pytest.mark.parametrize("nbits", [2, 4])
def test_indices_are_packed_in_the_state_dict(nbits: int) -> None:
    """The indices buffer is repacked, which is what makes the placeholder ui<nbits>."""
    model = _IndirectPalettizedModel(nbits).eval()
    program = torch.export.export(model, (torch.zeros((8, 4), dtype=torch.float16),))

    _inject_subbyte_in_lut(program)

    packed = program.state_dict["palettized_indices"]
    assert packed.dtype == getattr(torch, f"uint{nbits}")
    assert program.state_dict["palettized_lut"].dtype == torch.float16


def test_directly_wired_lut_indices_are_left_alone() -> None:
    """With no pass-through there is nothing to re-annotate, and nothing is touched.

    ``coreai::lut_to_dense`` takes its operands by name, so this is the common path; the
    re-annotation must not reach into it.
    """

    class DirectModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "indices", torch.randint(0, 4, (2, 3), dtype=torch.uint8)
            )
            self.register_buffer("lut", torch.ones((1, 1, 4, 1), dtype=torch.float16))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dense = torch.ops.coreai.lut_to_dense(
                indices=self.indices, lut=self.lut, axis=0
            )
            return dense + x

    model = DirectModel().eval()
    program = torch.export.export(model, (torch.zeros((2, 3), dtype=torch.float16),))

    _inject_subbyte_in_lut(program)

    # 4 palettes -> 2 bits, applied to the state dict as before.
    assert program.state_dict["indices"].dtype == torch.uint2
