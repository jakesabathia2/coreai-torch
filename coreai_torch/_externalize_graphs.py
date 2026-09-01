# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Splitting ``externalize``-marked graphs out of an :class:`AIProgram`."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from coreai._compiler._transforms.passes import (
    CorePasses,
    GlobalOptions,
    PassEntry,
    apply_passes_sync,
)
from coreai.authoring import AIProgram

__all__ = ["_externalize_graphs"]


def _externalize_graphs(program: AIProgram) -> AIProgram:
    """Return a new :class:`AIProgram` holding only the ``externalize``-marked graphs.

    **Experimental.** Underscore-prefixed on purpose: no backwards-compatibility
    guarantee, and it may be removed once a first-class mechanism exists.

    Use with :attr:`ExternalizeSpec._graph_externalize` to produce a side artifact that
    ships next to the base asset instead of being inlined into it as constants, so its
    weights can be replaced without re-exporting the base::

        program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=...,
                externalize_modules=[
                    ExternalizeSpec(SideBranch, _graph_externalize=True,
                                    _namespace="group_a.stage_one"),
                ],
            )
            .to_coreai()
        )

        extracted = _externalize_graphs(program)
        program.save_asset(out / "model.aimodel")            # base, unchanged
        extracted._save_bytecode(out / "extracted.mlirb")    # noqa: SLF001

    The source ``program`` is **left untouched**, so the two artifacts can be written
    in either order and both can be inspected afterwards. That is worth stating
    plainly because the underlying compiler pass is destructive: it deletes every
    graph it does not keep, and :class:`AIProgram` holds its MLIR module by
    reference, so running it directly on ``program`` would consume the base program
    you still need to save. This function deep-copies first to avoid that trap.

    Two properties of the result follow from the pass and the op definition:

    * only marked graphs survive — the entrypoints are gone, so the returned program
      has no callable ``main``;
    * the ``externalize`` attribute is **stripped** from the survivors, since per the
      op definition graphs "will not retain this property" once externalized.

    Args:
        program: An :class:`AIProgram` from ``TorchConverter.to_coreai()`` containing
            at least one graph marked ``externalize``.

    Returns:
        A new :class:`AIProgram` containing only the externalized subgraphs. If no
        graph was marked, the result contains no graphs.

    """
    cloned = deepcopy(program)
    # The pass writes nothing to disk for this pipeline, but GlobalOptions requires an
    # output directory; a temporary one keeps it from touching the caller's cwd.
    with TemporaryDirectory() as tmp:
        apply_passes_sync(
            cloned._mlir_module,  # noqa: SLF001
            passes=[PassEntry.get(CorePasses.ISOLATE_EXTERNALIZED_GRAPHS)],
            options=GlobalOptions(output_directory=Path(tmp)),
        )
    return cloned
