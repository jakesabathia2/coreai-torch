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
    """Split the ``externalize``-marked graphs into their own :class:`AIProgram`.

    Deep-copies first, so the source program survives and callers are order-independent:
    the underlying pass mutates its module in place and ``AIProgram`` holds it by
    reference. The ``externalize`` attribute is stripped from the graphs it keeps.
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
