# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Group the ops one submodule contributed, once the graph body is built.

Two groupings share the same machinery -- find the ops a matched module produced, work
out what crosses their boundary, and move them somewhere:

* :func:`wrap_ops_in_isolated_group` -> ``coreai.isolated_group<target>`` (delegation);
* :func:`outline_ops_into_graph` -> a separate ``coreai.graph`` + ``coreai.invoke``
  (externalization).

Both are driven by naming an ``nn.Module`` class: every op ``nn_module_stack`` attributes
to a matched instance is collected, and the run is moved somewhere as a unit.
"""

from __future__ import annotations

from collections.abc import Container

import coreai._compiler.dialects.coreai as coreai
from coreai._compiler.ir import InsertionPoint, OpView, Value
from torch.fx import Node

__all__ = [
    "module_instance_key",
    "outline_ops_into_graph",
    "wrap_ops_in_isolated_group",
]

_CONSTANT = "coreai.constant"


def module_instance_key(
    node: Node, targets: Container[str] | None = None
) -> tuple[str, str] | None:
    """``(module path, class name)`` for an FX node, or None if unattributed.

    Keyed on the path as well as the class so that two instances of the same class get
    one group each -- the legacy importer emitted one per call site, not one per class.

    With ``targets``, the match is against the **outermost** enclosing module whose class
    is named there, rather than the innermost. That is what makes a module with children
    groupable: an adapter's projections are attributed to their own ``Linear``, so keying
    on the leaf would give the adapter only the ops it performed *between* its children's
    and report them, correctly, as non-contiguous. Matching an ancestor collects the
    whole subtree, which is what "every op this module contributed" has to mean.
    """
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return None
    entries = [(str(path), str(cls).rsplit(".", 1)[-1]) for path, cls in stack.values()]
    if targets is None:
        return entries[-1]
    return next((entry for entry in entries if entry[1] in targets), None)


def _boundary_of(
    ops: list[OpView],
) -> tuple[list[OpView], list[Value], list[Value]]:
    """What crosses the boundary of ``ops``: ``(constants, inputs, outputs)``.

    ``inputs`` are values defined outside and used inside, ``outputs`` those defined
    inside and used outside. ``constants`` are ``coreai.constant`` ops used *only* by
    ``ops`` -- they come along rather than being passed in, which is what the legacy
    groups look like (they carry their own tables instead of taking them as arguments).
    """
    inner = {op.operation for op in ops}
    produced: set[Value] = {result for op in ops for result in op.results}

    def is_inner(op: object) -> bool:
        return getattr(op, "operation", op) in inner

    constants: list[OpView] = []
    seen_constants: set[object] = set()
    inputs: list[Value] = []
    for op in ops:
        for operand in op.operands:
            if operand in produced or operand in inputs:
                continue
            owner = operand.owner
            if getattr(owner, "name", None) == _CONSTANT and all(
                is_inner(use.owner) for use in operand.uses
            ):
                if owner.operation not in seen_constants:
                    seen_constants.add(owner.operation)
                    constants.append(owner)
                produced.add(operand)
                continue
            inputs.append(operand)

    outputs = [
        result
        for op in ops
        for result in op.results
        if any(not is_inner(use.owner) for use in result.uses)
    ]
    return constants, inputs, outputs


def wrap_ops_in_isolated_group(
    ops: list[OpView], target: str
) -> tuple[OpView, dict[Value, Value]] | None:
    """Move ``ops`` into an ``isolated_group<target>`` region, in place.

    ``ops`` must be in block order and share a block. Returns ``(group, remap)`` where
    ``remap`` maps each yielded inner value to the group result that replaced it outside
    the region -- callers holding those values (e.g. a graph-output map) must follow it.
    Returns None if ``ops`` is empty.

    ``coreai.isolated_group`` is ``IsolatedFromAbove``, so nothing inside may reference
    an outer value: every value defined outside and used inside becomes a region input.
    The exception is a ``coreai.constant`` used *only* by ``ops`` -- that moves into the
    region instead, so a group carries its own tables rather than taking them as
    arguments.
    """
    if not ops:
        return None

    constants, inputs, outputs = _boundary_of(ops)
    inner = {op.operation for op in ops}

    def is_inner(op: object) -> bool:
        return getattr(op, "operation", op) in inner

    first = ops[0].operation
    with first.context, first.location:
        with InsertionPoint(first):
            group = coreai.IsolatedGroupOp(
                [value.type for value in outputs], inputs, target
            )
        region_block = group.regions[0].blocks.append(*[v.type for v in inputs])

        # Constants first, so the region reads tables-then-compute.
        for op in [*constants, *ops]:
            region_block.append(op.operation)

        remap = dict(zip(inputs, region_block.arguments))
        for op in [*constants, *ops]:
            for i, operand in enumerate(op.operands):
                if operand in remap:
                    op.operands[i] = remap[operand]

        # Redirect the outside consumers *before* creating the yield, so the yield's own
        # use of each value is not itself redirected.
        for value, result in zip(outputs, group.results):
            for use in list(value.uses):
                if not is_inner(use.owner):
                    use.owner.operands[use.operand_number] = result

        with InsertionPoint(region_block):
            coreai.yield_(outputs)

    return group, dict(zip(outputs, group.results))


def outline_ops_into_graph(
    ops: list[OpView],
    name: str,
    *,
    callee: str | None = None,
    externalize: bool = True,
) -> coreai.GraphOp | None:
    """Move ``ops`` into a separate ``coreai.graph`` and call it instead, in place.

    ``ops`` must be in block order and share a block. Returns the new graph, or None if
    ``ops`` is empty. Consumers of the outlined values are re-pointed at the call, so a
    parent graph whose output was one of them needs no further fixing up.

    The boundary is wherever the ops happen to stop, and arguments are emitted unnamed:
    attribution has no parameter names to draw on.

    The graph lands at module top level. Pass ``callee`` when it will be moved into a
    namespace afterwards, so the call names the qualified path it will need
    (``"@ns::@inner::@name"``) rather than the flat symbol it has now.
    """
    if not ops:
        return None

    constants, inputs, outputs = _boundary_of(ops)
    inner = {op.operation for op in ops}
    first = ops[0].operation

    module_op = first
    while module_op.parent is not None:
        module_op = module_op.parent
    module_block = module_op.regions[0].blocks[0]

    with first.context, first.location:
        with InsertionPoint(module_block):
            graph_op = coreai.GraphOp(
                name=name,
                input_types=[value.type for value in inputs],
                result_types=[value.type for value in outputs],
                loc=first.location,
                no_inline=True,
                externalize=externalize,
            )

        # The call takes the outlined ops' place, so it is built *before* they move --
        # once they are gone there is nothing left in this block to anchor an insertion
        # point to, and their operands have become block arguments.
        with InsertionPoint(first):
            results = coreai.invoke(
                results=[value.type for value in outputs],
                callee=callee if callee is not None else graph_op.symbol_name,
                operands=inputs,
                loc=first.location,
            )

        for value, result in zip(outputs, results):
            for use in list(value.uses):
                if use.owner.operation not in inner:
                    use.owner.operands[use.operand_number] = result

            # Constants first, so the body reads tables-then-compute.
        for op in [*constants, *ops]:
            graph_op.entry_block.append(op.operation)

        remap = dict(zip(inputs, graph_op.arguments))
        for op in [*constants, *ops]:
            for i, operand in enumerate(op.operands):
                if operand in remap:
                    op.operands[i] = remap[operand]

        with InsertionPoint(graph_op.entry_block):
            coreai.output(outputs)

    return graph_op
