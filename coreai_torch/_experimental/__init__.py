# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Experimental additions, with no backwards-compatibility guarantee.

Everything here is underscore-prefixed on purpose and may change or disappear once a
first-class equivalent exists.
"""

from ._delegate import (
    module_instance_key,
    outline_ops_into_graph,
    wrap_ops_in_isolated_group,
)
from ._externalize_graphs import _externalize_graphs

__all__ = [
    "_externalize_graphs",
    "module_instance_key",
    "outline_ops_into_graph",
    "wrap_ops_in_isolated_group",
]
