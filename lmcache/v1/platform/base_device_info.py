# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility re-export shim for :class:`DeviceInfo`.

The canonical definition has moved to
:mod:`lmcache.v1.platform.base.device_info`.  This module re-exports
:class:`DeviceInfo` so that existing imports continue to resolve without
modification.
"""

# First Party
from lmcache.v1.platform.base.device_info import DeviceInfo  # noqa: F401
