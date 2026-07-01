# SPDX-License-Identifier: Apache-2.0
"""Base-class package for the platform abstraction layer.

Each module in this package defines exactly one abstract base class.
The universal registry (``lmcache.v1.platform._registry``) scans this
package automatically, so dropping a new ``.py`` file here with a base
class that declares a ``device_type: ClassVar[str] = ""`` attribute is
all that is needed to register a new capability.

Import each base class directly from its submodule to avoid pulling in
heavy dependencies (e.g. ``torch``) before they are needed::

    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
    from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
"""
