# SPDX-License-Identifier: Apache-2.0
"""Registry base classes for platform-specific backends.

Each module under this package defines one abstract base class that acts
as a capability interface.  The universal registry (``_registry.py``)
scans these modules at start-up: every class that is directly defined
here **and** subclasses :class:`abc.ABC` is treated as a *registry base
class* and gets its own slot in the 3-D lookup table
``(base_class, device_type, impl_key)``.

Current base classes:

* :class:`~lmcache.v1.platform.base.pin_memory.PinMemoryBackend` --
  host-memory pinning capability.  Provides a no-op fallback so devices
  that do not support pinning can still run without a concrete
  implementation.
"""
