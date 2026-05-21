# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# First Party
from lmcache.v1.distributed.memory_manager import _unlink_stale_shm


def _touch_shm(shm_name: str) -> str:
    path = os.path.join("/dev/shm", shm_name.lstrip("/"))
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    return path


def test_unlink_stale_shm_allows_default_lmcache_prefix() -> None:
    shm_name = f"lmcache_l1_pool_test_{os.getpid()}"
    path = _touch_shm(shm_name)
    try:
        _unlink_stale_shm(shm_name)
        assert not os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_unlink_stale_shm_allows_configured_custom_name() -> None:
    shm_name = f"custom_l1_pool_test_{os.getpid()}"
    path = _touch_shm(shm_name)
    try:
        _unlink_stale_shm(shm_name, configured_shm_name=shm_name)
        assert not os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_unlink_stale_shm_keeps_unconfigured_custom_name() -> None:
    shm_name = f"custom_l1_pool_test_{os.getpid()}"
    path = _touch_shm(shm_name)
    try:
        _unlink_stale_shm(shm_name)
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)
