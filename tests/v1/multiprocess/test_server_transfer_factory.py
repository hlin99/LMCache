# SPDX-License-Identifier: Apache-2.0
# Standard
import sys
from unittest.mock import MagicMock, patch


def _stub_optional_modules() -> dict[str, MagicMock]:
    return {
        "aiohttp": MagicMock(),
        "cpuinfo": MagicMock(),
        "cupy": MagicMock(),
        "lmcache.c_ops": MagicMock(),
        "lmcache.native_storage_ops": MagicMock(),
        "prometheus_client": MagicMock(),
        "psutil": MagicMock(),
    }


def test_create_transfer_strategy_selects_expected_mode() -> None:
    """Ensure the factory selects pickle or SHM based on the SHM pool config."""
    with patch.dict(sys.modules, _stub_optional_modules()):
        # First Party
        from lmcache.v1.multiprocess.modules.server_transfer import (
            PickleTransferStrategy,
            ShmTransferStrategy,
            create_transfer_strategy,
        )

        pickle_strategy = create_transfer_strategy(
            MagicMock(),
            shm_name="",
            pool_size=0,
            pending_writes={},
            pending_reads={},
            pending_lock=MagicMock(),
            transfer_key_factory=lambda key, instance_id: (instance_id, key),
        )

        shm_strategy = create_transfer_strategy(
            MagicMock(),
            shm_name="lmcache_l1_pool_test",
            pool_size=1024,
            pending_writes={},
            pending_reads={},
            pending_lock=MagicMock(),
            transfer_key_factory=lambda key, instance_id: (instance_id, key),
        )

    assert isinstance(pickle_strategy, PickleTransferStrategy)
    assert isinstance(shm_strategy, ShmTransferStrategy)
