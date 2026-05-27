# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
import argparse
import sys
import types

# Third Party
import pytest


def _install_config_import_stubs() -> None:
    l2_pkg = types.ModuleType("lmcache.v1.distributed.l2_adapters")
    l2_config_mod = types.ModuleType("lmcache.v1.distributed.l2_adapters.config")

    @dataclass
    class L2AdaptersConfig:
        adapters: list[object] = field(default_factory=list)

    def add_l2_adapters_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def parse_args_to_l2_adapters_config(_args: argparse.Namespace) -> L2AdaptersConfig:
        return L2AdaptersConfig([])

    l2_config_mod.L2AdaptersConfig = L2AdaptersConfig
    l2_config_mod.add_l2_adapters_args = add_l2_adapters_args
    l2_config_mod.parse_args_to_l2_adapters_config = parse_args_to_l2_adapters_config

    storage_controllers_pkg = types.ModuleType(
        "lmcache.v1.distributed.storage_controllers"
    )
    prefetch_policy_mod = types.ModuleType(
        "lmcache.v1.distributed.storage_controllers.prefetch_policy"
    )
    prefetch_policy_mod.get_registered_prefetch_policies = lambda: ["default"]
    store_policy_mod = types.ModuleType(
        "lmcache.v1.distributed.storage_controllers.store_policy"
    )
    store_policy_mod.get_registered_store_policies = lambda: ["default"]

    sys.modules["lmcache.v1.distributed.l2_adapters"] = l2_pkg
    sys.modules["lmcache.v1.distributed.l2_adapters.config"] = l2_config_mod
    sys.modules["lmcache.v1.distributed.storage_controllers"] = (
        storage_controllers_pkg
    )
    sys.modules[
        "lmcache.v1.distributed.storage_controllers.prefetch_policy"
    ] = prefetch_policy_mod
    sys.modules["lmcache.v1.distributed.storage_controllers.store_policy"] = (
        store_policy_mod
    )


_install_config_import_stubs()

# First Party
from lmcache.v1.distributed.config import add_storage_manager_args, parse_args_to_config
from lmcache.v1.multiprocess.config import add_mp_server_args, parse_args_to_mp_server_config


@pytest.mark.parametrize("shm_name", ["", "custom_name"])
def test_shm_name_moves_to_storage_manager_config(shm_name: str) -> None:
    """Verify explicit --shm-name values populate storage config only."""
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    add_storage_manager_args(parser)

    args = parser.parse_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--shm-name",
            shm_name,
        ]
    )

    mp_config = parse_args_to_mp_server_config(args)
    storage_config = parse_args_to_config(args)

    assert not hasattr(mp_config, "shm_name")
    assert storage_config.l1_manager_config.memory_config.shm_name == shm_name


def test_storage_manager_parser_keeps_default_shm_name_when_flag_omitted() -> None:
    """Verify omitting --shm-name preserves the L1 memory config default."""
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    add_storage_manager_args(parser)

    args = parser.parse_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
        ]
    )

    storage_config = parse_args_to_config(args)

    assert storage_config.l1_manager_config.memory_config.shm_name.startswith(
        "lmcache_l1_pool_"
    )


def test_mp_server_parser_rejects_shm_name_argument() -> None:
    """Verify the MP-only parser rejects --shm-name as an unknown argument."""
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--shm-name", "custom"])
    assert exc_info.value.code == 2
