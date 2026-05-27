# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
import argparse
import sys
import types


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


def test_shm_name_moves_to_storage_manager_config() -> None:
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
            "",
        ]
    )

    mp_config = parse_args_to_mp_server_config(args)
    storage_config = parse_args_to_config(args)

    assert not hasattr(mp_config, "shm_name")
    assert storage_config.l1_manager_config.memory_config.shm_name == ""


def test_mp_server_parser_rejects_shm_name_argument() -> None:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)

    try:
        parser.parse_args(["--shm-name", "custom"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("--shm-name should not be accepted by MP parser")
