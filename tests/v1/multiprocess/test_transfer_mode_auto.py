# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
import argparse
import importlib.util
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.config import add_mp_server_args

SERVER_FILE = Path(__file__).resolve().parents[3] / "lmcache/v1/multiprocess/server.py"


def _make_module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def server_module(monkeypatch):
    class _BaseModule:
        def __init__(self, ctx):
            self.ctx = ctx

    class LookupModule(_BaseModule):
        pass

    class ManagementModule(_BaseModule):
        pass

    class GPUTransferModule(_BaseModule):
        pass

    class NonGPUTransferModule(_BaseModule):
        pass

    class BlendModule(_BaseModule):
        pass

    @dataclass
    class HandlerSpec:
        request_type: object
        handler: object
        pool: object

    class ThreadPoolType:
        SYNC = "sync"
        AFFINITY = "affinity"
        NORMAL = "normal"

    stubs = {
        "zmq": _make_module("zmq", Context=SimpleNamespace(instance=lambda: object())),
        "lmcache.v1.distributed.config": _make_module(
            "lmcache.v1.distributed.config",
            StorageManagerConfig=object,
            add_storage_manager_args=lambda parser: parser,
            parse_args_to_config=lambda args: None,
        ),
        "lmcache.v1.mp_observability.config": _make_module(
            "lmcache.v1.mp_observability.config",
            ObservabilityConfig=object,
            add_observability_args=lambda parser: parser,
            init_observability=lambda *args, **kwargs: None,
            parse_args_to_observability_config=lambda args: None,
        ),
        "lmcache.v1.mp_observability.trace": _make_module(
            "lmcache.v1.mp_observability.trace",
            maybe_initialize_trace_recorder=lambda *args, **kwargs: None,
        ),
        "lmcache.v1.multiprocess.engine_context": _make_module(
            "lmcache.v1.multiprocess.engine_context",
            MPCacheEngineContext=object,
        ),
        "lmcache.v1.multiprocess.engine_module": _make_module(
            "lmcache.v1.multiprocess.engine_module",
            EngineModule=object,
            HandlerSpec=HandlerSpec,
            ThreadPoolType=ThreadPoolType,
        ),
        "lmcache.v1.multiprocess.modules.gpu_transfer": _make_module(
            "lmcache.v1.multiprocess.modules.gpu_transfer",
            GPUTransferModule=GPUTransferModule,
        ),
        "lmcache.v1.multiprocess.modules.lookup": _make_module(
            "lmcache.v1.multiprocess.modules.lookup",
            LookupModule=LookupModule,
        ),
        "lmcache.v1.multiprocess.modules.management": _make_module(
            "lmcache.v1.multiprocess.modules.management",
            ManagementModule=ManagementModule,
        ),
        "lmcache.v1.multiprocess.modules.non_gpu_transfer": _make_module(
            "lmcache.v1.multiprocess.modules.non_gpu_transfer",
            NonGPUTransferModule=NonGPUTransferModule,
        ),
        "lmcache.v1.multiprocess.modules.blend": _make_module(
            "lmcache.v1.multiprocess.modules.blend",
            BlendModule=BlendModule,
        ),
        "lmcache.v1.multiprocess.mq": _make_module(
            "lmcache.v1.multiprocess.mq",
            MessageQueueServer=object,
        ),
        "lmcache.v1.multiprocess.protocol": _make_module(
            "lmcache.v1.multiprocess.protocol",
            RequestType=object,
            get_handler_type=lambda request_type: None,
            get_payload_classes=lambda request_type: [],
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "lmcache.v1.multiprocess.server_test_double",
        SERVER_FILE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_transfer_mode_arg_defaults_to_auto():
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)

    args = parser.parse_args([])

    assert args.transfer_mode == "auto"


def test_transfer_mode_arg_accepts_auto():
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)

    args = parser.parse_args(["--transfer-mode", "auto"])

    assert args.transfer_mode == "auto"


def test_build_modules_loads_both_transfer_modules_in_auto_mode(server_module):
    modules = server_module._build_modules(
        object(),
        SimpleNamespace(transfer_mode="auto", engine_type="default"),
    )

    assert [type(module).__name__ for module in modules] == [
        "LookupModule",
        "ManagementModule",
        "GPUTransferModule",
        "NonGPUTransferModule",
    ]


def test_build_modules_allows_blend_with_auto_mode(server_module):
    modules = server_module._build_modules(
        object(),
        SimpleNamespace(transfer_mode="auto", engine_type="blend"),
    )

    assert [type(module).__name__ for module in modules] == [
        "LookupModule",
        "ManagementModule",
        "GPUTransferModule",
        "NonGPUTransferModule",
        "BlendModule",
    ]


def test_build_modules_rejects_blend_with_non_gpu_mode(server_module):
    with pytest.raises(
        ValueError,
        match="Blend engine requires transfer_mode in \\{'gpu', 'auto'\\}",
    ):
        server_module._build_modules(
            object(),
            SimpleNamespace(transfer_mode="non_gpu", engine_type="blend"),
        )
