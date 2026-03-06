# SPDX-License-Identifier: Apache-2.0
"""
Tests for lmcache.v1.compute.models.utils.infer_model_from_vllm.
"""

# Standard
from unittest.mock import MagicMock, patch
import sys
import types

# Third Party
import pytest


def _stub_vllm_modules():
    """Insert minimal vllm stub modules so imports don't fail in test env."""
    stub_names = [
        "vllm",
        "vllm.attention",
        "flashinfer",
    ]
    for name in stub_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


_stub_vllm_modules()

# First Party
from lmcache.v1.compute.models.utils import infer_model_from_vllm  # noqa: E402


def _make_mock_vllm_model(class_name: str):
    """Return an instance whose ``type().__name__`` equals *class_name*.

    Uses a dynamically created class so ``type(instance).__name__`` reports
    the desired name as ``infer_model_from_vllm`` relies on.
    """
    dynamic_cls = type(class_name, (), {})
    return dynamic_cls()


@pytest.mark.parametrize(
    "class_name,expected_adapter",
    [
        ("LlamaForCausalLM", "LMCLlamaModel"),
        ("Qwen2ForCausalLM", "LMCLlamaModel"),
        ("DeepseekForCausalLM", "LMCLlamaModel"),
        ("Qwen3ForCausalLM", "LMCQwen3Model"),
    ],
)
def test_infer_model_from_vllm_returns_correct_adapter(
    class_name: str, expected_adapter: str
):
    """infer_model_from_vllm should return the right adapter for each model class."""
    mock_model = _make_mock_vllm_model(class_name)
    mock_blender = MagicMock()
    mock_llama_cls = MagicMock()
    mock_qwen3_cls = MagicMock()

    llama_module = "lmcache.v1.compute.models.llama"
    qwen3_module = "lmcache.v1.compute.models.qwen3"

    with (
        patch.dict(
            sys.modules,
            {
                llama_module: types.SimpleNamespace(LMCLlamaModel=mock_llama_cls),
                qwen3_module: types.SimpleNamespace(LMCQwen3Model=mock_qwen3_cls),
            },
        ),
    ):
        infer_model_from_vllm(mock_model, mock_blender, enable_sparse=False)

        if expected_adapter == "LMCLlamaModel":
            mock_llama_cls.assert_called_once_with(mock_model, mock_blender, False)
            mock_qwen3_cls.assert_not_called()
        else:
            mock_qwen3_cls.assert_called_once_with(mock_model, mock_blender, False)
            mock_llama_cls.assert_not_called()


def test_infer_model_from_vllm_unsupported_raises():
    """infer_model_from_vllm should raise NotImplementedError for unknown models."""
    mock_model = _make_mock_vllm_model("SomeUnknownForCausalLM")
    mock_blender = MagicMock()

    with pytest.raises(NotImplementedError, match="SomeUnknownForCausalLM"):
        infer_model_from_vllm(mock_model, mock_blender)
