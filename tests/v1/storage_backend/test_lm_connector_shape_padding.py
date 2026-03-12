# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.storage_backend.connector.lm_connector import (
    _pad_shape_to_4d,
    _unpad_shape_from_4d,
)


class TestShapePadding:
    """Test shape padding/unpadding for layerwise cache support."""

    def test_pad_3d_to_4d_layerwise(self):
        """Test padding 3D layerwise shape to 4D."""
        # Layerwise shape: [2, num_tokens, hidden_dim]
        input_shape = torch.Size([2, 128, 4096])
        expected = torch.Size([2, 1, 128, 4096])
        result = _pad_shape_to_4d(input_shape, is_layerwise=True)
        assert result == expected

    def test_pad_4d_no_change_layerwise(self):
        """Test that 4D layerwise shapes are not modified."""
        # Already 4D shape (should not happen in practice, but handle gracefully)
        input_shape = torch.Size([2, 1, 128, 4096])
        result = _pad_shape_to_4d(input_shape, is_layerwise=True)
        assert result == input_shape

    def test_pad_4d_no_change_non_layerwise(self):
        """Test that 4D non-layerwise shapes are not modified."""
        # Non-layerwise shape: [2, num_layers, num_tokens, hidden_dim]
        input_shape = torch.Size([2, 32, 128, 4096])
        result = _pad_shape_to_4d(input_shape, is_layerwise=False)
        assert result == input_shape

    def test_pad_4d_with_num_layers_1_non_layerwise(self):
        """Test that 4D shapes with num_layers=1 are preserved for non-layerwise."""
        # This is the key test case: legitimate 4D tensor with num_layers=1
        input_shape = torch.Size([2, 1, 128, 4096])
        result = _pad_shape_to_4d(input_shape, is_layerwise=False)
        assert result == input_shape  # Should NOT be modified

    def test_pad_3d_non_layerwise_raises_error(self):
        """Test that 3D shapes raise error for non-layerwise cache."""
        with pytest.raises(ValueError, match="Unsupported shape dimension"):
            _pad_shape_to_4d(torch.Size([2, 128, 4096]), is_layerwise=False)

    def test_pad_invalid_dimension_layerwise(self):
        """Test that invalid dimensions raise error for layerwise."""
        with pytest.raises(ValueError, match="Unsupported shape dimension"):
            _pad_shape_to_4d(torch.Size([2, 3]), is_layerwise=True)

        with pytest.raises(ValueError, match="Unsupported shape dimension"):
            _pad_shape_to_4d(torch.Size([2, 3, 4, 5, 6]), is_layerwise=True)

    def test_unpad_layerwise_4d_to_3d(self):
        """Test unpadding layerwise 4D shape back to 3D."""
        # Padded layerwise shape with num_layers=1
        input_shape = torch.Size([2, 1, 128, 4096])
        expected = torch.Size([2, 128, 4096])
        result = _unpad_shape_from_4d(input_shape, is_layerwise=True)
        assert result == expected

    def test_unpad_non_layerwise_no_change(self):
        """Test that non-layerwise 4D shapes are not modified."""
        # Non-layerwise shape with num_layers > 1
        input_shape = torch.Size([2, 32, 128, 4096])
        result = _unpad_shape_from_4d(input_shape, is_layerwise=False)
        assert result == input_shape

    def test_unpad_non_layerwise_with_num_layers_1(self):
        """Test that 4D with num_layers=1 is preserved for non-layerwise."""
        # This is the key test case: should NOT unpad for non-layerwise
        input_shape = torch.Size([2, 1, 128, 4096])
        result = _unpad_shape_from_4d(input_shape, is_layerwise=False)
        assert result == input_shape  # Should NOT be unpacked to 3D

    def test_roundtrip_layerwise(self):
        """Test that padding and unpadding are inverse operations for layerwise."""
        original = torch.Size([2, 128, 4096])
        padded = _pad_shape_to_4d(original, is_layerwise=True)
        unpadded = _unpad_shape_from_4d(padded, is_layerwise=True)
        assert unpadded == original

    def test_roundtrip_non_layerwise(self):
        """Test that padding and unpadding are identity for non-layerwise."""
        original = torch.Size([2, 32, 128, 4096])
        padded = _pad_shape_to_4d(original, is_layerwise=False)
        unpadded = _unpad_shape_from_4d(padded, is_layerwise=False)
        assert unpadded == original
        assert padded == original  # No change in padding

    def test_roundtrip_non_layerwise_num_layers_1(self):
        """Test roundtrip for non-layerwise with num_layers=1."""
        original = torch.Size([2, 1, 128, 4096])
        padded = _pad_shape_to_4d(original, is_layerwise=False)
        unpadded = _unpad_shape_from_4d(padded, is_layerwise=False)
        assert unpadded == original
        assert padded == original  # No change

    def test_various_layerwise_shapes(self):
        """Test different layerwise cache shapes."""
        test_cases = [
            (torch.Size([2, 1, 1024]), torch.Size([2, 1, 1, 1024])),
            (torch.Size([2, 256, 8192]), torch.Size([2, 1, 256, 8192])),
            (torch.Size([2, 512, 2048]), torch.Size([2, 1, 512, 2048])),
        ]
        for input_shape, expected in test_cases:
            result = _pad_shape_to_4d(input_shape, is_layerwise=True)
            assert result == expected, f"Failed for {input_shape}"

            # Test roundtrip
            unpadded = _unpad_shape_from_4d(result, is_layerwise=True)
            assert unpadded == input_shape, f"Roundtrip failed for {input_shape}"

