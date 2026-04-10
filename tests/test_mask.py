"""Unit tests for research/transformer/Mask.py."""
import torch
import pytest

from Mask import Mask


class TestMask:
    def test_mask_shape_is_correct(self):
        m = Mask(B=2, L=5)
        assert m.mask.shape == (2, 1, 5, 5)

    def test_mask_is_upper_triangular(self):
        m = Mask(B=1, L=4)
        mask = m.mask[0, 0]  # shape (L, L)
        # Upper triangle (diagonal=1) should be True, rest False
        for i in range(4):
            for j in range(4):
                expected = j > i
                assert mask[i, j].item() == expected

    def test_mask_dtype_is_bool(self):
        m = Mask(B=1, L=3)
        assert m.mask.dtype == torch.bool

    def test_mask_on_cpu_by_default(self):
        m = Mask(B=2, L=4)
        assert m.mask.device.type == "cpu"

    def test_batch_size_one_has_correct_shape(self):
        m = Mask(B=1, L=6)
        assert m.mask.shape == (1, 1, 6, 6)

    def test_diagonal_is_false(self):
        # The causal mask should NOT mask the current position (diagonal=1 means
        # positions strictly above the diagonal are masked).
        m = Mask(B=1, L=4)
        mask = m.mask[0, 0]
        for i in range(4):
            assert mask[i, i].item() is False
