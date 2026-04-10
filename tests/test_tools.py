"""Unit tests for research/transformer/Tools.py (EarlyStopping, adjustLearningRate)."""
import os
import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from Tools import EarlyStopping, adjustLearningRate


# ---------------------------------------------------------------------------
# adjustLearningRate
# ---------------------------------------------------------------------------

class TestAdjustLearningRate:
    def _optimizer(self, lr=0.1):
        model = torch.nn.Linear(2, 1)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_lr_is_halved_each_epoch(self):
        opt = self._optimizer(lr=0.1)
        args = Namespace(learningRate=0.1)
        adjustLearningRate(opt, 2, args)
        # epoch=2 → lr * 0.5^((2-1)//1) = 0.1 * 0.5 = 0.05
        assert opt.param_groups[0]["lr"] == pytest.approx(0.05)

    def test_epoch_one_keeps_base_lr(self):
        opt = self._optimizer(lr=0.01)
        args = Namespace(learningRate=0.01)
        adjustLearningRate(opt, 1, args)
        # epoch=1 → lr * 0.5^0 = lr
        assert opt.param_groups[0]["lr"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStoppingInit:
    def test_not_triggered_at_construction(self):
        es = EarlyStopping(patience=3)
        assert es.earlyStop is False

    def test_counter_starts_at_zero(self):
        es = EarlyStopping(patience=5)
        assert es.counter == 0

    def test_best_score_is_none_initially(self):
        es = EarlyStopping()
        assert es.bestScore is None


class TestEarlyStoppingCall:
    def _make_es(self, patience=3):
        return EarlyStopping(patience=patience)

    def _mock_model(self):
        m = MagicMock()
        m.state_dict.return_value = {}
        return m

    def test_first_call_sets_best_score(self):
        es = self._make_es()
        with patch("torch.save"):
            es(0.5, self._mock_model(), "/tmp")
        assert es.bestScore == pytest.approx(-0.5)

    def test_improvement_resets_counter(self):
        es = self._make_es(patience=3)
        with patch("torch.save"):
            es(1.0, self._mock_model(), "/tmp")
            es(0.8, self._mock_model(), "/tmp")  # improvement
        assert es.counter == 0

    def test_no_improvement_increments_counter(self):
        es = self._make_es(patience=5)
        with patch("torch.save"):
            es(0.5, self._mock_model(), "/tmp")
            es(0.6, self._mock_model(), "/tmp")  # worse
            es(0.7, self._mock_model(), "/tmp")  # worse
        assert es.counter == 2

    def test_triggers_after_patience_exceeded(self):
        es = self._make_es(patience=2)
        with patch("torch.save"):
            es(0.5, self._mock_model(), "/tmp")
            es(0.6, self._mock_model(), "/tmp")
            es(0.7, self._mock_model(), "/tmp")
        assert es.earlyStop is True

    def test_does_not_trigger_before_patience_exceeded(self):
        es = self._make_es(patience=3)
        with patch("torch.save"):
            es(0.5, self._mock_model(), "/tmp")
            es(0.6, self._mock_model(), "/tmp")
        assert es.earlyStop is False


class TestEarlyStoppingSaveCheckpoint:
    def test_saves_checkpoint_to_correct_path(self):
        es = EarlyStopping()
        model = MagicMock()
        model.state_dict.return_value = {"w": torch.tensor(1.0)}

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch.save") as mock_save:
                es.saveCheckpoint(0.3, model, tmpdir)
                expected_path = os.path.join(tmpdir, "checkpoint.pth")
                mock_save.assert_called_once_with(
                    model.state_dict(), expected_path
                )

    def test_updates_val_loss_min(self):
        es = EarlyStopping()
        model = MagicMock()
        model.state_dict.return_value = {}
        with patch("torch.save"):
            es.saveCheckpoint(0.42, model, "/tmp")
        assert es.valLossMin == pytest.approx(0.42)
