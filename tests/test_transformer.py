"""Unit tests for the Transformer model (research/transformer/Model.py).

Tests verify:
  - The model instantiates without error given valid config.
  - The forward pass returns the correct output shape: (batch, pred_len, c_out).
  - The model is deterministic in eval() mode (dropout disabled).
  - Interface.predict() uses model.eval(), not model.train().
"""
import inspect
import pytest
import torch
from argparse import Namespace

# conftest.py adds research/ to sys.path
import transformer.Model as ModelModule


# ---------------------------------------------------------------------------
# Minimal model config
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_config():
    """Smallest valid config that exercises all model components."""
    return Namespace(
        predLen=5,
        encIn=3,
        decIn=3,
        cOut=1,
        dModel=16,
        nHeads=2,
        eLayers=2,
        dLayers=1,
        dFf=32,
        factor=1,
        dropout=0.1,
    )


def _make_inputs(batch: int, seq_len: int, label_len: int, pred_len: int,
                 n_features: int):
    """Return (enc_x, enc_mark, dec_x, dec_mark) random float tensors."""
    enc_x = torch.randn(batch, seq_len, n_features)
    enc_mark = torch.randn(batch, seq_len, 3)           # month, day, weekday
    dec_x = torch.randn(batch, label_len + pred_len, n_features)
    dec_mark = torch.randn(batch, label_len + pred_len, 3)
    return enc_x, enc_mark, dec_x, dec_mark


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestModelInstantiation:

    def test_model_builds_without_error(self, small_config):
        model = ModelModule.Model(small_config)
        assert model is not None

    def test_model_is_nn_module(self, small_config):
        import torch.nn as nn
        model = ModelModule.Model(small_config)
        assert isinstance(model, nn.Module)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestForwardPassShape:

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_output_shape_is_batch_predlen_cout(self, small_config, batch_size):
        """output[0].shape must be (batch, pred_len, c_out) regardless of batch."""
        model = ModelModule.Model(small_config).eval()
        enc_x, enc_mark, dec_x, dec_mark = _make_inputs(
            batch=batch_size, seq_len=30, label_len=10,
            pred_len=small_config.predLen, n_features=small_config.encIn,
        )
        with torch.no_grad():
            output, _ = model(enc_x, enc_mark, dec_x, dec_mark)
        expected_shape = (batch_size, small_config.predLen, small_config.cOut)
        assert output.shape == expected_shape, (
            f"Expected {expected_shape}, got {output.shape}"
        )

    def test_attention_weights_are_returned(self, small_config):
        """forward() returns (predictions, attentions); attentions must not be None."""
        model = ModelModule.Model(small_config).eval()
        enc_x, enc_mark, dec_x, dec_mark = _make_inputs(
            batch=2, seq_len=30, label_len=10,
            pred_len=small_config.predLen, n_features=small_config.encIn,
        )
        with torch.no_grad():
            output, attns = model(enc_x, enc_mark, dec_x, dec_mark)
        assert attns is not None


# ---------------------------------------------------------------------------
# Determinism / dropout behaviour
# ---------------------------------------------------------------------------

class TestModelDeterminism:

    def test_eval_mode_is_deterministic(self, small_config):
        """Two identical forward passes in eval() must produce identical output."""
        model = ModelModule.Model(small_config).eval()
        enc_x, enc_mark, dec_x, dec_mark = _make_inputs(
            batch=2, seq_len=30, label_len=10,
            pred_len=small_config.predLen, n_features=small_config.encIn,
        )
        with torch.no_grad():
            out1, _ = model(enc_x, enc_mark, dec_x, dec_mark)
            out2, _ = model(enc_x, enc_mark, dec_x, dec_mark)
        assert torch.allclose(out1, out2), (
            "Model output changed between identical eval() passes — "
            "dropout may still be active during inference."
        )

    def test_predict_interface_uses_eval_mode(self):
        """Interface.predict() must call model.eval(), not model.train().

        Guards against the regression fixed in fix/critical-bugs where
        predict() called self.model.train(), enabling dropout at inference time.
        """
        import transformer.Interface as InterfaceModule
        src = inspect.getsource(InterfaceModule.Model_Interface.predict)
        assert "model.eval()" in src, (
            "Interface.predict() does not call self.model.eval(). "
            "Dropout is active during inference making predictions "
            "non-deterministic. Replace self.model.train() with "
            "self.model.eval() in Interface.predict()."
        )
