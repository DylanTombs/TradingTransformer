"""Unit tests for research/transformer/Metrics.py."""
import numpy as np
import pytest

from Metrics import RSE, CORR, MAE, MSE, RMSE, MAPE, MSPE, metric


def _arrays(pred_vals, true_vals):
    return np.array(pred_vals, dtype=float), np.array(true_vals, dtype=float)


class TestMAE:
    def test_perfect_prediction_is_zero(self):
        p, t = _arrays([1, 2, 3], [1, 2, 3])
        assert MAE(p, t) == 0.0

    def test_constant_error(self):
        p, t = _arrays([2, 3, 4], [1, 2, 3])
        assert MAE(p, t) == pytest.approx(1.0)

    def test_symmetric_errors(self):
        p, t = _arrays([0, 2], [1, 1])
        assert MAE(p, t) == pytest.approx(1.0)


class TestMSE:
    def test_perfect_prediction_is_zero(self):
        p, t = _arrays([1, 2, 3], [1, 2, 3])
        assert MSE(p, t) == 0.0

    def test_unit_error_gives_one(self):
        p, t = _arrays([0, 0], [1, 1])
        assert MSE(p, t) == pytest.approx(1.0)

    def test_larger_errors_penalised_quadratically(self):
        p1, t1 = _arrays([0], [1])
        p2, t2 = _arrays([0], [2])
        assert MSE(p2, t2) > MSE(p1, t1)


class TestRMSE:
    def test_perfect_prediction_is_zero(self):
        p, t = _arrays([5, 5], [5, 5])
        assert RMSE(p, t) == pytest.approx(0.0)

    def test_rmse_is_sqrt_of_mse(self):
        p, t = _arrays([1, 3], [2, 2])
        assert RMSE(p, t) == pytest.approx(np.sqrt(MSE(p, t)))


class TestMAPE:
    def test_perfect_prediction_is_zero(self):
        p, t = _arrays([10, 20], [10, 20])
        assert MAPE(p, t) == pytest.approx(0.0)

    def test_fifty_percent_error(self):
        p, t = _arrays([1.5], [1.0])
        assert MAPE(p, t) == pytest.approx(0.5)


class TestMSPE:
    def test_perfect_prediction_is_zero(self):
        p, t = _arrays([3, 6], [3, 6])
        assert MSPE(p, t) == pytest.approx(0.0)

    def test_fifty_percent_error_squared(self):
        p, t = _arrays([1.5], [1.0])
        assert MSPE(p, t) == pytest.approx(0.25)


class TestRSE:
    def test_perfect_prediction_is_zero(self):
        p, t = _arrays([1, 2, 3], [1, 2, 3])
        assert RSE(p, t) == pytest.approx(0.0)

    def test_rse_is_non_negative(self):
        p, t = _arrays([1, 2, 3], [2, 3, 4])
        assert RSE(p, t) >= 0.0


class TestCORR:
    def test_perfect_correlation(self):
        # CORR expects 2-D inputs: (n_samples, n_features)
        arr = np.array([[1.0], [2.0], [3.0]])
        result = CORR(arr, arr)
        assert float(np.mean(result)) == pytest.approx(1.0, abs=1e-6)

    def test_returns_finite_value(self):
        p = np.array([[1.0], [2.0], [3.0], [4.0]])
        t = np.array([[4.0], [3.0], [2.0], [1.0]])
        assert np.isfinite(CORR(p, t))


class TestMetricBundle:
    def test_returns_five_values(self):
        p, t = _arrays([1, 2, 3], [1, 2, 3])
        result = metric(p, t)
        assert len(result) == 5

    def test_perfect_prediction_all_zeros(self):
        p, t = _arrays([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        mae, mse, rmse, mape, mspe = metric(p, t)
        assert mae == pytest.approx(0.0)
        assert mse == pytest.approx(0.0)
        assert rmse == pytest.approx(0.0)
        assert mape == pytest.approx(0.0)
        assert mspe == pytest.approx(0.0)

    def test_values_are_consistent_with_individual_functions(self):
        p, t = _arrays([1, 3, 5], [2, 2, 4])
        mae, mse, rmse, mape, mspe = metric(p, t)
        assert mae == pytest.approx(MAE(p, t))
        assert mse == pytest.approx(MSE(p, t))
        assert rmse == pytest.approx(RMSE(p, t))
        assert mape == pytest.approx(MAPE(p, t))
        assert mspe == pytest.approx(MSPE(p, t))
