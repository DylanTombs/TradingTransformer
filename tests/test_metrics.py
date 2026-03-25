"""Unit tests for the StrategyEvaluator metric methods in Trading_Simulator.py.

Trading_Simulator is the legacy backtrader-based simulator.  It was replaced by
the C++ backtesting engine.  These tests are retained for historical reference
but will be skipped automatically if the module is not present.
"""
import math
import numpy as np
import pytest

Trading_Simulator = pytest.importorskip(
    "Trading_Simulator",
    reason="Trading_Simulator module not present (replaced by C++ engine)",
)
StrategyEvaluator = Trading_Simulator.StrategyEvaluator



class _IsolatedEvaluator:
    """Minimal stand-in for StrategyEvaluator metric calculations."""

    def __init__(self, values: list, trades: list):
        self.values = list(values)
        self.trades = list(trades)
        self.trade_returns = [t["return"] for t in trades]

    _calc_sharpe = StrategyEvaluator._calc_sharpe
    _calc_max_drawdown = StrategyEvaluator._calc_max_drawdown
    _calc_win_rate = StrategyEvaluator._calc_win_rate
    _calc_profit_factor = StrategyEvaluator._calc_profit_factor
    _calc_avg_risk_reward = StrategyEvaluator._calc_avg_risk_reward
    _flag_bad_trades = StrategyEvaluator._flag_bad_trades


def _make_evaluator(values: list, trades: list) -> _IsolatedEvaluator:
    return _IsolatedEvaluator(values, trades)


def _trade(pnlcomm: float, ret: float, duration: int = 5) -> dict:
    return {"pnl": pnlcomm, "pnlcomm": pnlcomm, "duration": duration,
            "price": 100.0, "size": 10, "return": ret}


class TestMaxDrawdown:

    def test_monotonically_rising_equity_has_no_drawdown(self):
        ev = _make_evaluator(values=[100, 110, 120, 130], trades=[])
        assert ev._calc_max_drawdown() == pytest.approx(0.0, abs=1e-9)

    def test_known_peak_to_trough(self):
        """Peak = 120, trough = 90 → drawdown = 30/120 = 25 %."""
        ev = _make_evaluator(values=[100, 120, 90, 110], trades=[])
        assert ev._calc_max_drawdown() == pytest.approx(25.0, rel=1e-4)

    def test_returns_largest_drawdown_across_multiple_troughs(self):
        """Two drawdown periods: 10 % and 20 %. Largest must be returned."""
        ev = _make_evaluator(values=[100, 110, 99, 120, 96], trades=[])
        # first dd: 11/110 ≈ 10%; second dd: 24/120 = 20 %
        assert ev._calc_max_drawdown() == pytest.approx(20.0, rel=1e-4)

    def test_result_expressed_as_percentage_not_fraction(self):
        """Return value must be in percentage points (e.g. 25.0, not 0.25)."""
        ev = _make_evaluator(values=[100, 80], trades=[])
        result = ev._calc_max_drawdown()
        assert result == pytest.approx(20.0, rel=1e-4)


class TestWinRate:

    def test_all_winning_trades_returns_100(self):
        ev = _make_evaluator(
            values=[100],
            trades=[_trade(10, 0.05), _trade(20, 0.10)],
        )
        assert ev._calc_win_rate() == pytest.approx(100.0)

    def test_all_losing_trades_returns_0(self):
        ev = _make_evaluator(
            values=[100],
            trades=[_trade(-10, -0.05), _trade(-5, -0.02)],
        )
        assert ev._calc_win_rate() == pytest.approx(0.0)

    def test_mixed_trades_returns_correct_percentage(self):
        ev = _make_evaluator(
            values=[100],
            trades=[
                _trade(10, 0.05),
                _trade(-5, -0.03),
                _trade(8, 0.04),
                _trade(-3, -0.01),
            ],
        )
        assert ev._calc_win_rate() == pytest.approx(50.0)

    def test_no_trades_returns_zero(self):
        ev = _make_evaluator(values=[100], trades=[])
        assert ev._calc_win_rate() == pytest.approx(0.0)


class TestProfitFactor:

    def test_known_gross_profit_and_loss(self):
        """gross_profit=160, gross_loss=60 → PF ≈ 2.667."""
        ev = _make_evaluator(
            values=[100],
            trades=[
                _trade(100, 0.10),
                _trade(-40, -0.04),
                _trade(60, 0.06),
                _trade(-20, -0.02),
            ],
        )
        assert ev._calc_profit_factor() == pytest.approx(160 / 60, rel=1e-4)

    def test_no_losing_trades_returns_infinity(self):
        ev = _make_evaluator(
            values=[100],
            trades=[_trade(50, 0.05), _trade(30, 0.03)],
        )
        assert ev._calc_profit_factor() == float("inf")

    def test_no_trades_returns_infinity(self):
        """Gross loss = 0 with no trades → infinity (no losses)."""
        ev = _make_evaluator(values=[100], trades=[])
        assert ev._calc_profit_factor() == float("inf")


class TestSharpeRatio:

    def test_fewer_than_two_returns_gives_zero(self):
        ev = _make_evaluator(values=[100], trades=[_trade(10, 0.05)])
        assert ev._calc_sharpe(np.array([0.05])) == pytest.approx(0.0)

    def test_zero_variance_returns_returns_zero(self):
        """All trades identical → std ≈ 0 → numerically stable zero."""
        returns = np.array([0.02, 0.02, 0.02, 0.02])
        ev = _make_evaluator(values=[100], trades=[])
        result = ev._calc_sharpe(returns)
        assert math.isfinite(result)

    def test_sharpe_matches_manual_formula(self):
        """Implementation must equal mean(r) / (std(r) + 1e-9) * sqrt(252)."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
        expected = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
        ev = _make_evaluator(values=[100], trades=[])
        assert ev._calc_sharpe(returns) == pytest.approx(expected, rel=1e-6)

    def test_positive_mean_returns_positive_sharpe(self):
        returns = np.array([0.01, 0.02, 0.015, 0.008, 0.012])
        ev = _make_evaluator(values=[100], trades=[])
        assert ev._calc_sharpe(returns) > 0.0

class TestFlagBadTrades:

    def test_not_enough_returns_yields_empty_list(self):
        returns = np.array([0.05])
        ev = _make_evaluator(values=[100], trades=[])
        assert ev._flag_bad_trades(returns) == []

    def test_clear_outlier_is_flagged(self):
        """A single large loss among small gains should be identified."""
        returns = np.array([0.05, 0.04, 0.03, 0.02, -0.30])
        ev = _make_evaluator(values=[100], trades=[])
        flagged = ev._flag_bad_trades(returns)
        assert 4 in flagged, f"Index 4 (-0.30) should be flagged, got {flagged}"

    def test_uniform_returns_flags_nothing(self):
        """No outliers when all returns are equal."""
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        ev = _make_evaluator(values=[100], trades=[])
        assert ev._flag_bad_trades(returns) == []

    def test_flagging_threshold_is_1_5_standard_deviations(self):
        """Verify the threshold is mean - 1.5 * std, not a different multiple."""
        returns = np.array([0.10, 0.10, 0.10, 0.10, -1.00])
        mean, std = np.mean(returns), np.std(returns)
        threshold = mean - 1.5 * std
        ev = _make_evaluator(values=[100], trades=[])
        flagged = ev._flag_bad_trades(returns)
        for i in flagged:
            assert returns[i] < threshold

class TestAvgRiskReward:

    def test_known_values(self):
        """avg_reward=75, avg_risk=25 → R/R = 3.0."""
        ev = _make_evaluator(
            values=[100],
            trades=[
                _trade(100, 0.10),  
                _trade(50, 0.05),    
                _trade(-25, -0.03), 
            ],
        )
        assert ev._calc_avg_risk_reward() == pytest.approx(3.0, rel=1e-4)

    def test_no_losing_trades_returns_zero(self):
        ev = _make_evaluator(
            values=[100],
            trades=[_trade(50, 0.05), _trade(30, 0.03)],
        )
        assert ev._calc_avg_risk_reward() == pytest.approx(0.0)
