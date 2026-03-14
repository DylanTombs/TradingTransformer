"""Unit tests for research/features/technicalIndicators.py.

All indicator functions are implemented as methods that accept a strategy-like
`self` object whose `self.data` attribute mimics the backtrader line series API:
  - self.data.close[0]     → current bar value
  - self.data.close[-n]    → n bars ago
  - self.data.close.get(size=n) → list of last n values (oldest → newest)
  - len(self.data)         → number of bars loaded

MockLine and MockData replicate this contract without importing backtrader.
"""
import math
import numpy as np
import pytest

from technicalIndicators import (
    calculateRsi,
    calculateMacd,
    calculateVolatility,
    calculateVolumeZscore,
    calculateOvernightGap,
    calculateReturn,
    calculateSMA,
    calculateATR,
    calculatePctChange,
)

# ---------------------------------------------------------------------------
# Backtrader data-feed mock
# ---------------------------------------------------------------------------

class MockLine:
    """Mimics a backtrader LineSeries.

    Internally stores values oldest-first.  The indexing contract is:
      [0]  → most recent (current bar)
      [-n] → n bars ago
    which maps to _values[-1] and _values[-(n+1)] respectively,
    condensed as _values[idx - 1].
    """

    def __init__(self, values: list):
        self._values = list(values)

    def __getitem__(self, idx: int):
        return self._values[idx - 1]

    def get(self, size: int) -> list:
        """Return the last `size` values in chronological order."""
        return list(self._values[-size:])


class MockData:
    def __init__(self, closes, volumes=None, highs=None, lows=None, opens=None):
        n = len(closes)
        self.close = MockLine(closes)
        self.volume = MockLine(volumes if volumes is not None else [1_000] * n)
        self.high = MockLine(highs if highs is not None else closes)
        self.low = MockLine(lows if lows is not None else closes)
        self.open = MockLine(opens if opens is not None else closes)
        self._n = n

    def __len__(self):
        return self._n


class MockSelf:
    """Minimal stand-in for a backtrader Strategy instance."""

    def __init__(self, closes, volumes=None, highs=None, lows=None, opens=None):
        self.data = MockData(closes, volumes, highs, lows, opens)


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------

class TestCalculateRsi:

    def test_insufficient_data_returns_zero(self):
        """Fewer than 15 bars → guard returns 0.0."""
        ctx = MockSelf(closes=[100.0] * 14)
        assert calculateRsi(ctx) == 0.0

    def test_all_gains_returns_near_100(self):
        """Monotonically rising prices → RSI ≈ 100 (avgLoss ≈ 0)."""
        prices = list(range(15))           # [0, 1, …, 14]
        ctx = MockSelf(closes=prices)
        result = calculateRsi(ctx)
        assert result > 99.0, f"Expected RSI near 100, got {result}"

    def test_all_losses_returns_near_0(self):
        """Monotonically falling prices → RSI ≈ 0 (avgGain = 0)."""
        prices = list(range(14, -1, -1))   # [14, 13, …, 0]
        ctx = MockSelf(closes=prices)
        result = calculateRsi(ctx)
        assert result < 1.0, f"Expected RSI near 0, got {result}"

    def test_equal_gains_and_losses_returns_50(self):
        """7 equal gains and 7 equal losses → RSI = 50."""
        # Alternating: [0, 2, 0, 2, …, 0] — 15 values
        prices = [0.0 if i % 2 == 0 else 2.0 for i in range(15)]
        ctx = MockSelf(closes=prices)
        result = calculateRsi(ctx)
        assert abs(result - 50.0) < 0.01, f"Expected RSI ≈ 50, got {result}"

    def test_result_bounded_between_0_and_100(self):
        """RSI must always lie in [0, 100]."""
        rng = np.random.default_rng(seed=42)
        prices = rng.standard_normal(30).cumsum() + 100
        ctx = MockSelf(closes=prices.tolist())
        result = calculateRsi(ctx)
        assert 0.0 <= result <= 100.0


# ---------------------------------------------------------------------------
# MACD tests
# NOTE: The implementation uses simple arithmetic means, not true EMA.
# ---------------------------------------------------------------------------

class TestCalculateMacd:

    def test_insufficient_data_returns_zero(self):
        ctx = MockSelf(closes=[100.0] * 25)
        assert calculateMacd(ctx) == 0.0

    def test_constant_prices_returns_zero(self):
        """flat price series → ema12 == ema26 → MACD = 0."""
        ctx = MockSelf(closes=[50.0] * 26)
        assert calculateMacd(ctx) == pytest.approx(0.0, abs=1e-9)

    def test_rising_prices_is_positive(self):
        """Recent prices higher than longer history → MACD > 0."""
        ctx = MockSelf(closes=list(range(26)))  # [0, 1, …, 25]
        # ema12 = mean([14..25]) = 19.5, ema26 = mean([0..25]) = 12.5
        result = calculateMacd(ctx)
        assert result == pytest.approx(7.0, abs=1e-9)

    def test_falling_prices_is_negative(self):
        """Recent prices lower than longer history → MACD < 0."""
        ctx = MockSelf(closes=list(range(25, -1, -1)))  # [25, 24, …, 0]
        result = calculateMacd(ctx)
        assert result < 0.0


# ---------------------------------------------------------------------------
# Volatility tests
# ---------------------------------------------------------------------------

class TestCalculateVolatility:

    def test_insufficient_data_returns_zero(self):
        ctx = MockSelf(closes=[100.0] * 19)
        assert calculateVolatility(ctx) == 0.0

    def test_constant_prices_returns_zero(self):
        """No price movement → zero returns → zero std."""
        ctx = MockSelf(closes=[100.0] * 20)
        assert calculateVolatility(ctx) == pytest.approx(0.0, abs=1e-9)

    def test_known_returns_std(self):
        """Manually verify std of log-returns against numpy reference."""
        closes = [100.0 * (1.01 ** i) for i in range(20)]  # 1% daily gains
        ctx = MockSelf(closes=closes)
        arr = np.array(closes)
        returns = np.diff(arr) / arr[:-1]
        expected = returns.std()
        assert calculateVolatility(ctx) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Volume Z-score tests
# ---------------------------------------------------------------------------

class TestCalculateVolumeZscore:

    def test_insufficient_data_returns_zero(self):
        ctx = MockSelf(closes=[100.0] * 19, volumes=[1000] * 19)
        assert calculateVolumeZscore(ctx) == 0.0

    def test_mean_volume_returns_near_zero(self):
        """When current volume equals the mean, z-score ≈ 0."""
        volumes = [1_000.0] * 20
        ctx = MockSelf(closes=[100.0] * 20, volumes=volumes)
        result = calculateVolumeZscore(ctx)
        assert abs(result) < 1e-4, f"Expected z-score ≈ 0, got {result}"

    def test_high_volume_returns_positive_zscore(self):
        """Volume well above mean → positive z-score."""
        volumes = [1_000.0] * 19 + [10_000.0]   # large spike on current bar
        ctx = MockSelf(closes=[100.0] * 20, volumes=volumes)
        assert calculateVolumeZscore(ctx) > 1.0


# ---------------------------------------------------------------------------
# Overnight gap tests
# ---------------------------------------------------------------------------

class TestCalculateOvernightGap:

    def test_no_gap_returns_zero(self):
        """prev_close == current_open → log(1) = 0."""
        closes = [100.0] * 5
        opens = [100.0] * 5
        ctx = MockSelf(closes=closes, opens=opens)
        assert calculateOvernightGap(ctx) == pytest.approx(0.0, abs=1e-9)

    def test_gap_up_returns_positive(self):
        """Current open above previous close → positive gap."""
        closes = [100.0] * 5
        opens = [100.0, 100.0, 100.0, 100.0, 110.0]  # current open = 110
        ctx = MockSelf(closes=closes, opens=opens)
        result = calculateOvernightGap(ctx)
        assert result == pytest.approx(math.log(110 / 100), rel=1e-6)

    def test_gap_down_returns_negative(self):
        """Current open below previous close → negative gap."""
        closes = [100.0] * 5
        opens = [100.0, 100.0, 100.0, 100.0, 90.0]
        ctx = MockSelf(closes=closes, opens=opens)
        assert calculateOvernightGap(ctx) < 0.0


# ---------------------------------------------------------------------------
# Return lag tests
# ---------------------------------------------------------------------------

class TestCalculateReturn:

    def test_lag_1_known_value(self):
        """(current / 1-bar-ago) - 1 with known prices."""
        closes = [100.0] * 4 + [110.0]   # current = 110, 1 bar ago = 100
        ctx = MockSelf(closes=closes)
        assert calculateReturn(ctx, lag=1) == pytest.approx(0.10, rel=1e-6)

    def test_no_change_returns_zero(self):
        ctx = MockSelf(closes=[100.0] * 10)
        assert calculateReturn(ctx, lag=5) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# SMA tests
# ---------------------------------------------------------------------------

class TestCalculateSMA:

    def test_sma_equals_arithmetic_mean(self):
        """SMA over a window equals the arithmetic mean of that window."""
        closes = list(range(1, 21))        # [1, 2, …, 20]
        ctx = MockSelf(closes=closes)
        result = calculateSMA(ctx, period=5)
        expected = sum(range(16, 21)) / 5  # mean([16,17,18,19,20]) = 18.0
        assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------

class TestCalculateATR:

    def test_insufficient_data_returns_zero(self):
        ctx = MockSelf(closes=[10.0] * 13, highs=[12.0] * 13, lows=[8.0] * 13)
        assert calculateATR(ctx) == 0.0

    def test_constant_ohlc_true_range_equals_hl_spread(self):
        """When H=12, L=8, C=10 every bar, TR = H-L = 4 (no gaps)."""
        n = 20
        ctx = MockSelf(
            closes=[10.0] * n,
            highs=[12.0] * n,
            lows=[8.0] * n,
        )
        assert calculateATR(ctx) == pytest.approx(4.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Pct change tests
# ---------------------------------------------------------------------------

class TestCalculatePctChange:

    def test_known_pct_change(self):
        closes = [100.0, 105.0]   # prev = 100, current = 105
        ctx = MockSelf(closes=closes)
        assert calculatePctChange(ctx) == pytest.approx(5.0, rel=1e-6)

    def test_no_change_returns_zero(self):
        ctx = MockSelf(closes=[100.0, 100.0])
        assert calculatePctChange(ctx) == pytest.approx(0.0, abs=1e-9)
