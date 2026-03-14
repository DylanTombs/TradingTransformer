"""Unit tests for research/transformer/DataFrame.py (DataFrameDataset).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from DataFrame import DataFrameDataset


SEQ_LEN = 10
LABEL_LEN = 5
PRED_LEN = 3
SIZE = (SEQ_LEN, LABEL_LEN, PRED_LEN)
AUX_FEATURES = ["f1", "f2"]
TARGET = "close"


def _make_df(n_rows: int, ticker: str = "AAPL", seed: int = 0) -> pd.DataFrame:
    """Return a minimal DataFrame with the columns DataFrameDataset requires."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
            "close": rng.uniform(90, 110, n_rows),
            "f1": rng.standard_normal(n_rows),
            "f2": rng.standard_normal(n_rows),
            "ticker": ticker,
        }
    )


def _make_train_scalers(df: pd.DataFrame):
    """Fit and return (feature_scaler, target_scaler) on df — same logic as
    DataFrameDataset uses internally for the 'train' flag."""
    fs = StandardScaler().fit(df[AUX_FEATURES].values)
    ts = StandardScaler().fit(df[[TARGET]].values)
    return fs, ts


class TestWindowCount:

    def test_single_ticker_window_count_matches_formula(self):
        """len(dataset) == n_rows - seqLen - predLen + 1 for a single ticker."""
        n_rows = 50
        df = _make_df(n_rows)
        ds = DataFrameDataset(df, "train", SIZE, TARGET, AUX_FEATURES)
        expected = n_rows - SEQ_LEN - PRED_LEN + 1
        assert len(ds) == expected

    def test_pred_flag_always_returns_length_one(self):
        """The 'pred' flag is used for live inference; length is always 1."""
        df = _make_df(50)
        fs, ts = _make_train_scalers(df)
        ds = DataFrameDataset(df, "pred", SIZE, TARGET, AUX_FEATURES,
                              featureScaler=fs, targetScaler=ts)
        assert len(ds) == 1


# Scaler discipline

class TestScalerDiscipline:

    def test_train_dataset_creates_new_scalers(self):
        """Train flag must always fit fresh scalers from training data."""
        df = _make_df(50)
        ds = DataFrameDataset(df, "train", SIZE, TARGET, AUX_FEATURES)
        assert hasattr(ds, "featureScaler"), "featureScaler not created on train"
        assert hasattr(ds, "targetScaler"), "targetScaler not created on train"

    def test_val_dataset_uses_provided_scalers_without_refitting(self):
        """Validation data must use train-fitted scalers — not re-fit on val."""
        train_df = _make_df(60, seed=0)
        val_df = _make_df(20, seed=1)

        train_ds = DataFrameDataset(train_df, "train", SIZE, TARGET, AUX_FEATURES)
        val_ds = DataFrameDataset(
            val_df, "val", SIZE, TARGET, AUX_FEATURES,
            featureScaler=train_ds.featureScaler,
            targetScaler=train_ds.targetScaler,
        )
        # The scalers on val_ds must be the same objects passed in
        assert val_ds.featureScaler is train_ds.featureScaler
        assert val_ds.targetScaler is train_ds.targetScaler

    def test_val_without_scalers_raises_value_error(self):
        """Omitting scalers for val/test is a programming error."""
        df = _make_df(50)
        with pytest.raises(ValueError, match="Scalers must be provided"):
            DataFrameDataset(df, "val", SIZE, TARGET, AUX_FEATURES)



# Input validation

class TestInputValidation:

    def test_missing_auxiliary_feature_raises_value_error(self):
        df = _make_df(50).drop(columns=["f1"])
        with pytest.raises(ValueError, match="Missing auxiliary features"):
            DataFrameDataset(df, "train", SIZE, TARGET, AUX_FEATURES)

    def test_missing_target_column_raises_value_error(self):
        df = _make_df(50).drop(columns=["close"])
        with pytest.raises(ValueError, match="Target column"):
            DataFrameDataset(df, "train", SIZE, TARGET, AUX_FEATURES)

    def test_nan_in_auxiliary_features_raises_value_error(self):
        df = _make_df(50)
        df.loc[5, "f1"] = np.nan
        with pytest.raises(ValueError, match="NaN values found in auxiliary features"):
            DataFrameDataset(df, "train", SIZE, TARGET, AUX_FEATURES)

    def test_non_dataframe_input_raises_value_error(self):
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            DataFrameDataset("not_a_df", "train", SIZE, TARGET, AUX_FEATURES)


# Output shape

class TestOutputShape:

    def test_getitem_seq_x_has_correct_shape(self):
        """seqX.shape == (seqLen, n_features) where n_features = len(aux) + 1."""
        df = _make_df(50)
        ds = DataFrameDataset(df, "train", SIZE, TARGET, AUX_FEATURES)
        seq_x, seq_y, seq_x_mark, seq_y_mark = ds[0]
        n_features = len(AUX_FEATURES) + 1   # aux + target
        assert seq_x.shape == (SEQ_LEN, n_features)

    def test_getitem_seq_y_has_correct_shape(self):
        """seqY.shape == (labelLen + predLen, n_features)."""
        df = _make_df(50)
        ds = DataFrameDataset(df, "train", SIZE, TARGET, AUX_FEATURES)
        _, seq_y, _, _ = ds[0]
        n_features = len(AUX_FEATURES) + 1
        assert seq_y.shape == (LABEL_LEN + PRED_LEN, n_features)

    def test_getitem_time_mark_has_three_columns(self):
        """Time stamp tensors encode month, day, weekday — 3 columns."""
        df = _make_df(50)
        ds = DataFrameDataset(df, "train", SIZE, TARGET, AUX_FEATURES)
        seq_x, _, seq_x_mark, _ = ds[0]
        assert seq_x_mark.shape == (SEQ_LEN, 3)




class TestMultiTickerLeakage:

    @pytest.mark.xfail(
        reason=(
            "Known bug: valid_indices is computed per-ticker but __getitem__ "
            "uses sequential indices into the full concatenated dataX array. "
            "Windows at ticker boundaries span data from two different stocks."
        )
    )
    def test_windows_do_not_cross_ticker_boundaries(self):
        """Each window must contain data from exactly one ticker.

        With two tickers (20 rows each) and seqLen=10, predLen=3, a window
        starting at index 12 spans rows 12-21, crossing the ticker boundary
        at row 20.  The dataset should prevent this but currently does not.
        """
        n_per_ticker = 20
        df_a = _make_df(n_per_ticker, ticker="AAPL", seed=0)
        df_b = _make_df(n_per_ticker, ticker="GOOGL", seed=1)
        df = pd.concat([df_a, df_b], ignore_index=True)

        ds = DataFrameDataset(df, "train", SIZE, TARGET, AUX_FEATURES)

        # Window at index 12: rows [12, 22) — crosses the boundary at row 20
        boundary_window_idx = 12
        seq_x, _, _, _ = ds[boundary_window_idx]

        # The first n_per_ticker rows are AAPL; scaled values will differ
        # from GOOGL.  A cross-ticker window will mix the two distributions.
        # We detect leakage by checking that all rows in the window come from
        # the same slice of the original un-mixed scaled data.
        aapl_data = ds.dataX[:n_per_ticker]
        googl_data = ds.dataX[n_per_ticker:]

        window_rows = ds.dataX[boundary_window_idx: boundary_window_idx + SEQ_LEN]
        rows_in_aapl = all(
            any(np.allclose(row, aapl_row) for aapl_row in aapl_data)
            for row in window_rows
        )
        assert rows_in_aapl, (
            "Window crosses ticker boundary: contains rows from both AAPL and GOOGL"
        )
