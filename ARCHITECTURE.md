# Architecture

## Table of Contents

- [High-Level Design](#high-level-design)
- [Data Flow](#data-flow)
- [Key Design Decisions](#key-design-decisions)
- [Backtesting Design](#backtesting-design)
- [Reliability and Edge Cases](#reliability-and-edge-cases)
- [Scalability Considerations](#scalability-considerations)

---

## High-Level Design

The system is split into two independent layers with a well-defined contract at the boundary.

```
┌─────────────────────────────────────────────┐
│               Research Layer (Python)        │
│                                             │
│  StockDataPD.py  →  pipeline.py             │
│                      │                      │
│  Train.py        →  Interface.py            │
│                      │                      │
│  exportModel.py  → transformer.pt           │
│                     feature_scaler.csv      │
│                     target_scaler.csv       │
└────────────────────┬────────────────────────┘
                     │  CSV contract
                     │  (feature columns + model artefacts)
┌────────────────────▼────────────────────────┐
│             Execution Layer (C++)            │
│                                             │
│  FeatureCSVDataHandler                      │
│  MLStrategy + ScalerParams                  │
│  BacktestEngine (event loop)                │
│  Portfolio + RiskManager + Execution        │
└─────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Layer | Responsibility |
|---|---|---|
| `pipeline.py` | Python | Produce enriched feature CSVs from raw OHLCV; must be identical to training |
| `Interface.py` | Python | Train/validate the Transformer; checkpoint best model |
| `exportModel.py` | Python | Trace model to TorchScript; export scaler parameters to CSV |
| `FeatureCSVDataHandler` | C++ | Read feature CSVs bar-by-bar; emit `FeatureMarketEvent` per bar |
| `MLStrategy` | C++ | Buffer a rolling window; run LibTorch inference; emit `SignalEvent` |
| `ScalerParams` | C++ | Load and apply `sklearn.StandardScaler` parameters without Python |
| `BacktestEngine` | C++ | Dispatch events through the MARKET→SIGNAL→ORDER→FILL pipeline |
| `Portfolio` | C++ | Track cash, positions, equity curve, and trade log |
| `RiskManager` | C++ | Enforce pre-trade position limits before orders reach execution |
| `SimulatedExecution` | C++ | Convert approved orders to fills; apply commission |

---

## Data Flow

### 1. Data Acquisition

`StockDataPD.py` fetches OHLCV bars from Yahoo Finance and writes per-symbol CSVs to `data/`. Each file has the columns: `date`, `open`, `high`, `low`, `close`, `volume`, `adj close`.

### 2. Feature Engineering

`pipeline.py` processes each raw CSV and writes an enriched CSV to `features/`. Feature computation is **bar-by-bar** using a backtrader-compatible adapter layer (`_Line`, `_DataView`, `_BarCtx`) that wraps the pandas DataFrame. This adapter calls the same functions in `technicalIndicators.py` that were used during model training, making train/serve feature parity a structural property rather than a convention.

Output columns (34 total, in fixed order):

```
date | high | low | volume | adj close |
P R1 R2 R3 S1 S2 S3 |           ← pivot points
obv volume_zscore |              ← volume indicators
rsi macd macds macdh |          ← momentum
sma lma sema lema |             ← moving averages
overnight_gap return_lag_1 return_lag_3 return_lag_5 volatility |
SR_K SR_D SR_RSI_K SR_RSI_D |  ← stochastic oscillators
ATR HL_PCT PCT_CHG |            ← volatility
close                           ← prediction target
```

Column order is fixed and validated against `exportModel.py::load_args()`.

### 3. Model Training

`Train.py` creates a `Model_Interface` and calls `train()` with an argument dictionary loaded from `.env`. The `DataFrameDataset` class in `DataFrame.py` constructs sliding windows of length `seqLen + predLen` from the feature CSVs. The scaler is fit on the training split only and applied to both splits. The best checkpoint (by validation loss) is saved to `research/models/Model3.pth`.

### 4. Model Export

`exportModel.py` performs two operations:

**TorchScript export:**
```
TransformerInferenceWrapper(model)
        │
        │  torch.jit.trace(dummy_xEnc, dummy_xMarkEnc)
        ▼
models/transformer.pt          ← loads in C++ via torch::jit::load()
```

The wrapper fuses the encoder and decoder into a single `forward(xEnc, xMarkEnc)` call, matching the signature expected by `MLStrategy::runInference()`.

**Scaler export:**
```python
# feature_scaler.csv
feature,mean,scale
high,102.15,25.30
...34 rows

# target_scaler.csv
feature,mean,scale
close,150.80,35.45
```

This CSV format is the contract between Python and C++. `ScalerParams::loadFromCSV()` parses it on startup.

### 5. C++ Backtest Execution

`ml_main.cpp` constructs the engine components and calls `engine.run()`:

```
FeatureCSVDataHandler::streamNext()
        │ emits FeatureMarketEvent (symbol, price, timestamp, features[34], timeMark[3])
        ▼
BacktestEngine::run()  ← event loop
        │
        ├─ MARKET  → Portfolio::updateMarket()
        │            MLStrategy::onMarketEvent()
        │               ├─ dynamic_cast<FeatureMarketEvent*>   recover feature vector
        │               ├─ ScalerParams::transform()           normalise features
        │               ├─ featureBuffer_.push_back()          maintain rolling window
        │               └─ runInference()                      LibTorch forward pass
        │                      │
        │                      ▼ predicted close (inverse-scaled)
        │               emit SignalEvent (LONG or EXIT)
        │
        ├─ SIGNAL  → Portfolio::generateOrder()
        │            emit OrderEvent
        │
        ├─ ORDER   → RiskManager::approveOrder()
        │            SimulatedExecution::executeOrder()
        │            emit FillEvent
        │
        └─ FILL    → Portfolio::updateFill()
```

At the end of the run, `Portfolio::exportEquityCurve()` and `exportTrades()` write the output CSVs.

---

## Key Design Decisions

### 1. C++ execution engine, Python research layer

**Decision:** All live trading simulation runs in C++. Python is only used for data preparation, training, and export.

**Rationale:** Separating concerns at the language boundary makes each layer independently testable and avoids the overhead of running a Python interpreter per bar. More importantly, it forces a clean contract (CSV + TorchScript) that must be honoured by both sides — making integration failures explicit rather than silent.

**Trade-off:** Additional complexity at the boundary. Scaler parameters must be serialised manually and the feature column order is a shared assumption. These are managed by `exportModel.py::load_args()` being the single source of truth for both sides.

### 2. Transformer over LSTM / simpler models

**Decision:** Encoder-decoder Transformer with 34 input features, `seqLen=30`, `predLen=5`.

**Rationale:** The Transformer's attention mechanism can learn non-local temporal dependencies without the vanishing gradient problem of LSTMs. For financial time series where a regime shift 20 bars ago may be more informative than the last 3 bars, this is a meaningful property. The encoder-decoder structure naturally separates "what the market has done" (encoder input) from "what we are predicting" (decoder output).

**Trade-off:** Transformers require more data and more careful regularisation than LSTMs to avoid overfitting. The model is larger (~23 MB), slower to train, and requires `seqLen=30` bars before it can emit any signal — meaning the first 30 bars of every backtest are always skipped.

**Alternatives considered:**
- *LSTM:* Simpler and faster to train, well-suited to short sequences. Chosen against because long-range dependency learning is relevant for momentum strategies.
- *Linear regression / XGBoost:* Interpretable, less prone to overfitting. Chosen against because the target (future close) has a non-linear relationship with the feature set that tree models approximate poorly over continuous output.
- *1-D CNN:* Fast and memory-efficient. Would be a reasonable alternative for pure pattern recognition; chosen against here because the positional attention weights are informative for understanding which past bars drive predictions.

### 3. TorchScript via `torch.jit.trace`

**Decision:** Export the model with `torch.jit.trace` rather than `torch.jit.script`.

**Rationale:** `trace` requires only a single forward pass with dummy inputs and produces a frozen computation graph. `script` requires the Python source to be fully type-annotated and free of Python control flow that cannot be serialised. The model uses standard `nn.Module` composition with no branching, so `trace` is sufficient and significantly simpler.

**Trade-off:** A traced model is frozen to the input shapes seen at trace time (`batch=1`, `seq=30`, `features=34`). Any change to the sequence length or feature count requires re-export. This is acceptable because these are fixed hyperparameters.

### 4. `FeatureMarketEvent` inherits `MarketEvent`

**Decision:** `FeatureMarketEvent` is a subclass of `MarketEvent`, not a parallel event type.

**Rationale:** `BacktestEngine` dispatches on `EventType::MARKET` and casts to `MarketEvent`. If `FeatureMarketEvent` were a separate type, the engine would need modification to handle it — breaking the open/closed principle for what is a purely additive change. Inheritance means the engine is completely unmodified, and only `MLStrategy` needs to `dynamic_cast` to recover the extra payload.

**Trade-off:** Using `dynamic_cast` at runtime has a small cost and couples `MLStrategy` to the concrete subtype. This is acceptable for a strategy that is explicitly designed for feature-enriched events. A strategy that mistakenly receives a plain `MarketEvent` (which has no features) will receive `nullptr` from the cast and produce no signal — a safe failure mode.

### 5. Header-only `ScalerParams`

**Decision:** `ScalerParams` is implemented entirely in the header, not split into `.cpp`.

**Rationale:** The struct is simple enough that the implementation fits in ~60 lines without becoming hard to read. Keeping it header-only removes a compilation unit and makes it trivially usable by any target (including test binaries) without linking.

**Trade-off:** If the scaler logic grows substantially, it should be moved to a `.cpp` to avoid binary size inflation from inlining.

---

## Backtesting Design

### Simulation assumptions

| Assumption | Current behaviour |
|---|---|
| Fill price | Last bar's `close` (no slippage) |
| Commission | Fixed per trade (configurable, default $1.00) |
| Position sizing | Fixed 10 shares per trade |
| Market impact | Not modelled |
| Short selling | Not implemented (`SignalType::SHORT` is present but `generateOrder` does not handle it) |
| Latency | Zero (signal and fill occur on the same bar) |

### Event ordering within a bar

The engine processes exactly one event per `streamNext()` call. On a MARKET event, the strategy fires and may push a SIGNAL. That SIGNAL is processed on the same iteration (the inner `while queue is not empty` loop). This means a signal generated on bar *t* is filled at bar *t*'s close price — a mild look-ahead that is common in end-of-day backtesting but would not be acceptable in intraday simulation.

### Known correctness gap: EXIT handling

`Portfolio::generateOrder` for `SignalType::EXIT` currently returns an `OrderEvent` with `quantity=0`. `SimulatedExecution` passes this through, producing a fill that changes nothing. Positions opened by a LONG signal are never closed. This inflates returns and win rates in the reported results. It is tracked as a known bug.

---

## Reliability and Edge Cases

### Missing data

`FeatureCSVDataHandler` reads the CSV sequentially and emits one event per non-empty row. It does not detect gaps in the date sequence. If a symbol has missing bars (weekend gaps, trading halts), the engine will process them as if they are adjacent — the rolling window in `MLStrategy` will not be aware of a gap. This can cause the time mark encoding to be inconsistent.

**Mitigation:** `pipeline.py` produces output only for rows where all 34 features can be computed. Rows within the warm-up period (e.g. the first 26 bars needed for a 26-period EMA) are dropped, so the feature CSV starts at a bar where all values are valid.

### Data leakage prevention

The training/validation split in `DataFrameDataset` is performed by index before the scaler is fit. The scaler is fit on training windows only and applied to both splits. This is validated by `test_dataset.py::test_scaler_fit_on_train_only`.

A known, documented leakage issue exists at ticker boundaries: a sliding window that starts on the last bar of symbol A and ends on the first bar of symbol B is currently allowed. This is marked as `xfail` in `test_dataset.py` and is a tracked bug.

### Reproducibility

Training is not fully deterministic because PyTorch's multi-threaded operations have non-deterministic CUDA kernels by default. However, for CPU training, results are reproducible given the same environment and random seed. The exported `transformer.pt` is a fixed artefact — backtests against the same feature CSVs are deterministic.

### Feature consistency

The most common source of silent failure in ML systems is a mismatch between training features and inference features. This system prevents it by design:

1. `technicalIndicators.py` is the single source of all indicator logic.
2. `pipeline.py` calls those functions through an adapter; it does not reimplement them.
3. `exportModel.py::load_args()` defines the canonical feature list and column order.
4. `ml_main.cpp` passes that same list to `FeatureCSVDataHandler` at construction.

Any divergence in feature order causes `ScalerParams::transform` to throw a size-mismatch error at runtime.

---

## Scalability Considerations

The current system is designed for a research workflow, not production throughput. Key bottlenecks and their mitigations:

| Concern | Current state | Path to improvement |
|---|---|---|
| Single-symbol execution | One process per symbol | Run multiple symbols in parallel processes; portfolio layer aggregates fills |
| Full CSV in memory | `FeatureCSVDataHandler` streams row-by-row but loads the whole file at construction | Switch to memory-mapped reads or a database backend (e.g. DuckDB) for large symbol sets |
| Model loading | `torch::jit::load()` called once per process | Shared model loaded once; strategy instances share a reference |
| Position sizing | Hardcoded quantity | Risk-based sizing (fixed-fractional) requires total portfolio equity at signal time — available via `Portfolio::getCash()` + position MTM |
| Feature pipeline | Per-symbol serial processing | `pipeline.py` is embarrassingly parallel over symbols; trivially parallelisable with `multiprocessing.Pool` |
