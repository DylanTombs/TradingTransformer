# Phase 4 — Strategy Extension & Scalability

**Status:** Not Started  
**Prerequisites:** Phase 1 complete  
**Can run in parallel with:** Phase 2, Phase 3

---

## Objective

Extend the strategy surface (short selling) and remove known architectural ceilings (serial pipeline, per-symbol model loading, convention-based feature contract). Phase 4 transforms the system from a prototype that works for a small fixed symbol set into one that scales gracefully and supports a richer signal space.

**Why Phase 1 must precede Phase 4:** Short selling adds new accounting paths in `Portfolio` and `SimulatedExecution` — components that currently have zero unit tests (addressed in Phase 1). Implementing short selling without those tests would create an untested critical path. The feature schema contract (Task 4.3) also relies on the schema validation pattern introduced in Phase 1.

**Exit criteria (all must be satisfied before Phase 4 is closed):**
- [ ] Short positions open, track P&L, and close correctly in the backtester
- [ ] `allow_short` flag in `backtest_config.yaml`; default `false` (backward compatible)
- [ ] `feature_schema.json` generated during export; validated at C++ startup
- [ ] Feature pipeline runs in parallel over symbols; ≥ 2× speedup for 10 symbols on a 4-core machine
- [ ] Shared `torch::jit::Module` used across all `MLStrategy` instances in `MultiSymbolStrategy`
- [ ] Spearman correlation replaces Pearson in `Portfolio`'s correlation discount
- [ ] All new components have ≥ 80% unit test coverage
- [ ] All tasks in this document marked complete

---

## Task Breakdown

### 4.1 Short Selling Implementation (B-05)

**Files:**
- `backtester/include/strategy/MLStrategy.hpp` / `.cpp`
- `backtester/include/portfolio/Portfolio.hpp` / `.cpp`
- `backtester/include/execution/SimulatedExecution.hpp` / `.cpp`
- `backtester/include/config/BacktestConfig.hpp`
- `backtest_config.yaml`

**Issue:** `SignalType::SHORT` is declared in the event hierarchy but unhandled. `Portfolio::generateOrder()` does not have a SHORT branch. `SimulatedExecution` has no concept of short position accounting (margin, proceeds, cover cost).

#### 4.1.1 Config Addition

```yaml
allow_short:          false    # Enable short selling (default: false — backward compatible)
short_margin_rate:    1.0      # Fraction of position value required as margin (1.0 = 100%)
```

Add to schema validation: `short_margin_rate` must be > 0 and ≤ 2.0.

#### 4.1.2 MLStrategy Signal Generation

When `allow_short` is true, extend the signal logic:

```
Current:
  predicted > price × (1 + buyThreshold)  →  LONG
  predicted < price × (1 - exitThreshold) →  EXIT (close long)

Extended:
  predicted > price × (1 + buyThreshold)  →  LONG (open long or no-op if already long)
  predicted < price × (1 - exitThreshold) →  SHORT (open short or EXIT if covering long)

Position state machine:
  FLAT + LONG signal   →  open long
  LONG + EXIT signal   →  close long, go flat
  FLAT + SHORT signal  →  open short (if allow_short)
  SHORT + LONG signal  →  close short, go flat (or flip to long)
```

Add `positionDirection_` (enum: FLAT, LONG, SHORT) to `MLStrategy` to replace the current `hasPosition_` bool.

#### 4.1.3 Portfolio Short Accounting

**Opening a short position:**
```
proceeds  = qty × price × (1 - halfSpread - slippage)    // sell at bid
cash     += proceeds                                       // receive sale proceeds
shortPos[symbol] += qty                                    // record short shares
margin[symbol]    = qty × price × shortMarginRate         // reserved margin
```

**Mark-to-market unrealised P&L on short:**
```
unrealised_pnl[symbol] = shortPos[symbol] × (entryPrice - currentPrice)
// Short profits when price falls
```

**Closing a short position (cover):**
```
cost  = qty × price × (1 + halfSpread + slippage)        // buy at ask
cash -= cost                                              // pay to cover
realised_pnl = shortPos[symbol] × (entryPrice - price) - commission
shortPos[symbol] = 0
margin[symbol]   = 0
```

**Equity curve with shorts:**
```
equity = cash + sum(longPos[s] × price[s]) - sum(shortPos[s] × price[s]) + unrealised_pnl
```

The `EquityPoint` struct must reflect both long and short market value.

#### 4.1.4 Risk Manager Extension

Add short-specific limits to `RiskManager::approveOrder()`:
- Reject SHORT orders if `allow_short == false`.
- Reject SHORT orders that would cause `shortExposure > maxTotalExposure × equity`.
- Reject SHORT orders with `qty > maxPositionSize`.

#### 4.1.5 Test Requirements

**New tests in `test_portfolio.cpp`:**
- Short order generates correct proceeds (cash increases).
- Mark-to-market: price rises on short position → negative unrealised P&L in equity curve.
- Cover order: cash decreases, short position cleared, realised P&L correct.
- SHORT rejected when `allow_short = false`.
- SHORT + LONG flip: covers short then opens long correctly.
- Equity curve with simultaneous long (symbol A) + short (symbol B).
- Short margin reserved correctly; margin released on cover.

**New test in `test_execution.cpp`:**
- SHORT fill price = `price × (1 - halfSpread - slippage) - impact × qty`.
- COVER fill price = `price × (1 + halfSpread + slippage) + impact × qty`.

**Acceptance criteria:**
- `allow_short: false` backtest is byte-for-byte identical to pre-Phase-4 output (backward compatibility).
- `allow_short: true` backtest on a falling trend symbol shows net short P&L > 0.

---

### 4.2 Shared torch::jit::Module Across MLStrategy Instances (B-06)

**Files:**
- `backtester/include/strategy/MLStrategy.hpp` / `.cpp`
- `backtester/include/strategy/MultiSymbolStrategy.hpp` (if exists, else `backtester/ml_main.cpp`)

**Issue:** `MultiSymbolStrategy` creates one `MLStrategy` per symbol, each of which calls `torch::jit::load(modelPath)` independently. For 10 symbols, the same ~23 MB model is loaded 10 times into separate heap allocations. This wastes ~200 MB of memory and adds 10× the model load time at startup.

**Required change:**

Change `MLStrategy` constructor to accept a `std::shared_ptr<torch::jit::Module>`:

```cpp
// Before
MLStrategy(const std::string& modelPath, const ScalerParams& scaler, ...);

// After
MLStrategy(std::shared_ptr<torch::jit::Module> model, const ScalerParams& scaler, ...);
```

Load the model once in `ml_main.cpp` (or `MultiSymbolStrategy`):

```cpp
auto sharedModel = std::make_shared<torch::jit::Module>(
    torch::jit::load(config.modelPath)
);
sharedModel->eval();

for (const auto& sym : symbols) {
    strategies[sym] = std::make_unique<MLStrategy>(sharedModel, scaler, config, sym);
}
```

**Thread safety:** `torch::jit::Module::forward()` is thread-safe for inference when called with `torch::NoGradGuard` and separate input tensors per call. Each `MLStrategy` owns its own input buffer; no locking is required for the shared model.

**Test requirement:**
- Unit test: two `MLStrategy` instances created with the same `shared_ptr<Module>` produce identical outputs for identical inputs.
- Unit test: `use_count()` on the shared model equals `nSymbols + 1` (one per strategy plus the owning reference) after all strategies are constructed.
- Memory test (manual, not automated): confirm RSS memory does not scale linearly with symbol count when using shared model.

**Acceptance criteria:**
- `ml_main.cpp` calls `torch::jit::load()` exactly once regardless of symbol count.
- No change in backtest output values (inference produces identical signals).

---

### 4.3 Feature Column Schema Contract (TD-08)

**Files:**
- `research/exportModel.py`
- `backtester/src/market/FeatureCSVDataHandler.cpp`
- `backtester/include/market/FeatureCSVDataHandler.hpp`
- `backtester/CMakeLists.txt`

**Issue:** The 34-column feature order is maintained by convention: `exportModel.py::load_args()` is the canonical source, but the C++ side relies on `MODEL_FEATURE_COLUMNS` being manually kept in sync. A divergence only surfaces as a size-mismatch error, not a meaningful column-name mismatch.

**Required change:**

#### 4.3.1 Export schema file

In `exportModel.py`, after computing the feature list, serialise to JSON:

```python
schema = {
    "version": 1,
    "feature_count": len(feature_columns),
    "columns": [
        {"index": i, "name": col, "type": "float64"}
        for i, col in enumerate(feature_columns)
    ],
    "created_at": datetime.utcnow().isoformat()
}
with open("models/feature_schema.json", "w") as f:
    json.dump(schema, f, indent=2)
```

#### 4.3.2 C++ schema validation at startup

Add `nlohmann/json` via CMake `FetchContent`:

```cmake
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
    DOWNLOAD_NO_EXTRACT TRUE
)
```

In `FeatureCSVDataHandler` constructor, if `feature_schema_path` is set in config:
1. Load `feature_schema.json`.
2. Validate `feature_count` matches `MODEL_FEATURE_COLUMNS.size()`.
3. Validate each column name matches at the same index.
4. On mismatch: throw `std::runtime_error` with:
   ```
   [SCHEMA] Column mismatch at index 5: expected 'rsi', got 'macd'
   ```

Add to `BacktestConfig`:
```yaml
feature_schema_json:  /models/feature_schema.json   # optional; if set, validated at startup
```

**Test requirement:**
- `exportModel.py` generates `models/feature_schema.json` with correct column count and names.
- C++ test: schema with correct columns → no exception.
- C++ test: schema with one column renamed → exception with the index and names in the message.
- C++ test: schema with wrong `feature_count` → exception naming the count mismatch.
- C++ test: `feature_schema_json` not set in config → no schema check (backward compatible).

**Acceptance criteria:**
- A mismatch between Python export and C++ feature list is caught at startup with a human-readable error, not at inference time with a tensor size crash.

---

### 4.4 Parallel Feature Pipeline (TD-09)

**File:** `research/features/pipeline.py`  
**File:** `run_pipeline.py`

**Issue:** `pipeline.py` processes symbols one at a time in a Python for loop. For 10 symbols, feature engineering takes 10× as long as for 1 symbol. The computation is embarrassingly parallel — each symbol's features are independent.

**Required change:**

Add a `--workers` argument to `pipeline.py`:

```python
parser.add_argument("--workers", type=int, default=1,
                    help="Number of parallel worker processes (default: 1)")
```

Use `multiprocessing.Pool` when `workers > 1`:

```python
if args.workers > 1:
    with multiprocessing.Pool(processes=args.workers) as pool:
        results = pool.starmap(process_symbol, [(symbol, args) for symbol in symbols])
else:
    results = [process_symbol(symbol, args) for symbol in symbols]
```

`process_symbol()` must be a top-level function (not a method or lambda) for pickling.

**Ordering guarantee:** Sort output file list by symbol name after pool completes. The order of `feature_schema.json` column names must not depend on processing order.

Update `run_pipeline.py` to pass `--workers $(nproc)` by default:
```python
parser.add_argument("--pipeline-workers", type=int,
                    default=multiprocessing.cpu_count(),
                    help="Worker processes for feature pipeline")
```

**Test requirement:**
- Unit test: `process_symbol()` is importable as a top-level function (pickling test).
- Unit test: parallel run with 2 workers on 4 symbols produces identical output to serial run.
- Performance test (not automated — document in README): wall time for 10 symbols with `--workers 4` is ≤ 50% of wall time with `--workers 1`.

**Acceptance criteria:**
- `python pipeline.py data/ -o features/ --workers 4` completes without error.
- Output files are byte-for-byte identical to serial run (deterministic output).

---

### 4.5 Spearman Correlation for Position Sizing Discount (A-05)

**File:** `backtester/src/portfolio/Portfolio.cpp`  
**File:** `backtester/include/portfolio/Portfolio.hpp`

**Issue:** `Portfolio` uses Pearson correlation to compute the sizing discount for correlated positions. Pearson measures linear association and is sensitive to outliers — a single extreme return day can dominate the correlation estimate. Financial return series frequently have fat tails and occasional outlier bars, making Pearson unreliable for this use case. Spearman rank correlation is more robust: it measures monotonic association on the rank-transformed returns, which down-weights the influence of outliers.

**Required change:**

Replace the Pearson computation in `Portfolio::computeCorrelation()`:

**Pearson (current):**
```cpp
double pearsonCorr(const std::vector<double>& x, const std::vector<double>& y) {
    // sum of (xi - mean_x)(yi - mean_y) / (n * std_x * std_y)
}
```

**Spearman (new):**
```cpp
double spearmanCorr(const std::vector<double>& x, const std::vector<double>& y) {
    // 1. Compute rank vectors for x and y (average ties)
    // 2. Apply pearsonCorr on the rank vectors
}

std::vector<double> rankVector(const std::vector<double>& v) {
    // Returns rank of each element; tied values receive average rank
}
```

The Spearman formula reduces to Pearson on the rank-transformed inputs, so the existing `pearsonCorr` function is reused rather than duplicated.

**Backward compatibility:** The formula change will alter correlation values and therefore sizing discount values. Backtests will not be byte-for-byte identical before and after Phase 4. Document this in `DECISIONS.md` as `ADR-023`.

**Test requirement (add to `test_portfolio.cpp`):**
- `rankVector({3.0, 1.0, 2.0})` → `{3.0, 1.0, 2.0}`.
- `rankVector({1.0, 1.0, 3.0})` → `{1.5, 1.5, 3.0}` (tied values average rank).
- `spearmanCorr(identical series)` → 1.0.
- `spearmanCorr(perfectly inverse series)` → -1.0.
- `spearmanCorr` on series with one extreme outlier produces a result in `[-1, 1]` and is closer to the true monotonic relationship than `pearsonCorr` on the same data.

---

## New ADR Required

**ADR-023: Spearman over Pearson for correlation discount**

- **Decision:** Replace `pearsonCorr` with `spearmanCorr` in `Portfolio::computeCorrelation()`.
- **Rationale:** Financial return series have fat tails. A single outlier return day can inflate or deflate Pearson correlation, causing the position sizing discount to be applied incorrectly. Spearman rank correlation is O(n log n) for the ranking step and then reduces to Pearson on ranks — similar computational cost but materially more robust to outlier contamination.
- **Trade-offs:** Backtest results are not comparable to pre-Phase-4 outputs on the same symbol set. All benchmarks and the README results table must be regenerated after this change.

---

## Files Changed Summary

| File | Change Type |
|------|-------------|
| `backtester/include/config/BacktestConfig.hpp` | Add: `allow_short`, `short_margin_rate`, `feature_schema_json` |
| `backtester/include/strategy/MLStrategy.hpp` | Add: `positionDirection_`; accept `shared_ptr<Module>` |
| `backtester/src/strategy/MLStrategy.cpp` | Implement SHORT signal; shared model inference |
| `backtester/include/portfolio/Portfolio.hpp` | Add: short position tracking, margin, Spearman |
| `backtester/src/portfolio/Portfolio.cpp` | Short accounting; Spearman correlation |
| `backtester/src/execution/SimulatedExecution.cpp` | SHORT/COVER fill price calculation |
| `backtester/include/market/FeatureCSVDataHandler.hpp` | Add: schema validation path |
| `backtester/src/market/FeatureCSVDataHandler.cpp` | Load and validate `feature_schema.json` |
| `backtester/ml_main.cpp` | Load model once; pass `shared_ptr` to strategies |
| `backtester/CMakeLists.txt` | Add: `nlohmann/json` via `FetchContent` |
| `backtester/tests/test_portfolio.cpp` | Add: short position tests, Spearman tests |
| `backtester/tests/test_execution.cpp` | Add: SHORT/COVER fill tests |
| `research/exportModel.py` | Export `models/feature_schema.json` |
| `research/features/pipeline.py` | Add: `--workers` parallel processing |
| `run_pipeline.py` | Add: `--pipeline-workers` flag |
| `backtest_config.yaml` | Add: `allow_short`, `short_margin_rate`, `feature_schema_json` |
| `DECISIONS.md` | Add: ADR-023 (Spearman), ADR-024 (shared model) |
| `requirements.txt` | No new dependencies (uses stdlib `multiprocessing`) |

---

## Definition of Done

Phase 4 is complete when:
1. A backtest with `allow_short: true` on a 1-year declining symbol produces ≥ 1 short trade with correct P&L
2. A backtest with `allow_short: false` produces byte-for-byte identical output to the pre-Phase-4 binary
3. `ml_main.cpp` has a single `torch::jit::load()` call regardless of symbol count (confirmed by `grep -c "torch::jit::load"`)
4. `python pipeline.py data/ -o features/ --workers 4` produces identical output to `--workers 1` and is measurably faster on 10+ symbols
5. `models/feature_schema.json` exists after `run_pipeline.py`; C++ startup validates it
6. Running C++ with a mismatched schema exits with a column-level error message before any inference
7. `pytest tests/` passes (all new Python tests)
8. `ctest --test-dir build` passes (all new C++ tests)
9. ADR-023 and ADR-024 added to `DECISIONS.md`
10. README results table regenerated with Spearman-based sizing (note the change)
