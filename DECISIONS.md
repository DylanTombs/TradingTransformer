# Decision Log

A record of key architectural and implementation decisions. Ordered by subsystem.

---

## System Architecture

### ADR-001: Separate research (Python) and execution (C++) layers

**Decision:** All model development, feature engineering, and training runs in Python. All backtest simulation runs in C++. The boundary is a CSV contract (feature files) and a TorchScript model artefact.

**Rationale:** Keeping the two layers separate makes each independently testable and deployable. The Python layer benefits from the scientific Python ecosystem (pandas, sklearn, PyTorch). The C++ layer benefits from deterministic memory management, no GIL, and a build system that can produce a self-contained binary with no runtime dependencies other than LibTorch.

**Trade-offs:**
- Adds a serialisation step (export) that must be re-run whenever the model or feature set changes.
- The feature column order is a shared assumption maintained by convention (`exportModel.py::load_args()` is the canonical source). A schema registry would be more robust at scale.

---

## Feature Engineering

### ADR-002: Bar-by-bar adapter over batch pandas operations

**Decision:** `pipeline.py` wraps the DataFrame in a `_BarCtx` / `_DataView` / `_Line` adapter and computes indicators bar-by-bar rather than using vectorised pandas operations.

**Rationale:** The indicator functions in `technicalIndicators.py` are written for backtrader's API (`self.data.close[-1]`, `self.data.high[0]`, etc.). Re-implementing them in pandas would create two code paths that could diverge silently. The adapter calls the existing functions unchanged, making training and inference feature parity a structural guarantee rather than a tested assumption.

**Trade-offs:**
- Slower than vectorised pandas (O(n) Python loop instead of C-accelerated array ops).
- For the data sizes involved (daily bars over a few years per symbol), the performance difference is negligible.
- The adapter is more complex than a simple pandas implementation would be.

---

## Model

### ADR-003: Encoder-decoder Transformer over LSTM

**Decision:** Use a custom encoder-decoder Transformer (`seqLen=30`, `predLen=5`, `encIn=34`) rather than an LSTM or simpler model.

**Rationale:** The multi-head attention mechanism can learn arbitrary long-range temporal dependencies without the vanishing gradient problem. For financial time series — where regime context from 20+ bars ago may be more relevant than recent noise — this is a meaningful advantage. The encoder-decoder structure naturally separates historical context from the prediction horizon.

**Trade-offs:**
- Requires `seqLen=30` bars before the first signal can be emitted. The first 30 bars of every backtest are always skipped.
- Larger model (~23 MB) with more training parameters. Needs more data and regularisation than an LSTM to avoid overfitting.
- Slower training and inference than a linear model or shallow network.

### ADR-004: `torch.jit.trace` over `torch.jit.script`

**Decision:** Export the model using `torch.jit.trace` with fixed dummy inputs of shape `(1, 30, 34)`.

**Rationale:** `trace` requires no source-level annotation and works for any model that has no data-dependent control flow. `script` requires the entire model to be type-annotated and free of Python constructs that cannot be serialised. The model uses standard `nn.Module` composition — `trace` is sufficient and simpler.

**Trade-offs:**
- The traced graph is frozen to the input shapes used at trace time. Changing `seqLen` or `encIn` requires a re-export. This is acceptable because these are fixed hyperparameters defined in `exportModel.py::load_args()`.
- Dynamic batching is not possible with the traced model. Inference runs one sample at a time (`batch=1`), which is correct for a streaming bar-by-bar strategy.

---

## C++ Engine

### ADR-005: `FeatureMarketEvent` inherits `MarketEvent`

**Decision:** `FeatureMarketEvent` is a subclass of `MarketEvent` rather than a parallel event type.

**Rationale:** `BacktestEngine` dispatches on `EventType::MARKET` and upcasts to `MarketEvent`. A parallel type would require modifying the engine switch statement — a change unrelated to ML. Inheritance preserves the open/closed principle: the engine is unchanged, and `MLStrategy` recovers the feature payload with `dynamic_cast`. If the cast returns `nullptr` (plain `MarketEvent`), the strategy produces no signal — a safe, explicit failure mode rather than undefined behaviour.

**Trade-offs:**
- `dynamic_cast` at runtime has a small cost per bar.
- Any strategy that needs feature data must know to cast. This is documented in `FeatureMarketEvent.hpp`.

### ADR-006: Header-only `ScalerParams`

**Decision:** `ScalerParams` is implemented entirely in `ScalerParams.hpp` with no corresponding `.cpp`.

**Rationale:** The struct has no mutable state, no virtual methods, and fits in ~60 lines of straightforward arithmetic. Header-only removes a compilation unit and allows the struct to be included by any target without linking.

**Trade-offs:**
- If the implementation grows (e.g. adding RobustScaler support), the header-only approach will increase binary size from inlining. At that point it should be split.

### ADR-007: `RiskManager` as a pre-trade gate

**Decision:** A `RiskManager` sits between `Portfolio::generateOrder` and `SimulatedExecution::executeOrder`. Orders are only executed if `approveOrder` returns true.

**Rationale:** Separating position-limit logic from portfolio accounting keeps `Portfolio` focused on state tracking. The risk manager can be extended to enforce drawdown limits, position concentration caps, or sector exposure rules without modifying the portfolio class.

**Trade-offs:**
- Currently `RiskManager` only enforces a max position size. The interface is future-proof but the implementation is minimal.

---

## Testing

### ADR-008: `_IsolatedEvaluator` test double for `StrategyEvaluator`

**Decision:** `test_metrics.py` does not instantiate `StrategyEvaluator` directly. Instead, an `_IsolatedEvaluator` class is created and the metric methods from `StrategyEvaluator` are bound onto it.

**Rationale:** `StrategyEvaluator` inherits from `bt.Analyzer`. Backtrader's metaclass assigns `datas` and `_obj` attributes at instantiation that require a live `cerebro` environment. Instantiating the class in a test context throws immediately. The test double bypasses the metaclass by binding the methods onto a plain class that supplies the minimum attributes the methods actually access.

**Trade-offs:**
- The test double must be updated if `StrategyEvaluator` adds methods that access new backtrader-specific attributes.
- The technique is not obvious to a new contributor — it is documented in `tests/test_metrics.py`.

### ADR-009: Google Test via CMake `FetchContent`

**Decision:** The C++ test suite uses Google Test, fetched automatically at configure time via `FetchContent_Declare`.

**Rationale:** Requiring contributors to install Google Test manually is a friction point. `FetchContent` downloads and builds GTest as part of the CMake configure step. The version is pinned (`v1.14.0`), so builds are reproducible across machines.

**Trade-offs:**
- First configure requires a network connection and takes longer.
- The downloaded source is stored in `_deps/` (gitignored) and re-used on subsequent configures.

---

## CI/CD

### ADR-010: Three separate workflow files

**Decision:** Python tests, C++ build/test, and CodeQL are in separate workflow files rather than a single multi-job workflow.

**Rationale:** Separate files make individual workflow failures immediately identifiable by badge. A Python test failure does not block the C++ build result from being reported. Jobs within a single workflow have an implied dependency hierarchy that would require careful use of `needs:` to achieve the same isolation.

**Trade-offs:**
- Three files to maintain instead of one.
- No shared secrets or artefacts between workflows without using the GitHub Actions artefact store.
