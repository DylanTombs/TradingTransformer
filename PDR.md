# Product Development Roadmap (PDR)

**Project:** TradingTransformer  
**Last Updated:** 2026-04-01  
**Status:** Active Development  
**Author:** Dylan Tombs

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Success Metrics](#4-success-metrics)
5. [Current System State](#5-current-system-state)
6. [Known Issues Register](#6-known-issues-register)
7. [Technical Debt Inventory](#7-technical-debt-inventory)
8. [Phase Overview](#8-phase-overview)
9. [Risk Register](#9-risk-register)
10. [Dependency Graph](#10-dependency-graph)
11. [Deferred Items](#11-deferred-items)

---

## 1. Executive Summary

TradingTransformer is a production-grade ML backtesting system built across two independently deployable layers: a Python research layer (feature engineering, transformer training, TorchScript export) and a C++ execution layer (event-driven backtesting engine with LibTorch inference). The architecture is sound — typed event hierarchy, risk-based position sizing, correlation-aware allocation, realistic slippage model, and dual-layer CI — but the system has not yet reached the standard required to trust its reported results or deploy with confidence.

**The core problem:** Three categories of issues currently make the reported backtest results untrustworthy and the system incompletely validated:

1. **Correctness gaps** — a known data leakage bug (`xfail`), undetected date gaps in the data pipeline, and critical untested components (slippage execution, performance metrics, CSV parsing).
2. **Validation methodology** — all results are in-sample on a fixed historical window with no walk-forward assessment, manual hyperparameters, and non-reproducible training.
3. **Observability deficit** — raw CSV outputs only, unstructured `std::cout` logging, no performance tearsheet, and no configurable signal thresholds.

This roadmap addresses those categories in strict priority order across four phases, with explicit acceptance criteria and measurable exit conditions at each gate.

---

## 2. Problem Statement

### 2.1 Why results cannot currently be trusted

The README results table (BX +70.81%, UNH +512.22%, etc.) was produced **before** the slippage model and risk-based sizing were introduced. No updated results exist. Additionally:

- The training/inference data split is a fixed 70/15/15 split on a single historical window. There is no out-of-sample evaluation period that the model never touched during development.
- A known ticker-boundary data leakage bug (`test_dataset.py::xfail`) allows sliding windows to span two symbols during training, contaminating the model's learned representations.
- `MLStrategy` has a 0.0 exit threshold — any predicted decline triggers an EXIT — but a 0.5% buy threshold. These thresholds are hardcoded and cannot be tuned via config.

### 2.2 Why the system cannot be extended safely

- `SimulatedExecution`, `PerformanceMetrics`, `FeatureCSVDataHandler`, and `MultiAssetDataHandler` have no dedicated unit tests. Their correctness is assumed, not validated.
- Config YAML has no schema validation — invalid types silently receive defaults or crash with unhelpful messages.
- `SignalType::SHORT` is declared in the event hierarchy but unimplemented in both `Portfolio` and `SimulatedExecution`.

### 2.3 Why observability is insufficient

- The C++ engine uses raw `std::cout` with no levels, no timestamps, no structured format, and no output file. Diagnosing a backtest with unusual results requires reading the source.
- Performance output is three CSV files. There is no visual tearsheet — no monthly returns heatmap, no drawdown chart, no rolling Sharpe chart.

---

## 3. Goals & Non-Goals

### Goals

| # | Goal | Phase |
|---|------|-------|
| G-01 | All known bugs fixed; zero `xfail` tests | 1 |
| G-02 | 80%+ unit test coverage across both Python and C++ layers | 1 |
| G-03 | Config schema validation enforced at startup (both layers) | 1 |
| G-04 | Results table reflects current execution model (slippage + risk sizing) | 1 |
| G-05 | Walk-forward validation framework producing per-fold Sharpe/IR/drawdown | 2 |
| G-06 | Optuna sweep over transformer hyperparameters with logged results table | 2 |
| G-07 | Fully reproducible training (deterministic seeds, fixed environment) | 2 |
| G-08 | Structured `spdlog` logging in C++ engine at all levels | 3 |
| G-09 | HTML performance tearsheet auto-generated after every backtest run | 3 |
| G-10 | Signal thresholds (`buyThreshold`, `exitThreshold`) configurable via YAML | 3 |
| G-11 | Short selling fully implemented and tested end-to-end | 4 |
| G-12 | Feature column contract enforced by a schema file, not by convention | 4 |
| G-13 | Feature pipeline parallelised over symbols | 4 |
| G-14 | Shared `torch::jit::Module` across `MLStrategy` instances | 4 |

### Non-Goals

| Item | Reason |
|------|--------|
| Live trading integration | Separate project scope; operational complexity out of range |
| Intraday (minute/tick) simulation | Requires a distinct data ingestion pipeline |
| Factor-adjusted benchmark | Needs a market data API for factor returns |
| Square-root market impact model | Overkill for current order sizes per ADR-013 |
| DuckDB backend | Not needed until symbol count exceeds ~50 |
| Browser-based UI | Out of scope; CLI + HTML tearsheet is sufficient |

---

## 4. Success Metrics

All metrics must be measurable objectively. A phase is not complete until every metric in its set is satisfied.

### Phase 1 Gates
| Metric | Target | Measurement |
|--------|--------|-------------|
| Python test coverage | ≥ 80% | `pytest --cov` report |
| C++ unit tests | ≥ 80% line coverage | GCov/LCov report |
| `xfail` tests | 0 | `pytest` output, no `XFAIL` markers |
| Config schema violations caught | 100% of invalid inputs | Test suite with invalid YAML fixtures |
| Results table accuracy | Updated with current slippage + sizing | README diff confirmed |

### Phase 2 Gates
| Metric | Target | Measurement |
|--------|--------|-------------|
| Walk-forward folds | ≥ 5 expanding-window folds per symbol | `wf_results.csv` row count |
| Hyperparameter search trials | ≥ 50 Optuna trials | Optuna study database |
| Training reproducibility | Identical metrics across 3 runs with same seed | `diff` on metrics outputs |
| Out-of-sample Sharpe | Positive (> 0) on ≥ 3 of 5 symbols | Walk-forward summary table |

### Phase 3 Gates
| Metric | Target | Measurement |
|--------|--------|-------------|
| Log levels | `info`, `warn`, `error`, `debug` all used | Code review |
| Tearsheet components | ≥ 5 charts/panels | HTML inspection |
| Tearsheet generation time | < 10 seconds for 2-year backtest | `time` measurement |
| Signal threshold config | `buy_threshold` + `exit_threshold` in YAML | Integration test |

### Phase 4 Gates
| Metric | Target | Measurement |
|--------|--------|-------------|
| Short positions | Opens + closes correctly in backtest | Unit + integration tests |
| Feature schema | `feature_schema.json` validated at C++ startup | Mismatched schema test |
| Pipeline parallelism speedup | ≥ 2× faster for 10 symbols | `time` before/after |
| Memory reduction | Shared module uses < 2× single-symbol memory for 5 symbols | `valgrind` / `heaptrack` |

---

## 5. Current System State

### 5.1 Architecture

```
Python Research Layer                    C++ Execution Layer
─────────────────────────────────────    ─────────────────────────────────────
StockDataPD.py → data/*.csv             BacktestConfig (YAML parser)
pipeline.py    → features/*.csv    ────► FeatureCSVDataHandler
Train.py       → checkpoint.pt          MultiAssetDataHandler
exportModel.py → transformer.pt    ────► MLStrategy (LibTorch)
               → feature_scaler.csv      MultiSymbolStrategy
               → target_scaler.csv       BacktestEngine (event loop)
convert_scalers.py                       Portfolio + RiskManager
                                         SimulatedExecution
                                         PerformanceMetrics
                                    ────► ml_equity.csv
                                         ml_trades.csv
                                         ml_metrics.csv
```

### 5.2 Model Specification

| Parameter | Value | Location |
|-----------|-------|----------|
| Input features | 34 (33 indicators + close) | `exportModel.py::load_args()` |
| Encoder sequence length | 30 bars | `Train.py` |
| Decoder label length | 10 bars | `Train.py` |
| Prediction horizon | 1 bar (next-day close) | `Train.py` |
| Embedding dim (`d_model`) | 256 | `exportModel.py` |
| Attention heads (`n_heads`) | 8 | `exportModel.py` |
| Encoder layers | 3 | `exportModel.py` |
| Decoder layers | 2 | `exportModel.py` |
| FFN hidden dim (`d_ff`) | 512 | `exportModel.py` |
| Dropout | 0.1 | `exportModel.py` |
| Optimizer | Adam, lr=0.0005 | `Interface.py` |
| Loss | MSE | `Interface.py` |
| Early stopping patience | 10 epochs | `Interface.py` |
| Batch size | 128 | `Interface.py` |
| Max epochs | 100 | `Interface.py` |
| Data split | 70/15/15 | `Interface.py` |

### 5.3 Test Coverage Summary (Current)

| Layer | Covered | Not Covered |
|-------|---------|-------------|
| Python: model architecture | ✓ Forward pass, shapes, attention, eval determinism | Export to TorchScript |
| Python: dataset | ✓ Window generation, scaler consistency | Multi-symbol DataFrame, error paths |
| Python: indicators | ✓ RSI, MACD, OBV, Stochastic, ATR, volatility | Edge cases (NaN propagation, all-zero inputs) |
| Python: training loop | ✓ Legacy backtrader metrics | End-to-end train → export integration |
| C++: portfolio | ✓ Cash, fills, sizing, correlation, benchmark | — |
| C++: engine | ✓ Event dispatch, multi-symbol routing | — |
| C++: execution | ✗ Slippage, commission, market impact | All |
| C++: metrics | ✗ Sharpe, IR, drawdown, alpha | All |
| C++: data handler | ✗ CSV parsing, column validation, edge cases | All |
| C++: multi-asset sync | ✗ Timestamp synchronisation, gaps, misaligned symbols | All |

### 5.4 CI/CD Pipeline

| Workflow | Trigger | Checks |
|----------|---------|--------|
| `python-app.yml` | push/PR to main | `flake8` (syntax only) + `pytest tests/ -v` |
| `build.yml` | push/PR to main | `cmake` configure + build + `ctest` |
| `codeql.yml` | push/PR + weekly | CodeQL static analysis (Python + C++) |

**Gaps:** No coverage reporting, no linting beyond syntax errors (`flake8` does not check style), no integration test stage, no performance regression check.

---

## 6. Known Issues Register

| ID | Title | Severity | Component | Status | Phase |
|----|-------|----------|-----------|--------|-------|
| B-01 | Ticker-boundary data leakage in `DataFrameDataset` | **Critical** | `research/transformer/DataFrame.py` | `xfail` in test | 1 |
| B-02 | Date-gap blindness in `FeatureCSVDataHandler` | **High** | `backtester/src/market/FeatureCSVDataHandler.cpp` | Open | 1 |
| B-03 | Results table pre-dates slippage + risk-sizing model | **High** | `README.md` | Open | 1 |
| B-04 | `buyThreshold` (0.5%) and `exitThreshold` (0.0%) hardcoded in `MLStrategy` | **Medium** | `backtester/include/strategy/MLStrategy.hpp` | Open | 3 |
| B-05 | `SignalType::SHORT` declared but not handled | **Medium** | `Portfolio.cpp`, `SimulatedExecution.cpp` | Open | 4 |
| B-06 | Each `MLStrategy` loads the model file independently per symbol | **Low** | `backtester/src/strategy/MLStrategy.cpp` | Open | 4 |
| B-07 | `MultiAssetDataHandler` assumes ISO 8601 timestamps — mixed formats cause silent incorrect ordering | **Low** | `backtester/src/market/MultiAssetDataHandler.cpp` | Open | 1 |

---

## 7. Technical Debt Inventory

| ID | Item | Category | Priority | Phase |
|----|------|----------|----------|-------|
| TD-01 | `flake8` checks syntax only — no style (PEP8), no type hints, no complexity | Testing | High | 1 |
| TD-02 | `SimulatedExecution` has zero unit tests — slippage and commission unvalidated | Testing | Critical | 1 |
| TD-03 | `PerformanceMetrics` has zero unit tests — Sharpe, IR, drawdown unvalidated | Testing | Critical | 1 |
| TD-04 | `FeatureCSVDataHandler` has zero unit tests — CSV parsing and column validation unvalidated | Testing | High | 1 |
| TD-05 | `MultiAssetDataHandler` has zero unit tests — timestamp sync unvalidated | Testing | High | 1 |
| TD-06 | Config YAML has no schema — invalid keys and wrong types produce silent defaults or unhelpful crashes | Reliability | High | 1 |
| TD-07 | Training is non-deterministic — no seed exposed, CUDA non-determinism not suppressed | Reliability | High | 2 |
| TD-08 | Feature column order is convention, not a contract — divergence produces size-mismatch errors only | Reliability | Medium | 4 |
| TD-09 | `pipeline.py` processes symbols serially — O(n) with no parallelism | Performance | Low | 4 |
| TD-10 | No CI coverage reporting — coverage regressions go undetected | CI/CD | Medium | 1 |
| TD-11 | `std::cout` in all C++ components — no levels, no timestamps, no output file | Observability | Medium | 3 |
| TD-12 | `backtrader` in `requirements.txt` — legacy dependency, only used in `test_metrics.py` | Maintenance | Low | 2 |
| TD-13 | `research/training/Train.py` partially superseded by `Interface.py` — two training entry points | Maintenance | Low | 2 |

---

## 8. Phase Overview

| Phase | Title | Focus | Key Deliverables |
|-------|-------|-------|-----------------|
| **1** | Correctness & Data Integrity | Fix all bugs, close test gaps, enforce config schema | Zero xfails, 80%+ coverage, schema validation, updated results |
| **2** | Validation Methodology | Walk-forward, hyperparameter sweep, reproducibility | WF report, Optuna results table, deterministic training |
| **3** | Observability & Production Output | Structured logging, HTML tearsheet, configurable thresholds | spdlog integration, auto-generated tearsheet |
| **4** | Strategy Extension & Scalability | Short selling, shared model, parallel pipeline, schema contract | SHORT trades, feature schema, 2× pipeline speedup |

---

## 9. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|------------|
| R-01 | Walk-forward reveals model is overfit to training window — out-of-sample Sharpe < 0 | Medium | High | Accept result; use Phase 2 Optuna sweep to find more generalisable config |
| R-02 | Ticker-boundary leakage fix materially changes model performance on affected symbols | Medium | High | Re-run full backtest after fix; document delta vs. pre-fix results |
| R-03 | `spdlog` integration breaks LibTorch linkage on some platforms | Low | Medium | Test on Ubuntu CI before merging; keep spdlog as header-only |
| R-04 | Optuna sweep requires significant compute time (50+ trials × 100 epochs) | High | Low | Set `max_epochs=30` for sweep trials; full training only on best config |
| R-05 | Feature schema contract change breaks existing feature CSVs | Low | High | Schema is additive-only in Phase 4; existing CSVs validated against schema at load time |
| R-06 | Short position accounting introduces bugs in equity curve calculation | Medium | Medium | TDD: write tests for short P&L, margin, exit before implementing |
| R-07 | Parallel pipeline introduces non-deterministic feature CSV ordering | Low | Medium | Sort output by symbol name after pool completes |

---

## 10. Dependency Graph

```
Phase 1: Correctness & Data Integrity
│  ├── Fix B-01 (ticker leakage) → removes xfail from test_dataset.py
│  ├── Fix B-02 (date gaps) → adds warning in FeatureCSVDataHandler
│  ├── Add schema validation (TD-06) → Pydantic for Python, nlohmann/json for C++
│  ├── Fill test coverage gaps (TD-02, TD-03, TD-04, TD-05)
│  ├── Upgrade CI: coverage reporting (TD-10), flake8 style (TD-01)
│  └── Re-run results (B-03)
│
└──► Phase 2: Validation Methodology  [requires Phase 1 complete — results must be trustworthy]
     │  ├── Walk-forward framework → per-fold metrics CSV
     │  ├── Optuna sweep → best hyperparameter config
     │  ├── Reproducible training → seed exposed in config
     │  └── Consolidate Train.py / Interface.py (TD-13), remove backtrader dep (TD-12)
     │
     └──► Phase 3: Observability & Production Output  [requires Phase 1; independent of Phase 2]
          │  ├── spdlog integration → structured log output
          │  ├── HTML tearsheet → auto-generated from CSV outputs
          │  └── Configurable thresholds (B-04) → buy_threshold, exit_threshold in YAML
          │
          └──► Phase 4: Strategy Extension & Scalability  [requires Phase 1; independent of 2 & 3]
               ├── SHORT implementation (B-05) → Portfolio + SimulatedExecution
               ├── Feature schema contract (TD-08) → feature_schema.json
               ├── Parallel pipeline (TD-09) → multiprocessing.Pool
               └── Shared model reference (B-06) → std::shared_ptr<torch::jit::Module>

Note: Phase 3 and Phase 4 can proceed in parallel after Phase 1.
      Phase 2 can proceed in parallel with Phase 3 and Phase 4.
```

---

## 11. Deferred Items

Items that are out of scope for this roadmap cycle, with reasons:

| Item | Decision | Revisit |
|------|----------|---------|
| Live trading integration | Separate project. Requires brokerage API, order management, position reconciliation | After Phase 4 complete |
| Intraday bar simulation | Requires tick/minute data ingestion pipeline, separate from EOD | Not planned |
| Factor-adjusted benchmark | Needs factor returns API (Fama-French, BARRA) | Not planned |
| DuckDB backend for large symbol sets | Not needed until symbol count > 50 | Phase 4+ if needed |
| Attention visualisation / interpretability | Research task; not a production requirement | Not planned |
| Ensemble models | Architecturally feasible but requires model versioning | Not planned |
| Regime detection | Separate model; significant research investment | Not planned |
