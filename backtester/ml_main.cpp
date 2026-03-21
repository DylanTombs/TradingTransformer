/**
 * ml_main.cpp — ML backtester entry point.
 *
 * Wires together:
 *   FeatureCSVDataHandler  (reads pipeline-enriched CSV)
 *   →  BacktestEngine
 *   →  MLStrategy           (transformer inference via LibTorch)
 *
 * Usage:
 *   ./ml_backtest <feature_csv> <symbol>
 *                 [model_pt]            default: models/transformer.pt
 *                 [feature_scaler_csv]  default: models/feature_scaler.csv
 *                 [target_scaler_csv]   default: models/target_scaler.csv
 *
 * The feature CSV is produced by research/features/pipeline.py.
 * The model and scaler files are produced by research/exportModel.py.
 *
 * Feature column order must match the training configuration in
 * research/exportModel.py::load_args() (auxilFeatures + [target]).
 */

#include "engine/BacktestEngine.hpp"
#include "market/FeatureCSVDataHandler.hpp"
#include "strategy/MLStrategy.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Feature column list — must match load_args() in research/exportModel.py
// Order: auxilFeatures (index 0..N-1) then target ('close')
// ---------------------------------------------------------------------------
static const std::vector<std::string> MODEL_FEATURE_COLUMNS = {
    "high", "low", "volume", "adj close",
    "P", "R1", "R2", "R3", "S1", "S2", "S3",
    "obv", "volume_zscore",
    "rsi", "macd", "macds", "macdh",
    "sma", "lma", "sema", "lema",
    "overnight_gap",
    "return_lag_1", "return_lag_3", "return_lag_5",
    "volatility",
    "SR_K", "SR_D", "SR_RSI_K", "SR_RSI_D",
    "ATR", "HL_PCT", "PCT_CHG",
    "close"   // target — always last
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <feature_csv> <symbol>"
                     " [model_pt] [feature_scaler_csv] [target_scaler_csv]\n";
        return 1;
    }

    const std::string csvPath     = argv[1];
    const std::string symbol      = argv[2];
    const std::string modelPath   = (argc > 3) ? argv[3] : "models/transformer.pt";
    const std::string featScaler  = (argc > 4) ? argv[4] : "models/feature_scaler.csv";
    const std::string targScaler  = (argc > 5) ? argv[5] : "models/target_scaler.csv";

    std::cout << "=== ML Backtest ===" << std::endl;
    std::cout << "  CSV:     " << csvPath    << std::endl;
    std::cout << "  Symbol:  " << symbol     << std::endl;
    std::cout << "  Model:   " << modelPath  << std::endl;
    std::cout << "  FeatSc:  " << featScaler << std::endl;
    std::cout << "  TgtSc:   " << targScaler << std::endl;

    try {
        FeatureCSVDataHandler data(
            csvPath, symbol,
            MODEL_FEATURE_COLUMNS,
            /*closeColumn=*/ "close",
            /*dateColumn=*/  "date");

        MLStrategy strategy(
            modelPath, featScaler, targScaler,
            /*seqLen=*/        30,
            /*nFeatures=*/     static_cast<int>(MODEL_FEATURE_COLUMNS.size()),
            /*buyThreshold=*/  0.005,
            /*exitThreshold=*/ 0.0);

        BacktestEngine engine(strategy, data);
        engine.run();

        // ------------------------------------------------------------------
        // Summary
        // ------------------------------------------------------------------
        auto& portfolio = engine.getPortfolio();
        auto& curve     = portfolio.getEquityCurve();
        auto& trades    = portfolio.getTrades();

        if (curve.empty()) {
            std::cout << "\nNo market data processed." << std::endl;
            return 0;
        }

        const double startEquity = std::get<1>(curve.front());
        const double endEquity   = std::get<1>(curve.back());
        const double totalReturn = (endEquity / startEquity - 1.0) * 100.0;

        int wins = 0;
        for (const auto& t : trades)
            if (t.profit) ++wins;

        std::cout << "\n=== Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Start equity : $" << startEquity  << std::endl;
        std::cout << "  End equity   : $" << endEquity    << std::endl;
        std::cout << "  Total return : "  << totalReturn  << " %" << std::endl;
        std::cout << "  Total trades : "  << trades.size() << std::endl;
        if (!trades.empty())
            std::cout << "  Win rate     : "
                      << (100.0 * wins / static_cast<double>(trades.size()))
                      << " %" << std::endl;

        portfolio.exportEquityCurve("ml_equity.csv");
        portfolio.exportTrades("ml_trades.csv");
        std::cout << "\nSaved: ml_equity.csv  ml_trades.csv" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
