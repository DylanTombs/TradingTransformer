#pragma once

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// SymbolConfig — path/symbol pair for one instrument in the backtest
// ---------------------------------------------------------------------------
struct SymbolConfig {
    std::string symbol;
    std::string featureCsv;
};

// ---------------------------------------------------------------------------
// BacktestConfig — flat runtime configuration.
//
// Loaded from the YAML config file.  All fields have sensible defaults so
// the engine runs in tests without a config file.
//
// Multi-symbol YAML layout (backward-compatible):
//   symbol:          AAPL          ← first symbol (required)
//   feature_csv:     /f/AAPL.csv   ← first symbol CSV
//   symbol_1:        MSFT          ← additional symbols (optional)
//   feature_csv_1:   /f/MSFT.csv
//   symbol_2:        GOOGL
//   feature_csv_2:   /f/GOOGL.csv
//   ...up to symbol_19 / feature_csv_19
// ---------------------------------------------------------------------------
struct BacktestConfig {

    // ---- Symbols -----------------------------------------------------------
    std::vector<SymbolConfig> symbols;          ///< Populated from YAML
    std::string               modelPt;          ///< Path to transformer.pt
    std::string               featScalerCsv;    ///< Path to feature_scaler.csv
    std::string               targScalerCsv;    ///< Path to target_scaler.csv
    std::string               outputDir;        ///< Directory for CSV output

    // ---- Capital & sizing --------------------------------------------------
    double initialCash       = 100'000.0; ///< Starting portfolio capital ($)
    double riskFraction      = 0.10;      ///< Fraction of equity per LONG (10%)
    double maxSymbolExposure = 0.20;      ///< Max single-symbol weight (20%)
    double maxTotalExposure  = 0.80;      ///< Max total position weight (80%)
    int    maxPositionSize   = 10'000;    ///< Hard per-symbol share cap

    // ---- Execution realism -------------------------------------------------
    double halfSpread        = 0.0005;    ///< Half bid-ask spread per fill
    double slippageFraction  = 0.0005;    ///< Directional slippage per trade
    double marketImpact      = 0.0;       ///< Price impact per share ($)
    double commission        = 1.0;       ///< Fixed commission per fill ($)

    // ---- Risk / performance ------------------------------------------------
    double riskFreeRate       = 0.0;  ///< Annualised risk-free rate (Sharpe base)
    int    correlationWindow  = 60;   ///< Rolling days for correlation matrix
    double correlationThreshold = 0.7; ///< Correlation above this discounts size

    // ---- Factory -----------------------------------------------------------
    static BacktestConfig loadFromYAML(const std::string& path) {
        BacktestConfig cfg;
        std::ifstream file(path);
        if (!file.is_open()) return cfg;

        auto kv = parseKeyValues(file);

        auto setDouble = [&](const std::string& key, double& f) {
            if (kv.count(key)) f = std::stod(kv.at(key));
        };
        auto setInt = [&](const std::string& key, int& f) {
            if (kv.count(key)) f = std::stoi(kv.at(key));
        };
        auto setStr = [&](const std::string& key, std::string& f) {
            if (kv.count(key)) f = kv.at(key);
        };

        // Paths
        setStr("model_pt",          cfg.modelPt);
        setStr("feature_scaler_csv",cfg.featScalerCsv);
        setStr("target_scaler_csv", cfg.targScalerCsv);
        setStr("output_dir",        cfg.outputDir);

        // Capital
        setDouble("initial_cash",       cfg.initialCash);
        setDouble("risk_fraction",      cfg.riskFraction);
        setDouble("max_symbol_exposure",cfg.maxSymbolExposure);
        setDouble("max_total_exposure", cfg.maxTotalExposure);
        setInt   ("max_position_size",  cfg.maxPositionSize);

        // Execution
        setDouble("half_spread",        cfg.halfSpread);
        setDouble("slippage_fraction",  cfg.slippageFraction);
        setDouble("market_impact",      cfg.marketImpact);
        setDouble("commission",         cfg.commission);

        // Risk
        setDouble("risk_free_rate",        cfg.riskFreeRate);
        setInt   ("correlation_window",    cfg.correlationWindow);
        setDouble("correlation_threshold", cfg.correlationThreshold);

        // Symbol list: primary symbol + symbol_1..symbol_19
        if (kv.count("symbol") && kv.count("feature_csv"))
            cfg.symbols.push_back({kv.at("symbol"), kv.at("feature_csv")});

        for (int i = 1; i <= 19; ++i) {
            const std::string sk = "symbol_"      + std::to_string(i);
            const std::string fk = "feature_csv_" + std::to_string(i);
            if (kv.count(sk) && kv.count(fk))
                cfg.symbols.push_back({kv.at(sk), kv.at(fk)});
        }

        return cfg;
    }

private:
    static std::unordered_map<std::string, std::string>
    parseKeyValues(std::istream& stream) {
        std::unordered_map<std::string, std::string> result;
        std::string line;
        while (std::getline(stream, line)) {
            auto comment = line.find('#');
            if (comment != std::string::npos) line = line.substr(0, comment);
            auto colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string key   = trim(line.substr(0, colon));
            std::string value = trim(line.substr(colon + 1));
            if (!key.empty() && !value.empty()) result[key] = value;
        }
        return result;
    }

    static std::string trim(const std::string& s) {
        const auto start = s.find_first_not_of(" \t\r\n");
        const auto end   = s.find_last_not_of(" \t\r\n");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    }
};
