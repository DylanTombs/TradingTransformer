#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

/**
 * Runtime configuration for the backtesting engine.
 *
 * Loaded from the YAML config file (shared with docker/entrypoint.sh).
 * Only flat  key: value  pairs are parsed — no nested YAML.
 * All fields have conservative defaults so the engine runs correctly without
 * a config file (e.g. in unit tests).
 *
 * YAML keys consumed (all optional):
 *   initial_cash       — starting portfolio capital in USD
 *   risk_fraction      — fraction of equity invested per LONG signal (0–1)
 *   half_spread        — half the bid-ask spread applied to each fill
 *   slippage_fraction  — directional slippage fraction per trade
 *   market_impact      — additional price impact per share traded (0 = off)
 *   commission         — fixed commission per fill in USD
 *   max_position_size  — hard per-symbol position cap (RiskManager gate)
 */
struct BacktestConfig {

    // ---- Capital & sizing ---------------------------------------------------
    double initialCash     = 100'000.0; ///< Starting portfolio capital ($)
    double riskFraction    = 0.10;      ///< Fraction of equity per LONG (10%)
    int    maxPositionSize = 10'000;    ///< Hard cap: RiskManager rejects above this

    // ---- Execution realism --------------------------------------------------
    double halfSpread       = 0.0005;   ///< Half bid-ask spread, e.g. 0.05%
    double slippageFraction = 0.0005;   ///< Directional price slippage, e.g. 0.05%
    double marketImpact     = 0.0;      ///< Price impact per share (0 = disabled)
    double commission       = 1.0;      ///< Fixed commission per fill ($)

    // -------------------------------------------------------------------------
    // Factory: load from a flat YAML file.
    // Returns a default-initialised config if the file cannot be opened.
    // -------------------------------------------------------------------------
    static BacktestConfig loadFromYAML(const std::string& path) {
        BacktestConfig cfg;
        std::ifstream file(path);
        if (!file.is_open()) return cfg;

        auto kv = parseKeyValues(file);
        auto setDouble = [&](const std::string& key, double& field) {
            if (kv.count(key)) field = std::stod(kv.at(key));
        };
        auto setInt = [&](const std::string& key, int& field) {
            if (kv.count(key)) field = std::stoi(kv.at(key));
        };

        setDouble("initial_cash",      cfg.initialCash);
        setDouble("risk_fraction",     cfg.riskFraction);
        setDouble("half_spread",       cfg.halfSpread);
        setDouble("slippage_fraction", cfg.slippageFraction);
        setDouble("market_impact",     cfg.marketImpact);
        setDouble("commission",        cfg.commission);
        setInt   ("max_position_size", cfg.maxPositionSize);

        return cfg;
    }

private:
    // Parses  key: value  lines, stripping inline comments and whitespace.
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
