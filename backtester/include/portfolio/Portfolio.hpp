#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "../events/Events.hpp"
#include "../events/SignalEvent.hpp"
#include "../events/OrderEvent.hpp"
#include "../events/MarketEvent.hpp"
#include "../events/FillEvent.hpp"

// ---------------------------------------------------------------------------
// EquityPoint — one row of the equity curve, including the passive benchmark.
// ---------------------------------------------------------------------------
struct EquityPoint {
    std::string timestamp;
    double      equity          = 0.0; ///< Strategy net liquidation value
    double      price           = 0.0; ///< Symbol close price at this bar
    double      benchmarkEquity = 0.0; ///< Buy-and-hold of same symbol from t=0
};

// ---------------------------------------------------------------------------
// Portfolio
//
// Responsibilities:
//   - Track cash and positions after each fill
//   - Generate orders from signals using risk-based position sizing
//   - Record an equity curve (strategy vs buy-and-hold benchmark)
//   - Export equity curve and trade log to CSV
//
// Design notes:
//   - EXIT signals fully close the open position (sell all held shares)
//   - LONG quantity = floor(equity * riskFraction / price)
//   - Benchmark is a passive buy-and-hold from the first bar's price
// ---------------------------------------------------------------------------
class Portfolio {
public:
    explicit Portfolio(double initialCash, double riskFraction = 0.10);

    // ---- Trade record -------------------------------------------------------
    struct Trade {
        std::string timestamp;
        double      price;
        int         quantity;
        std::string direction; ///< "BUY" or "SELL"
        bool        profit;    ///< true if fill price > last buy price
    };

    // ---- Core event handlers ------------------------------------------------

    void updateMarket(const MarketEvent& event);

    /**
     * Converts a signal into an order:
     *   LONG  → BUY  qty = floor(equity * riskFraction / price)
     *   SHORT → SELL qty = floor(equity * riskFraction / price)
     *   EXIT  → SELL qty = current position for that symbol (full close)
     *
     * Returns an OrderType::HOLD with qty=0 when no action should be taken
     * (e.g. EXIT with no open position, or price data unavailable).
     */
    OrderEvent generateOrder(const SignalEvent& signal);

    void updateFill(const FillEvent& fill);

    // ---- Accessors ----------------------------------------------------------
    double getCash()                              const;
    int    getPosition(const std::string& symbol) const;
    double getTotalEquity()                       const;

    const std::vector<EquityPoint>& getEquityCurve() const;
    const std::vector<Trade>&       getTrades()       const;

    // ---- Export -------------------------------------------------------------
    void exportEquityCurve(const std::string& filename) const;
    void exportTrades     (const std::string& filename) const;

private:
    double initialCash_;
    double cash_;
    double riskFraction_;

    // Benchmark: buy-and-hold from the first bar
    bool   benchmarkInitialised_   = false;
    double benchmarkInitialPrice_  = 0.0;

    double lastBuyPrice_ = 0.0;

    std::unordered_map<std::string, double> latestPrices_;
    std::unordered_map<std::string, int>    positions_;
    std::unordered_map<std::string, std::string> latestTimestamps_;

    std::vector<EquityPoint> equityCurve_;
    std::vector<Trade>       trades_;
};
