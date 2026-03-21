#include "../../include/portfolio/Portfolio.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Portfolio::Portfolio(double initialCash, double riskFraction)
    : initialCash_(initialCash)
    , cash_(initialCash)
    , riskFraction_(riskFraction)
{}

// ---------------------------------------------------------------------------
// Market update — called once per bar before strategy fires
// ---------------------------------------------------------------------------

void Portfolio::updateMarket(const MarketEvent& event) {
    latestPrices_[event.symbol]     = event.price;
    latestTimestamps_[event.symbol] = event.timestamp;

    // Initialise buy-and-hold benchmark at the first bar seen
    if (!benchmarkInitialised_) {
        benchmarkInitialPrice_ = event.price;
        benchmarkInitialised_  = true;
    }

    const double benchmarkEquity =
        initialCash_ * (event.price / benchmarkInitialPrice_);

    equityCurve_.push_back({
        event.timestamp,
        getTotalEquity(),
        event.price,
        benchmarkEquity
    });
}

// ---------------------------------------------------------------------------
// Order generation — signal → order (no execution here)
// ---------------------------------------------------------------------------

OrderEvent Portfolio::generateOrder(const SignalEvent& signal) {
    const double price = latestPrices_.count(signal.symbol)
                         ? latestPrices_.at(signal.symbol)
                         : 0.0;

    if (price <= 0.0) {
        std::cerr << "[Portfolio] generateOrder: no price for "
                  << signal.symbol << " — order suppressed\n";
        return OrderEvent(signal.symbol, OrderType::HOLD, 0, 0.0);
    }

    if (signal.signalType == SignalType::LONG ||
        signal.signalType == SignalType::BUY) {
        const double equity = getTotalEquity();
        const int qty = std::max(1,
            static_cast<int>(std::floor((equity * riskFraction_) / price)));
        return OrderEvent(signal.symbol, OrderType::BUY, qty, price);
    }

    if (signal.signalType == SignalType::SHORT ||
        signal.signalType == SignalType::SELL) {
        const double equity = getTotalEquity();
        const int qty = std::max(1,
            static_cast<int>(std::floor((equity * riskFraction_) / price)));
        return OrderEvent(signal.symbol, OrderType::SELL, qty, price);
    }

    if (signal.signalType == SignalType::EXIT) {
        const int held = getPosition(signal.symbol);
        if (held <= 0) {
            // Nothing to close — return a HOLD so the engine skips execution
            return OrderEvent(signal.symbol, OrderType::HOLD, 0, price);
        }
        return OrderEvent(signal.symbol, OrderType::SELL, held, price);
    }

    return OrderEvent(signal.symbol, OrderType::HOLD, 0, price);
}

// ---------------------------------------------------------------------------
// Fill update — cash and position accounting after execution
// ---------------------------------------------------------------------------

void Portfolio::updateFill(const FillEvent& fill) {
    positions_[fill.symbol] += fill.quantity;
    cash_ -= fill.quantity * fill.price;
    cash_ -= fill.commission;

    const std::string ts = latestTimestamps_.count(fill.symbol)
                           ? latestTimestamps_.at(fill.symbol) : "";

    if (fill.quantity > 0) {
        lastBuyPrice_ = fill.price;
        trades_.push_back({ts, fill.price, fill.quantity, "BUY", true});
    } else {
        const bool profit = fill.price > lastBuyPrice_;
        trades_.push_back({ts, fill.price, -fill.quantity, "SELL", profit});
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

double Portfolio::getCash() const {
    return cash_;
}

int Portfolio::getPosition(const std::string& symbol) const {
    const auto it = positions_.find(symbol);
    return (it != positions_.end()) ? it->second : 0;
}

double Portfolio::getTotalEquity() const {
    double positionValue = 0.0;
    for (const auto& [symbol, qty] : positions_) {
        if (latestPrices_.count(symbol))
            positionValue += qty * latestPrices_.at(symbol);
    }
    return cash_ + positionValue;
}

const std::vector<EquityPoint>& Portfolio::getEquityCurve() const {
    return equityCurve_;
}

const std::vector<Portfolio::Trade>& Portfolio::getTrades() const {
    return trades_;
}

// ---------------------------------------------------------------------------
// CSV export
// ---------------------------------------------------------------------------

void Portfolio::exportEquityCurve(const std::string& filename) const {
    std::ofstream file(filename);
    file << "timestamp,equity,price,benchmark_equity\n";
    for (const auto& pt : equityCurve_)
        file << pt.timestamp << ","
             << pt.equity    << ","
             << pt.price     << ","
             << pt.benchmarkEquity << "\n";
}

void Portfolio::exportTrades(const std::string& filename) const {
    std::ofstream file(filename);
    file << "timestamp,price,quantity,direction,profit\n";
    for (const auto& t : trades_)
        file << t.timestamp  << ","
             << t.price      << ","
             << t.quantity   << ","
             << t.direction  << ","
             << t.profit     << "\n";
}
