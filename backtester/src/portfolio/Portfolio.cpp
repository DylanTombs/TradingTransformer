#include "../../include/portfolio/Portfolio.hpp"
#include <fstream>


Portfolio::Portfolio(double initialCash)
    : cash(initialCash) {}

void Portfolio::updateMarket(const MarketEvent& event) {
    latestPrices[event.symbol] = event.price;
    latestTimestamps[event.symbol] = event.timestamp;

    double positionValue = 0;
    for (const auto& [symbol, qty] : positions) {
        if (latestPrices.count(symbol))
            positionValue += qty * latestPrices[symbol];
    }

    equityCurve.push_back({event.timestamp, cash + positionValue, event.price});
}

OrderEvent Portfolio::generateOrder(const SignalEvent& signal) {

    int quantity = 10;

    if (signal.signalType == SignalType::LONG) {
        return OrderEvent(signal.symbol, OrderType::BUY, quantity);
    }

    if (signal.signalType == SignalType::SHORT) {
        return OrderEvent(signal.symbol, OrderType::SELL, quantity);
    }

    // EXIT or unknown → no order
    return OrderEvent(signal.symbol, OrderType::BUY, 0);
}

double Portfolio::getCash() const {
    return cash;
}

int Portfolio::getPosition(const std::string& symbol) const {

    auto it = positions.find(symbol);

    if (it != positions.end())
        return it->second;

    return 0;
}

const std::vector<std::tuple<std::string, double, double>>& Portfolio::getEquityCurve() const {
    return equityCurve;
}

void Portfolio::updateFill(const FillEvent& fill) {
    positions[fill.symbol] += fill.quantity;
    cash -= fill.quantity * fill.price;
    cash -= fill.commission;

    std::string ts = latestTimestamps.count(fill.symbol) 
                     ? latestTimestamps[fill.symbol] : "";

    if (fill.quantity > 0) { 
        lastBuyPrice = fill.price;
        trades.push_back({ts, fill.price, "BUY", true});
    } else { 
        bool profit = fill.price > lastBuyPrice;
        trades.push_back({ts, fill.price, "SELL", profit});
    }
}

void Portfolio::exportEquityCurve(const std::string& filename) {
    std::ofstream file(filename);
    file << "timestamp,equity,price\n";
    for (const auto& [ts, equity, price] : equityCurve)
        file << ts << "," << equity << "," << price << "\n";
}

void Portfolio::exportTrades(const std::string& filename) {
    std::ofstream file(filename);
    file << "timestamp,price,direction,profit\n";
    for (const auto& t : trades)
        file << t.timestamp << "," << t.price << "," 
             << t.direction << "," << t.profit << "\n";
}

const std::vector<Portfolio::Trade>& Portfolio::getTrades() const {
    return trades;
}