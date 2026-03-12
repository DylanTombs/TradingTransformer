#include "../../include/portfolio/Portfolio.hpp"
#include <fstream>

Portfolio::Portfolio(double initialCash)
    : cash(initialCash) {}

void Portfolio::updateMarket(const MarketEvent& event) {

    double positionValue = 0;

    if (positions.count(event.symbol))
        positionValue = positions[event.symbol] * event.price;

    double equity = cash + positionValue;

    equityCurve.push_back(equity);
}

OrderEvent Portfolio::generateOrder(const SignalEvent& signal) {

    int quantity = 100;

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

const std::vector<double>& Portfolio::getEquityCurve() const {
    return equityCurve;
}

const std::vector<double>& Portfolio::getPriceHistory() const {
    return priceHistory;
}

void Portfolio::updateFill(const FillEvent& fill) {

    positions[fill.symbol] += fill.quantity;

    cash -= fill.quantity * fill.price;

    cash -= fill.commission;
}

void Portfolio::exportEquityCurve(const std::string& filename) {

    std::ofstream file(filename);

    for (double e : equityCurve)
        file << e << "\n";
}