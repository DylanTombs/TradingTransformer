#include "../../include/portfolio/Portfolio.hpp"

Portfolio::Portfolio(double initialCash)
    : cash(initialCash) {}

void Portfolio::updateMarket(const MarketEvent& event) {
    // Future: mark-to-market portfolio value
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

void Portfolio::updateFill(const FillEvent& fill) {

    positions[fill.symbol] += fill.quantity;

    cash -= fill.quantity * fill.price;

    cash -= fill.commission;
}