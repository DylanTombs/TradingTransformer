#pragma once

#include <unordered_map>
#include <string>
#include "../events/Events.hpp"
#include "../events/SignalEvent.hpp"
#include "../events/OrderEvent.hpp"
#include "../events/MarketEvent.hpp"
#include "../events/FillEvent.hpp"

class Portfolio {

private:

    double cash;
    std::unordered_map<std::string, int> positions;

public:

    Portfolio(double initialCash);

    void updateMarket(const MarketEvent& event);

    OrderEvent generateOrder(const SignalEvent& signal);

    double getCash() const;

    int getPosition(const std::string& symbol) const;

    void updateFill(const FillEvent& fill);
};