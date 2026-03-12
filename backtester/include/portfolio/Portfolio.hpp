#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include "../events/Events.hpp"
#include "../events/SignalEvent.hpp"
#include "../events/OrderEvent.hpp"
#include "../events/MarketEvent.hpp"
#include "../events/FillEvent.hpp"

class Portfolio {

private:

    double cash;
    std::vector<double> equityCurve;
    std::vector<double> priceHistory;
    std::unordered_map<std::string, int> positions;

public:

    Portfolio(double initialCash);

    void updateMarket(const MarketEvent& event);

    OrderEvent generateOrder(const SignalEvent& signal);

    double getCash() const;

    int getPosition(const std::string& symbol) const;

    const std::vector<double>& getEquityCurve() const;

    const std::vector<double>& getPriceHistory() const;

    void updateFill(const FillEvent& fill);

    void exportEquityCurve(const std::string& filename);

};