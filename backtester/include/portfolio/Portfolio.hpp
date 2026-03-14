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
    std::vector<std::tuple<std::string, double, double>> equityCurve; 
    std::unordered_map<std::string, double> latestPrices;
    std::unordered_map<std::string, int> positions;
    std::unordered_map<std::string, std::string> latestTimestamps;

public:

    Portfolio(double initialCash);

    struct Trade {
      std::string timestamp;
      double price;
      std::string direction; 
      bool profit;           
    };  

    std::vector<Trade> trades;

    double lastBuyPrice = 0.0;

    void updateMarket(const MarketEvent& event);

    OrderEvent generateOrder(const SignalEvent& signal);

    double getCash() const;

    int getPosition(const std::string& symbol) const;

    const std::vector<std::tuple<std::string, double, double>>& getEquityCurve() const;

    void updateFill(const FillEvent& fill);

    void exportEquityCurve(const std::string& filename);

    void exportTrades(const std::string& filename);

    const std::vector<Trade>& getTrades() const;


};