#pragma once

#include "strategy/Strategy.hpp"
#include <deque>

class MovingAverageStrategy : public Strategy {
private:
    std::deque<double> prices;
    int window;

public:
    MovingAverageStrategy(int window);

    void onMarketEvent(const MarketEvent& event, EventQueue& queue) override;
};