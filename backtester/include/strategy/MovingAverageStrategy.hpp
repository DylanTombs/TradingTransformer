#pragma once

#include "strategy/Strategy.hpp"
#include "../../include/events/SignalEvent.hpp"
#include <deque>

class MovingAverageStrategy : public Strategy {
private:
    std::deque<double> prices;
    int window;
    SignalType lastSignal = SignalType::EXIT;

public:
    MovingAverageStrategy(int window);

    void onMarketEvent(const MarketEvent& event, EventQueue& queue) override;
};