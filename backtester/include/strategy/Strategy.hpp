#pragma once
#include "events/MarketEvent.hpp"
#include "events/EventQueue.hpp"

class Strategy {
public:
    virtual void onMarketEvent(const MarketEvent& event, EventQueue& queue) = 0;
};