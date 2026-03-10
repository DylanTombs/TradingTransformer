#pragma once

#include "../events/EventQueue.hpp"
#include "../strategy/Strategy.hpp"
#include "../market/DataHandler.hpp"

class BacktestEngine {
private:
    EventQueue queue;
    Strategy& strategy;
    DataHandler& dataHandler;

public:
    BacktestEngine(Strategy& strategy, DataHandler& dataHandler);

    void run();
};