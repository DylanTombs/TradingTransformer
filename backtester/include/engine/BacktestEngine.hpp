#pragma once

#include "../events/EventQueue.hpp"
#include "../strategy/Strategy.hpp"
#include "../market/DataHandler.hpp"
#include "../portfolio/Portfolio.hpp"
#include "../portfolio/RiskManager.hpp"
#include "../execution/SimulatedExecution.hpp"

class BacktestEngine {

private:

    EventQueue queue;

    Strategy& strategy;
    DataHandler& dataHandler;

    Portfolio portfolio;
    RiskManager riskManager;
    SimulatedExecution execution;

public:

    BacktestEngine(
        Strategy& strategy,
        DataHandler& dataHandler
    );

    void run();
};