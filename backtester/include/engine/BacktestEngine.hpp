#pragma once

#include "../config/BacktestConfig.hpp"
#include "../events/EventQueue.hpp"
#include "../strategy/Strategy.hpp"
#include "../market/DataHandler.hpp"
#include "../portfolio/Portfolio.hpp"
#include "../portfolio/RiskManager.hpp"
#include "../execution/SimulatedExecution.hpp"

class BacktestEngine {

private:

    EventQueue queue;

    Strategy&    strategy;
    DataHandler& dataHandler;

    Portfolio          portfolio;
    RiskManager        riskManager;
    SimulatedExecution execution;

public:

    /**
     * @param strategy    Concrete strategy — must outlive the engine
     * @param dataHandler Data source — must outlive the engine
     * @param config      Runtime parameters (slippage, sizing, capital).
     *                    Defaults produce a zero-friction, 10%-risk simulation.
     */
    BacktestEngine(Strategy&           strategy,
                   DataHandler&        dataHandler,
                   const BacktestConfig& config = BacktestConfig{});

    void run();
    Portfolio& getPortfolio();
};
