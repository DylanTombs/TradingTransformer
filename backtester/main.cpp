#include "engine/BacktestEngine.hpp"
#include "market/CSVDataHandler.hpp"
#include "strategy/MovingAverageStrategy.hpp"

#include <iostream>

int main() {

    std::cout << "Starting backtest..." << std::endl;

    CSVDataHandler data("backtester/data/AAPL.csv");

    MovingAverageStrategy strategy(5);

    BacktestEngine engine(strategy, data);

    engine.run();

    std::cout << "Backtest complete." << std::endl;

    return 0;
}