#include "../../include/events/MarketEvent.hpp"
#include "../../include/events/SignalEvent.hpp"
#include "../../include/events/OrderEvent.hpp"
#include "../../include/events/FillEvent.hpp"
#include "../../include/engine/BacktestEngine.hpp"

#include <iostream>

BacktestEngine::BacktestEngine(
    Strategy& strategy,
    DataHandler& dataHandler
)
    : strategy(strategy),
      dataHandler(dataHandler),
      portfolio(100000.0),     // starting capital
      riskManager(1000),
      execution(1.0) {}     // max position


void BacktestEngine::run() {

    while (true) {

        dataHandler.streamNext(queue);

        if (queue.empty())
            break;

        auto event = queue.pop();

        switch (event->getType()) {

        case EventType::MARKET: {

            auto marketEvent =
                std::static_pointer_cast<MarketEvent>(event);

            portfolio.updateMarket(*marketEvent);

            strategy.onMarketEvent(*marketEvent, queue);

            break;
        }

        case EventType::SIGNAL: {

            auto signal =
                std::static_pointer_cast<SignalEvent>(event);

            auto order = portfolio.generateOrder(*signal);

            queue.push(std::make_shared<OrderEvent>(order));

            break;
        }

        case EventType::ORDER: {

            auto order =
                std::static_pointer_cast<OrderEvent>(event);

            if (riskManager.approveOrder(*order)) {

                auto fill = execution.executeOrder(*order);

                queue.push(std::make_shared<FillEvent>(fill));
            }

            break;
        }

        case EventType::FILL: {

            auto fill =
                std::static_pointer_cast<FillEvent>(event);

            portfolio.updateFill(*fill);

            break;
        }

        }
    }
}

Portfolio& BacktestEngine::getPortfolio() {
    return portfolio;
}