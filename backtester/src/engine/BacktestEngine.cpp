#include "../../include/engine/BacktestEngine.hpp"
#include "../../include/events/MarketEvent.hpp"

BacktestEngine::BacktestEngine(Strategy& strategy, DataHandler& dataHandler)
    : strategy(strategy), dataHandler(dataHandler) {}

void BacktestEngine::run() {

    while (true) {

        dataHandler.streamNext(queue);

        if (queue.empty())
            break;

        auto event = queue.pop();

        if (event->getType() == EventType::MARKET_DATA) {

            auto market_event =
                std::static_pointer_cast<MarketEvent>(event);

            strategy.onMarketEvent(*market_event, queue);
        }
    }
}