#include "../../include/strategy/MovingAverageStrategy.hpp"
#include "../../include/events/SignalEvent.hpp"

MovingAverageStrategy::MovingAverageStrategy(int window)
    : window(window) {}

void MovingAverageStrategy::onMarketEvent(
        const MarketEvent& event,
        EventQueue& queue) {

    prices.push_back(event.price);

    if (prices.size() > window)
        prices.pop_front();

    if (prices.size() < window)
        return;

    double avg = 0;
    for (double p : prices)
        avg += p;
    avg /= window;

    SignalType newSignal = (event.price > avg) ? SignalType::LONG : SignalType::SHORT;

    // Only emit signal when it changes
    if (newSignal != lastSignal) {
        lastSignal = newSignal;
        queue.push(std::make_shared<SignalEvent>(event.symbol, newSignal));
    }
}