#pragma once
#include "./Events.hpp"

class MarketEvent : public Event {
public:
    double price;
    long timestamp;

    MarketEvent(double price, long timestamp)
        : price(price), timestamp(timestamp) {}

    EventType getType() const override {
        return EventType::MARKET_DATA;
    }
};