#pragma once
#include "./Events.hpp"
#include <string>

class MarketEvent : public Event {
public:
    std::string symbol;
    double price;
    std::string timestamp; 

    MarketEvent(const std::string& symbol, double price, const std::string& timestamp)
        : symbol(symbol), price(price), timestamp(timestamp) {}

    EventType getType() const override {
        return EventType::MARKET;
    }
};