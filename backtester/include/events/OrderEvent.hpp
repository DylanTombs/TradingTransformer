#pragma once

#include "Events.hpp"
#include <string>

enum class OrderType {
    BUY,
    SELL,
    HOLD
};

class OrderEvent : public Event {

public:

    std::string symbol;
    OrderType orderType;
    int quantity;

    OrderEvent(const std::string& symbol,
               OrderType type,
               int quantity)
        : symbol(symbol),
          orderType(type),
          quantity(quantity) {}

    EventType getType() const override {
        return EventType::ORDER;
    }
};