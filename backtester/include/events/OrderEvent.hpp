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
    double price;

    OrderEvent(const std::string& symbol,
               OrderType type,
               int quantity, double price)
        : symbol(symbol),
          orderType(type),
          quantity(quantity),
          price(price) {}

    EventType getType() const override {
        return EventType::ORDER;
    }
};