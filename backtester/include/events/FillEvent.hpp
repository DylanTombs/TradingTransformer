#pragma once

#include "Events.hpp"
#include <string>

class FillEvent : public Event {

public:

    std::string symbol;
    int quantity;
    double price;
    double commission;

    FillEvent(const std::string& symbol,
              int quantity,
              double price,
              double commission)
        : symbol(symbol),
          quantity(quantity),
          price(price),
          commission(commission) {}

    EventType getType() const override {
        return EventType::FILL;
    }
};