#pragma once

#include "Events.hpp"
#include <string>

enum class SignalType {
    BUY,
    SELL,
    LONG,
    SHORT,
    EXIT
};

class SignalEvent : public Event {

public:

    std::string symbol;
    SignalType signalType;
    double strength;

    SignalEvent(const std::string& symbol,
                SignalType signalType,
                double strength = 1.0)
        : symbol(symbol),
          signalType(signalType),
          strength(strength) {}

    EventType getType() const override {
        return EventType::SIGNAL;
    }
};