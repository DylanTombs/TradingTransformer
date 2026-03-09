#pragma once
#include "./Events.hpp"

enum class SignalType {
    BUY,
    SELL
};

class SignalEvent : public Event {
public:
    SignalType signal;
    double price;

    SignalEvent(SignalType signal, double price)
        : signal(signal), price(price) {}

    EventType getType() const override {
        return EventType::SIGNAL;
    }
};