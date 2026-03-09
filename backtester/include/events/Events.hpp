#pragma once

enum class EventType {
    MARKET_DATA,
    SIGNAL,
    ORDER,
    FILL
};

class Event {
public:
    virtual EventType getType() const = 0;
    virtual ~Event() = default;
};