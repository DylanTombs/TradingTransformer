#pragma once
#include <queue>
#include <memory>
#include "./Events.hpp"

class EventQueue {
private:
    std::queue<std::shared_ptr<Event>> queue;

public:
    void push(std::shared_ptr<Event> event) {
        queue.push(event);
    }

    std::shared_ptr<Event> pop() {
        auto event = queue.front();
        queue.pop();
        return event;
    }

    bool empty() const {
        return queue.empty();
    }
};