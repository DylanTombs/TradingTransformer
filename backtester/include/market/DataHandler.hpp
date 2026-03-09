#pragma once
#include "../events/EventQueue.hpp"

class DataHandler {
public:
    virtual void streamNext(EventQueue& queue) = 0;
};