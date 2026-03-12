#pragma once

#include "../events/OrderEvent.hpp"

class RiskManager {

private:

    int maxPositionSize;

public:

    RiskManager(int maxPositionSize);

    bool approveOrder(const OrderEvent& order);
};