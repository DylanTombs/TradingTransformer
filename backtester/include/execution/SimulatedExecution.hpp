#pragma once

#include "../events/OrderEvent.hpp"
#include "../events/FillEvent.hpp"

class SimulatedExecution {

private:

    double commission;

public:

    SimulatedExecution(double commission = 0.5);

    FillEvent executeOrder(const OrderEvent& order);
};