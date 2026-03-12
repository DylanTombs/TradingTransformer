#include "../../include/execution/SimulatedExecution.hpp"

SimulatedExecution::SimulatedExecution(double commission)
    : commission(commission) {}

FillEvent SimulatedExecution::executeOrder(const OrderEvent& order) {

    double fillPrice = 100.0; // placeholder price for now

    int quantity = order.quantity;

    if (order.orderType == OrderType::SELL)
        quantity = -quantity;

    return FillEvent(
        order.symbol,
        quantity,
        fillPrice,
        commission
    );
}