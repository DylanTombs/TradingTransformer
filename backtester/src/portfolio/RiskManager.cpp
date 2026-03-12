#include "../../include/portfolio/RiskManager.hpp"

RiskManager::RiskManager(int maxPositionSize)
    : maxPositionSize(maxPositionSize) {}

bool RiskManager::approveOrder(const OrderEvent& order) {

    if (order.quantity > maxPositionSize)
        return false;

    return true;
}