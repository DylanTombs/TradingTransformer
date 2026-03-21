#pragma once

#include "../events/OrderEvent.hpp"
#include "../events/FillEvent.hpp"

/**
 * SimulatedExecution — realistic fill simulation.
 *
 * Each order is filled at a price adjusted for:
 *   1. Bid-ask spread:  BUY fills above mid, SELL fills below mid
 *   2. Slippage:        directional price drift from order submission latency
 *   3. Market impact:   optional per-share price impact for large orders
 *
 * Fill price model:
 *   BUY  fillPrice = orderPrice * (1 + halfSpread + slippage) + impact * qty
 *   SELL fillPrice = orderPrice * (1 - halfSpread - slippage) - impact * qty
 *
 * Each fill is logged to stdout with the raw price, fill price, and total
 * cost of transaction friction (slippage + spread).
 */
class SimulatedExecution {
public:
    /**
     * @param commission       Fixed commission per fill ($). Default: $1.00
     * @param halfSpread       Half the bid-ask spread fraction (e.g. 0.0005 = 0.05%)
     * @param slippageFraction Directional slippage fraction (e.g. 0.0005 = 0.05%)
     * @param marketImpact     Additional price movement per share (e.g. 0.001 = $0.001/share)
     */
    explicit SimulatedExecution(double commission       = 1.0,
                                double halfSpread       = 0.0,
                                double slippageFraction = 0.0,
                                double marketImpact     = 0.0);

    FillEvent executeOrder(const OrderEvent& order);

private:
    double commission_;
    double halfSpread_;
    double slippageFraction_;
    double marketImpact_;
};
