/**
 * Unit tests for SimulatedExecution.
 *
 * Verifies fill-price arithmetic for BUY and SELL orders under various
 * cost parameter combinations.
 */
#include <cmath>
#include <gtest/gtest.h>

#include "execution/SimulatedExecution.hpp"
#include "events/OrderEvent.hpp"
#include "events/FillEvent.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static OrderEvent makeBuy(double price, int qty = 10,
                          const std::string& sym = "AAPL") {
    return OrderEvent(sym, OrderType::BUY, qty, price);
}

static OrderEvent makeSell(double price, int qty = 10,
                           const std::string& sym = "AAPL") {
    return OrderEvent(sym, OrderType::SELL, qty, price);
}

// ---------------------------------------------------------------------------
// BUY fill price
// ---------------------------------------------------------------------------

TEST(ExecutionBuy, ZeroCostFillPriceEqualsRaw) {
    SimulatedExecution exec(0.0, 0.0, 0.0, 0.0);
    auto fill = exec.executeOrder(makeBuy(100.0));
    EXPECT_DOUBLE_EQ(fill.price, 100.0);
}

TEST(ExecutionBuy, SpreadOnlyAddsHalfSpreadToRaw) {
    // fillPrice = raw * (1 + halfSpread)
    SimulatedExecution exec(0.0, 0.005, 0.0, 0.0);
    auto fill = exec.executeOrder(makeBuy(200.0));
    EXPECT_DOUBLE_EQ(fill.price, 200.0 * 1.005);
}

TEST(ExecutionBuy, SlippageAddedOnTopOfSpread) {
    // fillPrice = raw * (1 + halfSpread + slippage)
    SimulatedExecution exec(0.0, 0.002, 0.003, 0.0);
    auto fill = exec.executeOrder(makeBuy(100.0));
    EXPECT_DOUBLE_EQ(fill.price, 100.0 * (1.0 + 0.002 + 0.003));
}

TEST(ExecutionBuy, MarketImpactScalesWithQuantity) {
    // fillPrice = raw * (1 + spread + slip) + impact * qty
    SimulatedExecution exec(0.0, 0.0, 0.0, 0.01);
    auto fill = exec.executeOrder(makeBuy(50.0, 20));
    EXPECT_DOUBLE_EQ(fill.price, 50.0 + 0.01 * 20);
}

TEST(ExecutionBuy, AllCostComponentsCombined) {
    SimulatedExecution exec(1.5, 0.001, 0.002, 0.005);
    auto fill = exec.executeOrder(makeBuy(100.0, 10));
    const double expected = 100.0 * (1.0 + 0.001 + 0.002) + 0.005 * 10;
    EXPECT_DOUBLE_EQ(fill.price, expected);
}

TEST(ExecutionBuy, CommissionPassedThroughToFill) {
    SimulatedExecution exec(2.5, 0.0, 0.0, 0.0);
    auto fill = exec.executeOrder(makeBuy(100.0));
    EXPECT_DOUBLE_EQ(fill.commission, 2.5);
}

TEST(ExecutionBuy, QuantityIsPositiveInFill) {
    SimulatedExecution exec(0.0, 0.0, 0.0, 0.0);
    auto fill = exec.executeOrder(makeBuy(100.0, 7));
    EXPECT_EQ(fill.quantity, 7);
}

TEST(ExecutionBuy, SymbolPropagatedToFill) {
    SimulatedExecution exec(0.0, 0.0, 0.0, 0.0);
    auto fill = exec.executeOrder(makeBuy(100.0, 1, "MSFT"));
    EXPECT_EQ(fill.symbol, "MSFT");
}

// ---------------------------------------------------------------------------
// SELL fill price
// ---------------------------------------------------------------------------

TEST(ExecutionSell, ZeroCostFillPriceEqualsRaw) {
    SimulatedExecution exec(0.0, 0.0, 0.0, 0.0);
    auto fill = exec.executeOrder(makeSell(100.0));
    EXPECT_DOUBLE_EQ(fill.price, 100.0);
}

TEST(ExecutionSell, SpreadReducesFillBelowRaw) {
    // fillPrice = raw * (1 - halfSpread)
    SimulatedExecution exec(0.0, 0.005, 0.0, 0.0);
    auto fill = exec.executeOrder(makeSell(200.0));
    EXPECT_DOUBLE_EQ(fill.price, 200.0 * (1.0 - 0.005));
}

TEST(ExecutionSell, SlippageAndSpreadBothSubtracted) {
    SimulatedExecution exec(0.0, 0.002, 0.003, 0.0);
    auto fill = exec.executeOrder(makeSell(100.0));
    EXPECT_DOUBLE_EQ(fill.price, 100.0 * (1.0 - 0.002 - 0.003));
}

TEST(ExecutionSell, MarketImpactSubtracted) {
    SimulatedExecution exec(0.0, 0.0, 0.0, 0.01);
    auto fill = exec.executeOrder(makeSell(50.0, 20));
    EXPECT_DOUBLE_EQ(fill.price, 50.0 - 0.01 * 20);
}

TEST(ExecutionSell, QuantityIsNegatedInFill) {
    SimulatedExecution exec(0.0, 0.0, 0.0, 0.0);
    auto fill = exec.executeOrder(makeSell(100.0, 5));
    EXPECT_EQ(fill.quantity, -5);
}

TEST(ExecutionSell, CommissionPassedThroughToFill) {
    SimulatedExecution exec(3.0, 0.0, 0.0, 0.0);
    auto fill = exec.executeOrder(makeSell(100.0));
    EXPECT_DOUBLE_EQ(fill.commission, 3.0);
}

// ---------------------------------------------------------------------------
// Friction cost symmetry
// ---------------------------------------------------------------------------

TEST(ExecutionFriction, BuyCostIsAlwaysNonNegative) {
    SimulatedExecution exec(0.0, 0.001, 0.001, 0.002);
    auto fill = exec.executeOrder(makeBuy(100.0, 10));
    EXPECT_GE(fill.price, 100.0);
}

TEST(ExecutionFriction, SellCostNeverExceedsRawPrice) {
    SimulatedExecution exec(0.0, 0.001, 0.001, 0.002);
    auto fill = exec.executeOrder(makeSell(100.0, 10));
    EXPECT_LE(fill.price, 100.0);
}
