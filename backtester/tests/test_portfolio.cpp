/**
 * Unit tests for Portfolio.
 */
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include "portfolio/Portfolio.hpp"
#include "events/MarketEvent.hpp"
#include "events/SignalEvent.hpp"
#include "events/FillEvent.hpp"

class PortfolioTest : public ::testing::Test {
protected:
    static constexpr double INITIAL_CASH = 10'000.0;
    Portfolio portfolio{INITIAL_CASH};
};


TEST_F(PortfolioTest, InitialCashMatchesConstructorArgument) {
    EXPECT_DOUBLE_EQ(portfolio.getCash(), INITIAL_CASH);
}

TEST_F(PortfolioTest, InitialPositionForUnknownSymbolIsZero) {
    EXPECT_EQ(portfolio.getPosition("AAPL"), 0);
}

TEST_F(PortfolioTest, InitialEquityCurveIsEmpty) {
    EXPECT_TRUE(portfolio.getEquityCurve().empty());
}

TEST_F(PortfolioTest, BuyFillDecreasesCorrectCashAmount) {
    FillEvent fill("AAPL", 5, 100.0, 0.5);
    portfolio.updateFill(fill);
    EXPECT_DOUBLE_EQ(portfolio.getCash(), 9'499.5);
}

TEST_F(PortfolioTest, BuyFillIncreasesPositionByQuantity) {
    FillEvent fill("AAPL", 5, 100.0, 0.5);
    portfolio.updateFill(fill);
    EXPECT_EQ(portfolio.getPosition("AAPL"), 5);
}

TEST_F(PortfolioTest, BuyFillRecordsTradeWithBuyDirection) {
    FillEvent fill("AAPL", 3, 100.0, 0.0);
    portfolio.updateFill(fill);
    ASSERT_EQ(portfolio.getTrades().size(), 1u);
    EXPECT_EQ(portfolio.getTrades()[0].direction, "BUY");
}

TEST_F(PortfolioTest, SellFillIncreasesPortfolioCash) {
    portfolio.updateFill(FillEvent("AAPL", 5, 100.0, 0.5));
    portfolio.updateFill(FillEvent("AAPL", -5, 110.0, 0.5));
    EXPECT_DOUBLE_EQ(portfolio.getCash(), 10'049.0);
}

TEST_F(PortfolioTest, SellFillReducesPosition) {
    portfolio.updateFill(FillEvent("AAPL", 5, 100.0, 0.0));
    portfolio.updateFill(FillEvent("AAPL", -5, 110.0, 0.0));
    EXPECT_EQ(portfolio.getPosition("AAPL"), 0);
}

TEST_F(PortfolioTest, ProfitableSellRecordedWithProfitTrue) {
    portfolio.lastBuyPrice = 100.0;
    FillEvent sell("AAPL", -5, 110.0, 0.0);
    portfolio.updateFill(sell);
    auto& trades = portfolio.getTrades();
    ASSERT_FALSE(trades.empty());
    EXPECT_TRUE(trades.back().profit);
}

TEST_F(PortfolioTest, LosingTradeRecordedWithProfitFalse) {
    portfolio.lastBuyPrice = 100.0;
    FillEvent sell("AAPL", -5, 90.0, 0.0); 
    portfolio.updateFill(sell);
    EXPECT_FALSE(portfolio.getTrades().back().profit);
}

TEST_F(PortfolioTest, CommissionDeductedSeparatelyFromTradeValue) {
    FillEvent fill("AAPL", 5, 100.0, 2.0);
    portfolio.updateFill(fill);
    EXPECT_DOUBLE_EQ(portfolio.getCash(), 9'498.0);
}

TEST_F(PortfolioTest, UpdateMarketAppendsOneEquityCurveEntry) {
    MarketEvent event("AAPL", 150.0, "2024-01-01");
    portfolio.updateMarket(event);
    EXPECT_EQ(portfolio.getEquityCurve().size(), 1u);
}

TEST_F(PortfolioTest, UpdateMarketEquityEqualsNetLiquidationValue) {
    MarketEvent event("AAPL", 150.0, "2024-01-01");
    portfolio.updateMarket(event);
    auto [ts, equity, price] = portfolio.getEquityCurve().back();
    EXPECT_DOUBLE_EQ(equity, INITIAL_CASH);
    EXPECT_DOUBLE_EQ(price, 150.0);
}

TEST_F(PortfolioTest, UpdateMarketReflectsOpenPositionValue) {
    portfolio.updateFill(FillEvent("AAPL", 10, 100.0, 0.0));
    MarketEvent event("AAPL", 120.0, "2024-01-02");
    portfolio.updateMarket(event);
    auto [ts, equity, price] = portfolio.getEquityCurve().back();
    EXPECT_DOUBLE_EQ(equity, 10'200.0);
}

TEST_F(PortfolioTest, LongSignalGeneratesBuyOrder) {
    portfolio.updateMarket(MarketEvent("AAPL", 150.0, "2024-01-01"));
    SignalEvent signal("AAPL", SignalType::LONG);
    OrderEvent order = portfolio.generateOrder(signal);
    EXPECT_EQ(order.orderType, OrderType::BUY);
    EXPECT_EQ(order.quantity, 10);          
    EXPECT_DOUBLE_EQ(order.price, 150.0);
}

TEST_F(PortfolioTest, ShortSignalGeneratesSellOrder) {
    portfolio.updateMarket(MarketEvent("AAPL", 150.0, "2024-01-01"));
    SignalEvent signal("AAPL", SignalType::SHORT);
    OrderEvent order = portfolio.generateOrder(signal);
    EXPECT_EQ(order.orderType, OrderType::SELL);
}

TEST_F(PortfolioTest, ExitSignalGeneratesZeroQuantityOrder) {
    portfolio.updateMarket(MarketEvent("AAPL", 150.0, "2024-01-01"));
    SignalEvent signal("AAPL", SignalType::EXIT);
    OrderEvent order = portfolio.generateOrder(signal);
    EXPECT_EQ(order.quantity, 0);
}

TEST_F(PortfolioTest, ExportEquityCurveWritesValidCsvWithHeader) {
    portfolio.updateMarket(MarketEvent("AAPL", 100.0, "2024-01-01"));
    portfolio.updateMarket(MarketEvent("AAPL", 110.0, "2024-01-02"));

    const std::string path = "/tmp/test_equity_curve.csv";
    portfolio.exportEquityCurve(path);

    std::ifstream file(path);
    ASSERT_TRUE(file.is_open()) << "CSV file was not created at " << path;

    std::string header;
    std::getline(file, header);
    EXPECT_EQ(header, "timestamp,equity,price");

    int line_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        EXPECT_FALSE(line.empty());
        ++line_count;
    }
    EXPECT_EQ(line_count, 2); 
}
