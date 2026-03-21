/**
 * Integration tests for BacktestEngine.
 */
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "config/BacktestConfig.hpp"
#include "engine/BacktestEngine.hpp"
#include "market/DataHandler.hpp"
#include "strategy/Strategy.hpp"
#include "events/MarketEvent.hpp"
#include "events/SignalEvent.hpp"


class MockDataHandler : public DataHandler {
public:
    explicit MockDataHandler(std::string symbol, std::vector<double> prices)
        : symbol_(std::move(symbol)), prices_(std::move(prices)), current_(0) {}

    void streamNext(EventQueue& queue) override {
        if (current_ >= static_cast<int>(prices_.size())) return;
        queue.push(std::make_shared<MarketEvent>(
            symbol_, prices_[current_],
            "2024-01-0" + std::to_string(current_ + 1)));
        ++current_;
    }

private:
    std::string          symbol_;
    std::vector<double>  prices_;
    int                  current_;
};

class AlwaysBuyStrategy : public Strategy {
public:
    int market_events_received = 0;

    void onMarketEvent(const MarketEvent& event, EventQueue& queue) override {
        ++market_events_received;
        if (market_events_received == 1)
            queue.push(std::make_shared<SignalEvent>(event.symbol, SignalType::LONG));
    }
};

class BuyThenExitStrategy : public Strategy {
public:
    void onMarketEvent(const MarketEvent& event, EventQueue& queue) override {
        ++bar_;
        if (bar_ == 1)
            queue.push(std::make_shared<SignalEvent>(event.symbol, SignalType::LONG));
        if (bar_ == 2)
            queue.push(std::make_shared<SignalEvent>(event.symbol, SignalType::EXIT));
    }
private:
    int bar_ = 0;
};

class NeverSignalStrategy : public Strategy {
public:
    void onMarketEvent(const MarketEvent&, EventQueue&) override {}
};


// ---- Zero-friction config for deterministic quantity assertions -------------
static BacktestConfig zeroFriction() {
    BacktestConfig cfg;
    cfg.halfSpread       = 0.0;
    cfg.slippageFraction = 0.0;
    cfg.marketImpact     = 0.0;
    cfg.commission       = 0.0;
    return cfg;
}


TEST(BacktestEngineTest, EmptyDataStreamProducesNoTrades) {
    NeverSignalStrategy strategy;
    MockDataHandler     data("AAPL", {});
    BacktestEngine      engine(strategy, data);
    engine.run();
    EXPECT_TRUE(engine.getPortfolio().getTrades().empty());
}

TEST(BacktestEngineTest, MarketEventsPopulateEquityCurve) {
    NeverSignalStrategy strategy;
    MockDataHandler     data("AAPL", {100.0, 102.0, 104.0});
    BacktestEngine      engine(strategy, data);
    engine.run();
    EXPECT_EQ(engine.getPortfolio().getEquityCurve().size(), 3u);
}

TEST(BacktestEngineTest, LongSignalResultsInOpenPosition) {
    AlwaysBuyStrategy strategy;
    MockDataHandler   data("AAPL", {100.0, 102.0, 104.0});
    BacktestEngine    engine(strategy, data, zeroFriction());
    engine.run();
    EXPECT_GT(engine.getPortfolio().getPosition("AAPL"), 0)
        << "Expected a non-zero AAPL position after a LONG signal";
}

TEST(BacktestEngineTest, FullBuyLoopDecreasesAvailableCash) {
    AlwaysBuyStrategy strategy;
    MockDataHandler   data("AAPL", {100.0, 102.0});
    BacktestEngine    engine(strategy, data, zeroFriction());

    const double cash_before = engine.getPortfolio().getCash();
    engine.run();

    EXPECT_LT(engine.getPortfolio().getCash(), cash_before)
        << "Cash should decrease after executing a BUY order";
}

TEST(BacktestEngineTest, LongSignalQuantityMatchesRiskBasedSizing) {
    // equity=$100k, riskFraction=10%, price=$100, zero friction
    // expected qty = floor(100000 * 0.10 / 100) = 100
    AlwaysBuyStrategy strategy;
    MockDataHandler   data("AAPL", {100.0});
    BacktestEngine    engine(strategy, data, zeroFriction());
    engine.run();
    EXPECT_EQ(engine.getPortfolio().getPosition("AAPL"), 100);
}

TEST(BacktestEngineTest, ExitSignalFullyClosesPosition) {
    // Bar 1: LONG → buy some shares.  Bar 2: EXIT → sell all shares.
    BuyThenExitStrategy strategy;
    MockDataHandler     data("AAPL", {100.0, 105.0});
    BacktestEngine      engine(strategy, data, zeroFriction());
    engine.run();
    EXPECT_EQ(engine.getPortfolio().getPosition("AAPL"), 0)
        << "EXIT signal should fully close the open position";
}

TEST(BacktestEngineTest, StrategyReceivesAllMarketEvents) {
    AlwaysBuyStrategy strategy;
    const int n_bars = 5;
    MockDataHandler data("AAPL", std::vector<double>(n_bars, 100.0));
    BacktestEngine  engine(strategy, data);
    engine.run();
    EXPECT_EQ(strategy.market_events_received, n_bars);
}

TEST(BacktestEngineTest, EquityCurveIncludesBenchmarkColumn) {
    NeverSignalStrategy strategy;
    MockDataHandler     data("AAPL", {100.0, 120.0});
    BacktestEngine      engine(strategy, data);
    engine.run();

    const auto& curve = engine.getPortfolio().getEquityCurve();
    ASSERT_EQ(curve.size(), 2u);
    // Benchmark at t=0: initialCash * (100/100) = initialCash
    EXPECT_DOUBLE_EQ(curve[0].benchmarkEquity, curve[0].equity);
    // Benchmark at t=1: initialCash * (120/100) = 1.2x
    EXPECT_DOUBLE_EQ(curve[1].benchmarkEquity, curve[0].equity * 1.2);
}
