/**
 * Integration tests for BacktestEngine.
 */
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

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
    std::string symbol_;
    std::vector<double> prices_;
    int current_;
};

class AlwaysBuyStrategy : public Strategy {
public:
    int market_events_received = 0;

    void onMarketEvent(const MarketEvent& event, EventQueue& queue) override {
        ++market_events_received;
        if (market_events_received == 1) {
            queue.push(std::make_shared<SignalEvent>(event.symbol, SignalType::LONG));
        }
    }
};

class NeverSignalStrategy : public Strategy {
public:
    void onMarketEvent(const MarketEvent&, EventQueue&) override {}
};


TEST(BacktestEngineTest, EmptyDataStreamProducesNoTrades) {
    NeverSignalStrategy strategy;
    MockDataHandler data("AAPL", {});  
    BacktestEngine engine(strategy, data);
    engine.run();
    EXPECT_TRUE(engine.getPortfolio().getTrades().empty());
}

TEST(BacktestEngineTest, MarketEventsPopulateEquityCurve) {
    NeverSignalStrategy strategy;
    MockDataHandler data("AAPL", {100.0, 102.0, 104.0});
    BacktestEngine engine(strategy, data);
    engine.run();
    EXPECT_EQ(engine.getPortfolio().getEquityCurve().size(), 3u);
}

TEST(BacktestEngineTest, LongSignalResultsInOpenPosition) {
    AlwaysBuyStrategy strategy;
    MockDataHandler data("AAPL", {100.0, 102.0, 104.0});
    BacktestEngine engine(strategy, data);
    engine.run();
    EXPECT_GT(engine.getPortfolio().getPosition("AAPL"), 0)
        << "Expected a non-zero AAPL position after a LONG signal";
}

TEST(BacktestEngineTest, FullBuyLoopDecreasesAvailableCash) {
    AlwaysBuyStrategy strategy;
    MockDataHandler data("AAPL", {100.0, 102.0});
    BacktestEngine engine(strategy, data);

    const double cash_before = engine.getPortfolio().getCash();
    engine.run();
    const double cash_after = engine.getPortfolio().getCash();

    EXPECT_LT(cash_after, cash_before)
        << "Cash should decrease after executing a BUY order";
}

TEST(BacktestEngineTest, RiskManagerRejectsOversizedOrders) {
    AlwaysBuyStrategy strategy;
    MockDataHandler data("AAPL", {100.0});
    BacktestEngine engine(strategy, data);
    engine.run();
    EXPECT_EQ(engine.getPortfolio().getPosition("AAPL"), 10);
}

TEST(BacktestEngineTest, StrategyReceivesAllMarketEvents) {
    AlwaysBuyStrategy strategy;
    const int n_bars = 5;
    MockDataHandler data("AAPL", std::vector<double>(n_bars, 100.0));
    BacktestEngine engine(strategy, data);
    engine.run();
    EXPECT_EQ(strategy.market_events_received, n_bars);
}
