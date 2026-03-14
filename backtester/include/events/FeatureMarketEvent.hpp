#pragma once

#include "events/MarketEvent.hpp"
#include <vector>

/**
 * Extends MarketEvent with a pre-computed feature vector for ML strategies.
 *
 * BacktestEngine uses static_pointer_cast<MarketEvent> — valid because this
 * class inherits from MarketEvent.  MLStrategy receives a const MarketEvent&
 * and uses dynamic_cast to recover the full feature payload.
 *
 * Feature ordering must match the training dataset exactly:
 *   auxilFeatures[0..N-1]  then  target ('close')
 * i.e. the same order as DataFrameDataset.colsData.
 */
class FeatureMarketEvent : public MarketEvent {
public:
    // Scaled feature values in model column order (length == encIn).
    // Populated by FeatureCSVDataHandler; scaled by MLStrategy.
    std::vector<double> features;

    // Time encoding matching DataFrameDataset._processTimeFeatures():
    //   [month (1-12), day (1-31), weekday (0=Mon … 6=Sun)]
    std::vector<double> timeMark;

    FeatureMarketEvent(const std::string& symbol,
                       double             price,
                       const std::string& timestamp,
                       std::vector<double> features,
                       std::vector<double> timeMark)
        : MarketEvent(symbol, price, timestamp)
        , features(std::move(features))
        , timeMark(std::move(timeMark)) {}
};
