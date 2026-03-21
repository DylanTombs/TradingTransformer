#pragma once

#include "market/DataHandler.hpp"
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * Reads a feature-rich CSV produced by research/features/pipeline.py and
 * emits FeatureMarketEvent objects.
 *
 * The CSV must have a header row.  Required columns:
 *   - A date column (ISO 8601 "YYYY-MM-DD")
 *   - One column per feature in the order expected by the model
 *     (auxilFeatures followed by the target column)
 *
 * Column ordering is resolved at construction time by matching the
 * user-supplied featureColumns list against the CSV header, so the
 * CSV columns do not need to be pre-sorted.
 *
 * Example construction:
 *   FeatureCSVDataHandler handler(
 *       "data/AAPL_features.csv", "AAPL",
 *       {"high","low","volume",...,"close"},   // model feature order
 *       "close",   // price column for MarketEvent.price
 *       "date"     // timestamp column
 *   );
 */
class FeatureCSVDataHandler : public DataHandler {
public:
    FeatureCSVDataHandler(const std::string&              filename,
                          const std::string&              symbol,
                          const std::vector<std::string>& featureColumns,
                          const std::string&              closeColumn = "close",
                          const std::string&              dateColumn  = "date");

    void streamNext(EventQueue& queue) override;

private:
    std::ifstream file_;
    std::string   symbol_;

    // Indices into a parsed CSV row for each model feature, the close
    // price, and the date — resolved from the header at construction.
    std::vector<int> featureColIndices_;
    int              closeColIndex_ = -1;
    int              dateColIndex_  = -1;

    std::vector<std::string> splitCSVRow(const std::string& line) const;

    // Returns [month (1-12), day (1-31), weekday (0=Mon … 6=Sun)]
    // matching Python's datetime.weekday() convention.
    static std::vector<double> parseTimeMark(const std::string& timestamp);
};
