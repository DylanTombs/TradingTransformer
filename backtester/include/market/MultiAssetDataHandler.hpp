#pragma once

#include "market/DataHandler.hpp"
#include "events/EventQueue.hpp"
#include "events/MarketEvent.hpp"

#include <memory>
#include <string>
#include <vector>

/**
 * MultiAssetDataHandler — synchronises multiple per-symbol DataHandlers.
 *
 * On each streamNext() call it emits ALL events whose timestamp equals the
 * earliest un-emitted timestamp across all handlers.  For daily data this
 * means all symbols on the same date are pushed to the queue before the
 * engine processes any of them — ensuring Portfolio::updateMarket is called
 * for every symbol before any strategy signal is acted upon.
 *
 * Timestamp comparison is lexicographic on ISO-8601 date strings
 * (YYYY-MM-DD), which orders correctly without date parsing.
 *
 * Usage:
 *   MultiAssetDataHandler multi;
 *   multi.addHandler(std::make_unique<FeatureCSVDataHandler>(...));
 *   multi.addHandler(std::make_unique<FeatureCSVDataHandler>(...));
 *   BacktestEngine engine(strategy, multi, config);
 */
class MultiAssetDataHandler : public DataHandler {
public:
    MultiAssetDataHandler() = default;

    /**
     * Add a data source.  Each handler is pre-fetched at construction so
     * the earliest timestamp is known before the first streamNext() call.
     * Must be called before BacktestEngine::run().
     */
    void addHandler(std::unique_ptr<DataHandler> handler);

    /**
     * Pushes all buffered events whose timestamp matches the global minimum
     * to the caller's queue, then pre-fetches the next event from each
     * exhausted-slot handler.
     *
     * Returns without pushing anything once all handlers are exhausted.
     */
    void streamNext(EventQueue& queue) override;

private:
    struct Channel {
        std::unique_ptr<DataHandler> handler;
        std::shared_ptr<Event>       buffered;   ///< next event (pre-fetched)
        bool                         done = false;

        /// Fetch the next event from the underlying handler into buffered.
        void advance() {
            EventQueue tmp;
            handler->streamNext(tmp);
            if (tmp.empty()) {
                done     = true;
                buffered = nullptr;
            } else {
                buffered = tmp.pop();
            }
        }

        const std::string& timestamp() const {
            return std::static_pointer_cast<MarketEvent>(buffered)->timestamp;
        }
    };

    std::vector<Channel> channels_;
};
