#include "../../include/market/MultiAssetDataHandler.hpp"

void MultiAssetDataHandler::addHandler(std::unique_ptr<DataHandler> handler) {
    Channel ch;
    ch.handler = std::move(handler);
    ch.advance();   // pre-fetch first event
    channels_.push_back(std::move(ch));
}

void MultiAssetDataHandler::streamNext(EventQueue& queue) {
    // ---- Find the earliest timestamp across all active channels ------------
    std::string earliest;
    for (const auto& ch : channels_) {
        if (ch.done) continue;
        if (earliest.empty() || ch.timestamp() < earliest)
            earliest = ch.timestamp();
    }
    if (earliest.empty()) return;   // all handlers exhausted

    // ---- Emit every buffered event at that timestamp -----------------------
    // This ensures Portfolio::updateMarket is called for ALL symbols on a
    // given date before any strategy signal is converted to an order.
    for (auto& ch : channels_) {
        if (ch.done) continue;
        if (ch.timestamp() == earliest) {
            queue.push(ch.buffered);
            ch.advance();
        }
    }
}
