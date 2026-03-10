#include "../../include/market/CSVDataHandler.hpp"
#include "../../include/events/MarketEvent.hpp"
#include <sstream>

CSVDataHandler::CSVDataHandler(const std::string& filename)
    : file(filename) {}

void CSVDataHandler::streamNext(EventQueue& queue) {

    std::string line;

    if (!std::getline(file, line))
        return;

    std::stringstream ss(line);
    std::string ts, price;

    std::getline(ss, ts, ',');
    std::getline(ss, price, ',');

    long timestamp = std::stol(ts);
    double p = std::stod(price);

    queue.push(std::make_shared<MarketEvent>(p, timestamp));
}