#include "../../include/market/CSVDataHandler.hpp"
#include "../../include/events/MarketEvent.hpp"
#include <sstream>
#include <iostream>

CSVDataHandler::CSVDataHandler(const std::string& filename)
    : file(filename) {
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file: " << filename << std::endl;
    } else {
        std::cout << "Opened file: " << filename << std::endl;
    }
    std::string header;
    std::getline(file, header); 
}

void CSVDataHandler::streamNext(EventQueue& queue) {
    std::string line;
    if (!std::getline(file, line))
        return;

    std::stringstream ss(line);
    std::string ts, open, high, low, close;

    std::getline(ss, ts, ',');
    std::getline(ss, open, ',');
    std::getline(ss, high, ',');
    std::getline(ss, low, ',');
    std::getline(ss, close, ',');

    double p = std::stod(close);

    queue.push(std::make_shared<MarketEvent>("AAPL", p, ts));
}