#pragma once

#include "DataHandler.hpp"
#include <fstream>
#include <string>

class CSVDataHandler : public DataHandler {
private:
    std::ifstream file;

public:
    CSVDataHandler(const std::string& filename);

    void streamNext(EventQueue& queue) override;
};