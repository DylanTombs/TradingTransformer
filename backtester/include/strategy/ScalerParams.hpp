#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Holds the mean and scale (std-dev) arrays produced by sklearn's
 * StandardScaler.  Mirrors the Python call:
 *
 *   x_scaled   = (x - mean_) / scale_
 *   x_original = x_scaled * scale_ + mean_
 *
 * Parameters are loaded from a CSV written by research/exportModel.py:
 *
 *   feature,mean,scale
 *   high,102.15,25.30
 *   low,98.05,24.50
 *   ...
 */
struct ScalerParams {
    std::vector<double> mean;
    std::vector<double> scale;

    bool empty() const { return mean.empty(); }

    // ------------------------------------------------------------------
    // Factory
    // ------------------------------------------------------------------

    /**
     * Load scaler parameters from a two-column CSV.
     * Skips the header row; expects columns: feature, mean, scale.
     * Throws std::runtime_error if the file cannot be opened.
     */
    static ScalerParams loadFromCSV(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open())
            throw std::runtime_error("ScalerParams: cannot open " + path);

        ScalerParams params;
        std::string line;
        std::getline(file, line);  // skip header: feature,mean,scale

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string name, meanStr, scaleStr;
            std::getline(ss, name,    ',');
            std::getline(ss, meanStr, ',');
            std::getline(ss, scaleStr, ',');
            params.mean.push_back(std::stod(meanStr));
            params.scale.push_back(std::stod(scaleStr));
        }
        return params;
    }

    // ------------------------------------------------------------------
    // Transform / inverse-transform
    // ------------------------------------------------------------------

    /**
     * Apply StandardScaler forward transform to a feature vector.
     * x_scaled[i] = (x[i] - mean[i]) / scale[i]
     */
    std::vector<double> transform(const std::vector<double>& x) const {
        if (x.size() != mean.size())
            throw std::runtime_error("ScalerParams::transform: size mismatch");
        std::vector<double> out(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            out[i] = (x[i] - mean[i]) / scale[i];
        return out;
    }

    /**
     * Inverse-transform a single scaled value using the scaler at index idx.
     * Used to recover the original price from the model's scaled prediction.
     */
    double inverseTransform(double xScaled, std::size_t idx = 0) const {
        if (idx >= mean.size())
            throw std::runtime_error("ScalerParams::inverseTransform: idx out of range");
        return xScaled * scale[idx] + mean[idx];
    }
};
