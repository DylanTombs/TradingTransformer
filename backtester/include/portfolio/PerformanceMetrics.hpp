#pragma once

#include "portfolio/Portfolio.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

/**
 * PerformanceMetrics — computes industry-standard risk/return statistics
 * from a completed Portfolio equity curve.
 *
 * All Sharpe / IR figures are annualised using the supplied factor
 * (default 252 for daily bars).
 *
 * Formulas:
 *   daily_return[t]   = (equity[t] - equity[t-1]) / equity[t-1]
 *   excess_return[t]  = daily_return[t] - riskFreeRate / annFactor
 *   Sharpe            = mean(excess) / std(excess) * sqrt(annFactor)
 *
 *   active_return[t]  = daily_return[t] - benchmark_daily_return[t]
 *   InformationRatio  = mean(active) / std(active) * sqrt(annFactor)
 *
 *   maxDrawdown       = max over t of (peak_equity - equity[t]) / peak_equity
 *   annualisedReturn  = (endEquity/startEquity)^(annFactor/n) - 1
 */
struct PerformanceMetrics {

    double sharpeRatio      = 0.0;  ///< Annualised Sharpe (strategy vs risk-free)
    double informationRatio = 0.0;  ///< Annualised IR (strategy vs benchmark)
    double maxDrawdown      = 0.0;  ///< Maximum peak-to-trough drawdown (fraction)
    double totalReturn      = 0.0;  ///< Total strategy return over the period
    double annualisedReturn = 0.0;  ///< CAGR over the period
    double benchmarkReturn  = 0.0;  ///< Total buy-and-hold return
    double alpha            = 0.0;  ///< totalReturn - benchmarkReturn
    int    tradingDays      = 0;    ///< Number of equity curve entries

    // -------------------------------------------------------------------------
    // Factory
    // -------------------------------------------------------------------------
    static PerformanceMetrics compute(
        const std::vector<EquityPoint>& curve,
        double riskFreeRate      = 0.0,
        int    annualisationFactor = 252)
    {
        PerformanceMetrics m;
        if (curve.size() < 2) return m;

        m.tradingDays = static_cast<int>(curve.size());

        const double startEquity = curve.front().equity;
        const double endEquity   = curve.back().equity;

        m.totalReturn  = (endEquity / startEquity) - 1.0;
        m.annualisedReturn = std::pow(endEquity / startEquity,
                                     static_cast<double>(annualisationFactor) /
                                     static_cast<double>(curve.size() - 1)) - 1.0;

        m.benchmarkReturn = (curve.back().benchmarkEquity /
                             curve.front().benchmarkEquity) - 1.0;
        m.alpha = m.totalReturn - m.benchmarkReturn;

        // ---- Daily returns -------------------------------------------------
        const int n = static_cast<int>(curve.size()) - 1;
        std::vector<double> stratReturns(n);
        std::vector<double> benchReturns(n);
        std::vector<double> activeReturns(n);

        for (int i = 0; i < n; ++i) {
            stratReturns[i] =
                (curve[i + 1].equity - curve[i].equity) / curve[i].equity;
            benchReturns[i] =
                (curve[i + 1].benchmarkEquity - curve[i].benchmarkEquity) /
                curve[i].benchmarkEquity;
            activeReturns[i] = stratReturns[i] - benchReturns[i];
        }

        // ---- Sharpe ratio --------------------------------------------------
        const double rfDaily = riskFreeRate / annualisationFactor;
        std::vector<double> excess(n);
        for (int i = 0; i < n; ++i) excess[i] = stratReturns[i] - rfDaily;

        m.sharpeRatio = annualisedSharpe(excess, annualisationFactor);

        // ---- Information ratio (Sharpe vs benchmark) ----------------------
        m.informationRatio = annualisedSharpe(activeReturns, annualisationFactor);

        // ---- Maximum drawdown ---------------------------------------------
        double peak = curve.front().equity;
        for (const auto& pt : curve) {
            peak = std::max(peak, pt.equity);
            const double dd = (peak - pt.equity) / peak;
            m.maxDrawdown = std::max(m.maxDrawdown, dd);
        }

        return m;
    }

    // -------------------------------------------------------------------------
    // Output
    // -------------------------------------------------------------------------
    void print(std::ostream& out) const {
        out << std::fixed;
        out.precision(2);
        out << "\n=== Performance Metrics ===\n"
            << "  Trading days       : "  << tradingDays                    << "\n"
            << "  Total return       :  " << totalReturn      * 100 << " %\n"
            << "  Benchmark return   :  " << benchmarkReturn  * 100 << " %\n"
            << "  Alpha              :  " << alpha            * 100 << " %\n"
            << "  Annualised return  :  " << annualisedReturn * 100 << " %\n";
        out.precision(4);
        out << "  Sharpe ratio       :  " << sharpeRatio       << "\n"
            << "  Information ratio  :  " << informationRatio  << "\n";
        out.precision(2);
        out << "  Max drawdown       :  " << maxDrawdown * 100 << " %\n";
    }

    void exportCSV(const std::string& path) const {
        std::ofstream f(path);
        f << "metric,value\n"
          << "trading_days,"       << tradingDays                    << "\n"
          << "total_return,"       << totalReturn      * 100         << "\n"
          << "benchmark_return,"   << benchmarkReturn  * 100         << "\n"
          << "alpha,"              << alpha            * 100         << "\n"
          << "annualised_return,"  << annualisedReturn * 100         << "\n"
          << "sharpe_ratio,"       << sharpeRatio                    << "\n"
          << "information_ratio,"  << informationRatio               << "\n"
          << "max_drawdown,"       << maxDrawdown      * 100         << "\n";
    }

private:
    // Annualised Sharpe = mean(r) / std(r) * sqrt(annFactor)
    // Returns 0 when sample size < 2 or std-dev is near-zero.
    static double annualisedSharpe(const std::vector<double>& r, int annFactor) {
        const int n = static_cast<int>(r.size());
        if (n < 2) return 0.0;

        const double mean = std::accumulate(r.begin(), r.end(), 0.0) / n;

        double var = 0.0;
        for (double v : r) var += (v - mean) * (v - mean);
        var /= (n - 1);  // sample variance (Bessel-corrected)

        const double sd = std::sqrt(var);
        if (sd < 1e-12) return 0.0;

        return (mean / sd) * std::sqrt(static_cast<double>(annFactor));
    }
};
