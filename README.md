# Transformer-Powered Trading Simulator

A powerful **backtesting framework** built on [Backtrader](https://www.backtrader.com/) that integrates a **custom Transformer model** for time series forecasting. This project demonstrates how deep learning — specifically a **Transformer trained on market features** — can be used to guide trading decisions in a systematic, backtestable way.

---

## Project Focus

This project is **centered around a Transformer neural network** that I have created and trained on multivariate financial time series data (price + technical indicators). It uses the model's **future price prediction** to influence live trading decisions within a backtesting simulation.

---

## Most Recent Results

### Strategy Performance

<p align="center">
  <img src="Results/Results2/BX_Strategy.png" width="45%" />
  <img src="Results/Results2/KDP_Strategy.png" width="45%" />
</p>

<p align="center">
  <img src="Results/Results2/PEP_Strategy.png" width="45%" />
  <img src="Results/Results2/KDP_Strategy.png" width="45%" />
</p>

<p align="center">
  <img src="Results/Results2/UNH_Strategy.png" width="90%" />
</p>

### Equity Curves

<p align="center">
  <img src="Results/Results2/BX_Equity_Curve.png" width="45%" />
  <img src="Results/Results2/KDP_Equity_Curve.png" width="45%" />
</p>

<p align="center">
  <img src="Results/Results2/PEP_Equity_Curve.png" width="45%" />
  <img src="Results/Results2/KDP_Equity_Curve.png" width="45%" />
</p>

<p align="center">
  <img src="Results/Results2/UNH_Equity_Curve.png" width="90%" />
</p>

### Multi-Stock Comparisons

<p align="center">
  <img src="Results/Results2/trade_distributions.png" width="90%" />
</p>

<p align="center">
  <img src="Results/Results2/returns_by_symbol.png" width="90%" />
</p>

<p align="center">
  <img src="Results/Results2/risk_reward.png" width="90%" />
</p>

<p align="center">
  <img src="Results/Results2/performance_comparison.png" width="90%" />
</p>


## Strategy: `RsiEmaStrategy`

This strategy is a **hybrid of deep learning and technical analysis**:

- Uses RSI + EMA for signal confirmation
- The core decision logic is driven by my **Transformer model** trained to predict the `close` price based on a feature-rich window of past data
- Trades are only triggered when both the **Transformer prediction** and traditional indicators align

### Input Features to Transformer:
- `close`
- `volume_zscore`, `rsi`, `macd`, `overnight_gap`, `return_lag_1`, `return_lag_3`, `return_lag_5`, `volatility`

### New Input Features to Transformer:
- `close`
- `high`, `low`, `volume`,`adj close`,`P`, `R1`, `R2`, `R3`, `S1`, `S2`, `S3`,`obv`, `volume_zscore`, `rsi`, `macd`,`macds`,`macdh`, `sma`,`lma`,`sema`,`lema`,`overnight_gap`, `return_lag_1`, `return_lag_3`, `return_lag_5`, `volatility`, `SR_K`, `SR_D`, `SR_RSI_K`, `SR_RSI_D`, `ATR`, `HL_PCT`, `PCT_CHG`

### Trigger Logic:

```python
If RSI < 40 and Predicted Close > Current Price * 1.005 → BUY
If RSI > 60 and Predicted Close < Current Price * 0.995 → SELL
```

## Transformer Model Details

### Architecture: Encoder-decoder Transformer

### Trained with:

seq_len = 30
label_len = 10
pred_len = 5
Loss Function: MSE
Framework: PyTorch
Uses multivariate inputs and predicts future close prices
Scaling handled via scaler.pkl

### Previous Analysis

### Calculates:

- Sharpe Ratio
- Max Drawdown
- Win Rate
- Profit Factor
- Avg Risk/Reward
- Total Return

### Previous Performance Metrics 

- Sharpe Ratio: 0.43
- Max Drawdown: 3.63%
- Win Rate: 78.09%
- Profit Factor: 3.43
- Total Return: 95%

![Test One](Results/Results1/Results2.png)




