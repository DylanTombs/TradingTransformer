import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./research/analysis/equity.csv", parse_dates=["timestamp"])
trades = pd.read_csv("./research/analysis/trades.csv", parse_dates=["timestamp"])

fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.set_xlabel("Date")
ax1.set_ylabel("Equity ($)", color="tab:blue")
ax1.plot(df["timestamp"], df["equity"], color="tab:blue", label="Equity")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Price ($)", color="tab:orange")
ax2.plot(df["timestamp"], df["price"], color="tab:orange", alpha=0.4, label="Price")
ax2.tick_params(axis="y", labelcolor="tab:orange")

# Plot trade arrows on price axis
for _, trade in trades.iterrows():
    color = "green" if trade["profit"] else "red"
    marker = "^" if trade["direction"] == "BUY" else "v"
    ax2.scatter(trade["timestamp"], trade["price"],
                color=color, marker=marker, zorder=5, s=100)

fig.suptitle("Strategy Equity Curve vs Price")
fig.tight_layout()
plt.show()