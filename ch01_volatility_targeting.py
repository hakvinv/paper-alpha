"""
Chapter 1 — Volatility Targeting
Paper: Moreira & Muir, "Volatility-Managed Portfolios", Journal of Finance, 2017.

Scale equity exposure inversely to recent volatility using EWMA.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity, plot_drawdown

# ── Parameters ────────────────────────────────────────────────────────────────
TICKER = "SPY"
START = "2000-01-01"
LAMBDA = 0.94          # RiskMetrics decay factor
VOL_TARGET = 0.10      # 10% annualized vol target
MAX_LEV = 2.0          # max leverage cap

# ── Data ──────────────────────────────────────────────────────────────────────
prices = download(TICKER, start=START)
ret = prices[TICKER].pct_change().dropna()

# ── EWMA variance ─────────────────────────────────────────────────────────────
var = pd.Series(index=ret.index, dtype=float)
var.iloc[0] = ret.iloc[:20].var()
for i in range(1, len(ret)):
    var.iloc[i] = LAMBDA * var.iloc[i-1] + (1 - LAMBDA) * ret.iloc[i-1]**2

vol = np.sqrt(var) * np.sqrt(252)   # annualize
w = (VOL_TARGET / vol).clip(upper=MAX_LEV)

# ── Strategy returns ──────────────────────────────────────────────────────────
strat = w.shift(1) * ret  # yesterday's weight × today's return
strat = strat.dropna()
bench = ret.loc[strat.index]

# ── Stats ─────────────────────────────────────────────────────────────────────
print_stats({"Buy & Hold": bench, "Vol Targeted": strat})

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(vol.index, vol.values * 100, color="red", alpha=0.7, linewidth=0.8)
axes[0].axhline(VOL_TARGET * 100, color="steelblue", linestyle="--", alpha=0.7)
axes[0].set_ylabel("Ann. vol (%)")
axes[0].set_title("SPY Annualized Volatility vs 10% Target")

axes[1].fill_between(w.index, w.values, alpha=0.3, color="steelblue")
axes[1].plot(w.index, w.values, color="steelblue", linewidth=0.5)
axes[1].set_ylabel("Position size")
axes[1].set_title("Vol-Targeted Weight")
plt.tight_layout()
plt.savefig("ch01_vol_target.png", dpi=150)

plot_equity({"Buy & Hold": bench, "Vol Targeted": strat},
            title="Ch.1 — Volatility Targeting")
plt.savefig("ch01_equity.png", dpi=150)

plot_drawdown({"Buy & Hold": bench, "Vol Targeted": strat})
plt.savefig("ch01_drawdown.png", dpi=150)

plt.show()
