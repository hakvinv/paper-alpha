"""
Chapter 2 — Momentum
Paper: Jegadeesh & Titman, "Returns to Buying Winners and Selling Losers",
       Journal of Finance, 1993.

12-1 momentum on sector ETFs: long top 3, rebalance monthly.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity

# ── Parameters ────────────────────────────────────────────────────────────────
TICKERS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
START = "2002-01-01"
TOP_N = 3

# ── Data ──────────────────────────────────────────────────────────────────────
prices = download(TICKERS, start=START)

# Monthly prices and returns
mp = prices.resample('ME').last()
monthly_ret = mp.pct_change()

# 12-1 momentum signal: 12-month return, skip most recent month
signal = mp.pct_change(12).shift(1)

# ── Build portfolio ───────────────────────────────────────────────────────────
port = pd.Series(dtype=float)
bench = pd.Series(dtype=float)

for date in monthly_ret.index[13:]:
    sig = signal.loc[date].dropna().sort_values()
    if len(sig) < 2 * TOP_N:
        continue
    top = sig.index[-TOP_N:]
    port[date] = monthly_ret.loc[date, top].mean()
    bench[date] = monthly_ret.loc[date, TICKERS].mean()

# ── Stats ─────────────────────────────────────────────────────────────────────
print_stats({"Equal weight (all 9)": bench, f"Momentum (top {TOP_N})": port}, freq="monthly")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plot_equity({"Equal Weight": bench, f"Momentum Top {TOP_N}": port},
                  title="Ch.2 — Sector Momentum (12-1)")
plt.savefig("ch02_momentum.png", dpi=150)
plt.show()
