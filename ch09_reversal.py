"""
Chapter 9 — Short-Term Reversal
Paper: Lehmann, "Fads, Martingales, and Market Efficiency", QJE, 1990.
       Lo & MacKinlay, "When Are Contrarian Profits Due to Stock Market Overreaction?",
       Review of Financial Studies, 1990.

Sector ETF weekly reversal: buy last week's losers, sell last week's winners.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity

# ── Parameters ────────────────────────────────────────────────────────────────
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
START = "2004-01-01"
TOP_N = 3

# ── Data ──────────────────────────────────────────────────────────────────────
prices = download(SECTORS, start=START)

# Weekly prices and returns
wp = prices.resample('W-FRI').last()
wr = wp.pct_change().dropna()

# ── Build reversal portfolio ──────────────────────────────────────────────────
reversal_long = pd.Series(dtype=float)
reversal_ls = pd.Series(dtype=float)
bench = pd.Series(dtype=float)

for i in range(1, len(wr)):
    date = wr.index[i]
    prev = wr.iloc[i-1].dropna().sort_values()

    if len(prev) < 2 * TOP_N:
        continue

    losers = prev.index[:TOP_N]      # buy last week's worst
    winners = prev.index[-TOP_N:]    # sell last week's best

    r_losers = wr.loc[date, losers].mean()
    r_winners = wr.loc[date, winners].mean()

    reversal_long[date] = r_losers
    reversal_ls[date] = r_losers - r_winners
    bench[date] = wr.loc[date].mean()

# ── Stats ─────────────────────────────────────────────────────────────────────
print_stats({
    "Equal Weight Sectors": bench,
    "Reversal Long-Only": reversal_long,
    "Reversal L/S": reversal_ls,
}, freq="weekly")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plot_equity({
    "Equal Weight": bench,
    "Reversal Long (last week losers)": reversal_long,
    "Reversal L/S": reversal_ls,
}, title="Ch.9 — Short-Term Reversal (Weekly, Sector ETFs)")
plt.savefig("ch09_reversal.png", dpi=150)
plt.show()
