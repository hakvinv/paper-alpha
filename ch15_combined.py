"""
Chapter 15 — Putting It Together
Paper: Hamill, Rattray & Van Hemert, "Trend Following: Equity and Bond
       Crisis Alpha", Man Group / AHL, 2016.

Combines trend following (Ch.6) + risk parity (Ch.11) into a single portfolio.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity, plot_drawdown

# ── Parameters ────────────────────────────────────────────────────────────────
TICKERS = ['SPY', 'TLT', 'GLD', 'DBC']
START = "2007-01-01"
VOL_WINDOW = 12   # months for risk parity vol estimate
TREND_WINDOW = 12  # months for trend signal

# ── Data ──────────────────────────────────────────────────────────────────────
prices = download(TICKERS, start=START)
mp = prices.resample('ME').last()
mr = mp.pct_change().dropna()
sig = mp.pct_change(TREND_WINDOW).shift(1)

# ── Build portfolios ──────────────────────────────────────────────────────────
combo = pd.Series(dtype=float)
rp_only = pd.Series(dtype=float)
tf_only = pd.Series(dtype=float)
sixty_forty = pd.Series(dtype=float)

for i in range(TREND_WINDOW, len(mr)):
    date = mr.index[i]
    r = mr.iloc[i]

    # Risk parity weights (inverse vol)
    vols = mr[TICKERS].iloc[i-VOL_WINDOW:i].std()
    rp_w = (1.0 / vols) / (1.0 / vols).sum()

    # Trend + Risk Parity: apply trend filter to risk parity weights
    total = 0.0
    for tk in TICKERS:
        trend_on = sig.loc[date, tk] > 0
        total += rp_w[tk] * r[tk] * (1.0 if trend_on else 0.0)
    combo[date] = total

    # Risk parity only
    rp_only[date] = (rp_w * r[TICKERS]).sum()

    # Trend following only (equal weight)
    n = len(TICKERS)
    tf_only[date] = sum(r[tk] / n for tk in TICKERS if sig.loc[date, tk] > 0)

    # 60/40
    sixty_forty[date] = 0.6 * r['SPY'] + 0.4 * r['TLT']

# ── Stats ─────────────────────────────────────────────────────────────────────
print_stats({
    "60/40 (SPY/TLT)": sixty_forty,
    "Risk Parity only": rp_only,
    "Trend Following only": tf_only,
    "Trend + Risk Parity": combo,
}, freq="monthly")

# ── Plot: equity curves ───────────────────────────────────────────────────────
fig = plot_equity({
    "60/40": sixty_forty,
    "Risk Parity": rp_only,
    "Trend (equal wt)": tf_only,
    "Trend + Risk Parity": combo,
}, title="Ch.15 — Combined Strategy Comparison")
plt.savefig("ch15_combined.png", dpi=150)

# ── Plot: drawdowns ───────────────────────────────────────────────────────────
plot_drawdown({
    "60/40": sixty_forty,
    "Trend + Risk Parity": combo,
}, title="Drawdown: 60/40 vs Combined")
plt.savefig("ch15_drawdown.png", dpi=150)

plt.show()

# ── Three-portfolio menu ──────────────────────────────────────────────────────
print("=" * 60)
print("THE THREE-PORTFOLIO MENU")
print("=" * 60)
print()
print("Conservative (Max DD target: <12%)")
print("  Trend + Risk Parity from this chapter.")
print("  4 ETFs: SPY, TLT, GLD, DBC")
print("  Monthly rebalancing, risk parity weights + 12-month trend filter")
print()
print("Moderate (Max DD target: <20%)")
print("  Risk Parity from Chapter 11, without trend filter.")
print("  Same 4 ETFs, inverse-volatility weights, monthly rebalancing")
print()
print("Growth (Max DD target: <30%)")
print("  60/40 SPY/TLT with vol targeting from Chapter 1.")
print("  Scale equity exposure inversely to realized volatility")
