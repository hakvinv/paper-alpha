"""
Chapter 6 — Trend Following
Paper: Moskowitz, Ooi & Pedersen, "Time Series Momentum",
       Journal of Financial Economics, 2012.

3-asset trend following: SPY, TLT, GLD. Binary 12-month signal.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity

# ── Parameters ────────────────────────────────────────────────────────────────
TICKERS = ['SPY', 'TLT', 'GLD']
START = "2005-01-01"
LOOKBACK = 12  # months

# ── Data ──────────────────────────────────────────────────────────────────────
prices = download(TICKERS, start=START)
mp = prices.resample('ME').last()
mr = mp.pct_change().dropna()
sig = mp.pct_change(LOOKBACK).shift(1)
n = len(TICKERS)

# ── Build portfolio ───────────────────────────────────────────────────────────
tf = pd.Series(dtype=float)
bh = pd.Series(dtype=float)

for date in mr.index[13:]:
    tf_r = sum(mr.loc[date, tk] / n for tk in TICKERS if sig.loc[date, tk] > 0)
    bh_r = mr.loc[date, TICKERS].mean()
    tf[date] = tf_r
    bh[date] = bh_r

# ── Stats ─────────────────────────────────────────────────────────────────────
print_stats({"Buy & Hold (equal wt)": bh, f"Trend Following ({LOOKBACK}m)": tf}, freq="monthly")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plot_equity({
    "Buy & Hold (equal wt SPY/TLT/GLD)": bh,
    f"Trend Following ({LOOKBACK}m signal)": tf,
}, title="Ch.6 — Trend Following, 3-asset")
plt.savefig("ch06_trend.png", dpi=150)
plt.show()
