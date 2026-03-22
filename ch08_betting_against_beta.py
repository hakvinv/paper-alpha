"""
Chapter 8 — Betting Against Beta
Paper: Frazzini & Pedersen, "Betting Against Beta",
       Journal of Financial Economics, 2014.

Sector ETF implementation: sort by trailing 12-month beta, long low-beta, short high-beta.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity

# ── Parameters ────────────────────────────────────────────────────────────────
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
START = "2005-01-01"
BETA_WINDOW = 252  # trailing 12-month daily
TOP_N = 3          # number of low/high-beta sectors

# ── Data ──────────────────────────────────────────────────────────────────────
all_tickers = SECTORS + ['SPY']
prices = download(all_tickers, start=START)
ret = prices.pct_change().dropna()

# ── Compute rolling betas ─────────────────────────────────────────────────────
spy_ret = ret['SPY']
betas = pd.DataFrame(index=ret.index, columns=SECTORS, dtype=float)

for tk in SECTORS:
    cov = ret[tk].rolling(BETA_WINDOW).cov(spy_ret)
    var = spy_ret.rolling(BETA_WINDOW).var()
    betas[tk] = cov / var

# ── Monthly rebalancing ───────────────────────────────────────────────────────
monthly_prices = prices.resample('ME').last()
monthly_ret = monthly_prices.pct_change()
monthly_betas = betas.resample('ME').last()

low_beta_port = pd.Series(dtype=float)
high_beta_port = pd.Series(dtype=float)
bab_port = pd.Series(dtype=float)
bench_port = pd.Series(dtype=float)

for date in monthly_ret.index[13:]:
    b = monthly_betas.loc[date, SECTORS].dropna()
    if len(b) < 2 * TOP_N:
        continue

    b_sorted = b.sort_values()
    low = b_sorted.index[:TOP_N]
    high = b_sorted.index[-TOP_N:]

    r_low = monthly_ret.loc[date, low].mean()
    r_high = monthly_ret.loc[date, high].mean()
    r_bench = monthly_ret.loc[date, 'SPY']

    low_beta_port[date] = r_low
    high_beta_port[date] = r_high
    bab_port[date] = r_low - r_high
    bench_port[date] = r_bench

# ── Stats ─────────────────────────────────────────────────────────────────────
print_stats({
    "SPY": bench_port,
    "Low Beta (long only)": low_beta_port,
    "BAB L/S (low − high)": bab_port,
}, freq="monthly")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plot_equity({
    "SPY": bench_port,
    "Low Beta Sectors": low_beta_port,
    "BAB L/S": bab_port,
}, title="Ch.8 — Betting Against Beta (Sector ETFs)")
plt.savefig("ch08_bab.png", dpi=150)
plt.show()
