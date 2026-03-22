"""
Chapter 3 — Value
Paper: Fama & French, "The Cross-Section of Expected Stock Returns",
       Journal of Finance, 1992.

ETF proxy: VTV (Value) vs VUG (Growth) vs SPY. Compute HML premium.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity

# ── Data ──────────────────────────────────────────────────────────────────────
prices = download(['VTV', 'VUG', 'SPY'], start="2005-01-01")
ret = prices.pct_change().dropna()

hml = ret['VTV'] - ret['VUG']

# Rolling 3-year premium
rolling_hml = hml.rolling(756).mean() * 252

# ── Stats ─────────────────────────────────────────────────────────────────────
print(f"HML annual return: {hml.mean()*252:.1%}")
print(f"Rolling 3yr range: {rolling_hml.min():.1%} to {rolling_hml.max():.1%}")
print()
print_stats({
    "SPY (market)": ret['SPY'],
    "Value (VTV)": ret['VTV'],
    "Growth (VUG)": ret['VUG'],
    "Value−Growth": hml,
})

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plot_equity({
    "SPY": ret['SPY'],
    "Value (VTV)": ret['VTV'],
    "Growth (VUG)": ret['VUG'],
}, title="Ch.3 — Value vs Growth, 2005–present")
plt.savefig("ch03_value.png", dpi=150)
plt.show()
