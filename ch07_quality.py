"""
Chapter 7 — Quality Minus Junk
Paper: Asness, Frazzini & Pedersen, "Quality Minus Junk",
       Review of Accounting Studies, 2019.

ETF proxy: QUAL (iShares MSCI USA Quality) vs SPY.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity

# ── Data ──────────────────────────────────────────────────────────────────────
prices = download(['QUAL', 'SPY'], start="2013-08-01")
ret = prices.pct_change().dropna()

qmj = ret['QUAL'] - ret['SPY']

# ── Stats ─────────────────────────────────────────────────────────────────────
print_stats({
    "SPY": ret['SPY'],
    "Quality (QUAL)": ret['QUAL'],
    "QMJ (QUAL − SPY)": qmj,
})

# Worst-month analysis
monthly_ret = ret.resample('ME').apply(lambda x: (1+x).prod()-1)
worst_months = monthly_ret['SPY'].nsmallest(10)
print("During 10 worst SPY months:")
for date in worst_months.index:
    print(f"  {date.strftime('%Y-%m')}: SPY {monthly_ret.loc[date,'SPY']:.1%}  "
          f"QUAL {monthly_ret.loc[date,'QUAL']:.1%}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plot_equity({
    "SPY": ret['SPY'],
    "Quality (QUAL)": ret['QUAL'],
}, title="Ch.7 — Quality (QUAL) vs SPY, 2013–present")
plt.savefig("ch07_quality.png", dpi=150)
plt.show()
