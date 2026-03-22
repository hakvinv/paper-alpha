"""
Chapter 4 — Carry Trade
Paper: Lustig, Roussanov & Verdelhan, "Common Risk Factors in Currency Markets",
       Review of Financial Studies, 2011.

ETF proxy: FXA (AUD) vs FXY (JPY) — long high-rate, short low-rate currency.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity

# ── Data ──────────────────────────────────────────────────────────────────────
prices = download(['FXA', 'FXY'], start="2007-06-01")
ret = prices.pct_change().dropna()

carry = ret['FXA'] - ret['FXY']
equal_fx = (ret['FXA'] + ret['FXY']) / 2

# ── Stats ─────────────────────────────────────────────────────────────────────
# Note: ETFs capture exchange rate movement only.
# The actual interest rate differential (2-3% historically)
# adds to the carry return but isn't reflected in ETF prices.
print_stats({
    "Carry (AUD−JPY)": carry,
    "Equal weight FX": equal_fx,
})

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plot_equity({
    "Carry (long AUD / short JPY)": carry,
    "Equal weight FX": equal_fx,
}, title="Ch.4 — AUD/JPY Carry Trade")
plt.savefig("ch04_carry.png", dpi=150)
plt.show()
