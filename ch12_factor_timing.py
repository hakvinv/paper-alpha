"""
Chapter 12 — Factor Timing
Paper: Asness, "The Siren Song of Factor Timing",
       Journal of Portfolio Management, 2016.

No backtest by design — academic consensus says factor timing doesn't work.
This script computes the value spread to illustrate the one signal with
empirical support (extreme value spreads).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats

# ── Value spread proxy ────────────────────────────────────────────────────────
# Use VTV/VUG price ratio as a rough proxy for the value spread
prices = download(['VTV', 'VUG'], start="2005-01-01")
ratio = prices['VTV'] / prices['VUG']

# Rolling percentile of the value spread
rolling_pct = ratio.rolling(756).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
)

# ── Factor returns (momentum, value, low-vol) ────────────────────────────────
factor_prices = download(['MTUM', 'VLUE', 'SPLV', 'SPY'], start="2013-11-01")
factor_ret = factor_prices.pct_change().dropna()

# Rolling 12-month factor returns
rolling_12m = factor_ret.rolling(252).apply(lambda x: (1+x).prod()-1, raw=False)

# ── Stats for static factor portfolio ─────────────────────────────────────────
factor_equal = (factor_ret['MTUM'] + factor_ret['VLUE'] + factor_ret['SPLV']) / 3
print_stats({
    "SPY": factor_ret['SPY'],
    "Equal-wt Factor (MTUM+VLUE+SPLV)": factor_equal,
}, freq="daily")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

axes[0].plot(ratio.index, ratio.values, color='steelblue', linewidth=0.8)
axes[0].set_title("Value Spread Proxy (VTV / VUG price ratio)")
axes[0].set_ylabel("Ratio")
axes[0].grid(True, alpha=0.3)

axes[1].plot(rolling_pct.dropna().index, rolling_pct.dropna().values,
             color='darkorange', linewidth=0.8)
axes[1].axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90th pctl')
axes[1].axhline(0.1, color='green', linestyle='--', alpha=0.5, label='10th pctl')
axes[1].set_title("Rolling 3yr Percentile of Value Spread")
axes[1].set_ylabel("Percentile")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ch12_factor_timing.png", dpi=150)
plt.show()

print("\nConclusion: Hold a diversified factor portfolio at static weights.")
print("Rebalance annually. Don't try to predict which factor wins next quarter.")
print("The one signal with empirical support: extreme value spreads (>90th pctl).")
