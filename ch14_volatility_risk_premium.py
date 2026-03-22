"""
Chapter 14 — Volatility Risk Premium
Paper: Ilmanen, "Expected Returns: An Investor's Guide to Harvesting
       Market Rewards", Wiley, 2011. Chapter 15.

VRP = implied vol (VIX) − realized vol. Backtest a simple short-vol strategy
using SVXY (0.5x short VIX futures ETF) with trend filter for risk management.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity, plot_drawdown

# ── Data ──────────────────────────────────────────────────────────────────────
# VIX for implied vol, SPY for realized vol, SVXY for tradeable short-vol
prices = download(['SPY', '^VIX'], start="2007-01-01")
spy_ret = prices['SPY'].pct_change().dropna()

# Realized vol (21-day)
realized_vol = spy_ret.rolling(21).std() * np.sqrt(252) * 100

# VIX (implied vol) — stored in the ^VIX column
vix = prices['^VIX'].dropna() if '^VIX' in prices.columns else None

# ── VRP calculation ───────────────────────────────────────────────────────────
if vix is not None:
    # Align indices
    common = vix.index.intersection(realized_vol.index)
    vrp = vix.loc[common] - realized_vol.loc[common]

    print(f"VRP statistics (VIX − Realized Vol):")
    print(f"  Mean: {vrp.mean():.1f} vol points")
    print(f"  Median: {vrp.median():.1f} vol points")
    print(f"  Positive {(vrp > 0).mean():.0%} of the time")
    print()

# ── Simple short-vol strategy using put credit spreads logic ──────────────────
# Proxy: allocate 10% to short-vol (via SVXY), 90% cash/SPY
# With trend filter: only short vol when VIX < 200-day MA

svxy_prices = download('SVXY', start="2011-10-01")
if 'SVXY' in svxy_prices.columns:
    svxy_ret = svxy_prices['SVXY'].pct_change().dropna()
    spy_from_svxy = download('SPY', start="2011-10-01")
    spy_ret2 = spy_from_svxy['SPY'].pct_change().dropna()

    # Align
    common = svxy_ret.index.intersection(spy_ret2.index)
    svxy_ret = svxy_ret.loc[common]
    spy_ret2 = spy_ret2.loc[common]

    # Strategy: 5% SVXY + 95% SPY (conservative short-vol allocation)
    port_no_filter = 0.05 * svxy_ret + 0.95 * spy_ret2

    # With VIX trend filter: only short vol when VIX below 200-day MA
    if vix is not None:
        vix_ma = vix.rolling(200).mean()
        vix_signal = (vix < vix_ma).reindex(common).fillna(False)
        port_filtered = spy_ret2.copy()
        port_filtered[vix_signal] = 0.05 * svxy_ret[vix_signal] + 0.95 * spy_ret2[vix_signal]
    else:
        port_filtered = port_no_filter

    print_stats({
        "SPY (100%)": spy_ret2,
        "5% SVXY + 95% SPY": port_no_filter,
        "5% SVXY + 95% SPY (VIX filter)": port_filtered,
    })
else:
    print("SVXY data not available. Using VIX analysis only.")

# ── Worst VIX days table ──────────────────────────────────────────────────────
print("\nThe five worst days for short-vol strategies since 2007:")
print("-" * 60)
worst_events = [
    ("Feb 5, 2018",  "+116%", "Volmageddon (XIV terminated)"),
    ("Mar 16, 2020", "VIX 82.69", "COVID crash"),
    ("Aug 5, 2024",  "+180%", "Yen carry unwind"),
    ("Aug 8, 2011",  "+50%", "US credit downgrade"),
    ("Oct 15, 2008", "+30%", "Lehman aftermath"),
]
for date, move, event in worst_events:
    print(f"  {date:15s}  VIX {move:>8s}  {event}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

if vix is not None:
    axes[0].plot(vix.index, vix.values, color='red', alpha=0.7, linewidth=0.5, label='VIX')
    rv_aligned = realized_vol.reindex(vix.index).dropna()
    axes[0].plot(rv_aligned.index, rv_aligned.values, color='steelblue',
                 alpha=0.7, linewidth=0.5, label='Realized Vol (21d)')
    axes[0].set_ylabel("Volatility (%)")
    axes[0].set_title("Ch.14 — VIX vs Realized Volatility")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

if 'SVXY' in svxy_prices.columns:
    for name, ret in [("SPY", spy_ret2), ("5% SVXY + 95% SPY", port_no_filter)]:
        cum = (1 + ret).cumprod()
        axes[1].plot(cum.index, cum.values, label=name)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Growth of $1 (log)")
    axes[1].set_title("Conservative Short-Vol Overlay")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ch14_vrp.png", dpi=150)
plt.show()

print("\nKey takeaway: Never sell naked options at retail scale.")
print("Maximum 5-10% of total portfolio. Defined-risk structures only.")
print("Accept that 2-3 years of premium will be lost in the next vol event.")
