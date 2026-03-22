"""
Chapter 10 — Pairs Trading
Paper: Gatev, Goetzmann & Rouwenhorst, "Pairs Trading: Performance of a
       Relative-Value Arbitrage Rule", Review of Financial Studies, 2006.

Three classic pairs: KO/PEP, XOM/CVX, GS/MS. Z-score entry/exit.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity

# ── Parameters ────────────────────────────────────────────────────────────────
PAIRS = [('KO', 'PEP'), ('XOM', 'CVX'), ('GS', 'MS')]
START = "2005-01-01"
LOOKBACK = 60       # days for z-score mean/std
ENTRY = 2.0         # z-score entry threshold
EXIT = 0.5          # z-score exit threshold
STOP = 4.0          # z-score stop-loss

# ── Data ──────────────────────────────────────────────────────────────────────
all_tickers = list(set(t for pair in PAIRS for t in pair))
prices = download(all_tickers, start=START)

# ── Pairs trading engine ──────────────────────────────────────────────────────
def backtest_pair(prices, a, b, lookback=60, entry=2.0, exit_z=0.5, stop=4.0):
    """Backtest a single pair. Returns daily PnL series and trade count."""
    ratio = prices[a] / prices[b]
    mu = ratio.rolling(lookback).mean()
    sigma = ratio.rolling(lookback).std()
    z = (ratio - mu) / sigma

    position = pd.Series(0.0, index=z.index)
    trades = 0

    for i in range(1, len(z)):
        prev_pos = position.iloc[i-1]
        z_val = z.iloc[i]

        if np.isnan(z_val):
            position.iloc[i] = 0
            continue

        if prev_pos == 0:
            if z_val > entry:
                position.iloc[i] = -1   # short spread (short A, long B)
                trades += 1
            elif z_val < -entry:
                position.iloc[i] = 1    # long spread (long A, short B)
                trades += 1
            else:
                position.iloc[i] = 0
        elif prev_pos > 0:  # long spread
            if z_val > -exit_z or abs(z_val) > stop:
                position.iloc[i] = 0
            else:
                position.iloc[i] = prev_pos
        elif prev_pos < 0:  # short spread
            if z_val < exit_z or abs(z_val) > stop:
                position.iloc[i] = 0
            else:
                position.iloc[i] = prev_pos

    # PnL: position × (return_A − return_B)
    ret_a = prices[a].pct_change()
    ret_b = prices[b].pct_change()
    spread_ret = ret_a - ret_b
    pnl = position.shift(1) * spread_ret
    return pnl.dropna(), trades

# ── Run all pairs ─────────────────────────────────────────────────────────────
results = {}
for a, b in PAIRS:
    pnl, trades = backtest_pair(prices, a, b, LOOKBACK, ENTRY, EXIT, STOP)
    pair_name = f"{a} / {b}"
    results[pair_name] = pnl
    stats = {
        "Return": f"{pnl.mean()*252:.1%}",
        "Vol": f"{pnl.std()*np.sqrt(252):.1%}",
        "Sharpe": f"{pnl.mean()/pnl.std()*np.sqrt(252):.2f}" if pnl.std()>0 else "0",
        "Max DD": f"{((1+pnl).cumprod() / (1+pnl).cumprod().cummax() - 1).min():.1%}",
        "Trades": trades,
    }
    print(f"{pair_name:12s}  {stats}")

print()
print_stats(results)

# ── Plot: Z-score for KO/PEP ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

a, b = 'KO', 'PEP'
ratio = prices[a] / prices[b]
mu = ratio.rolling(LOOKBACK).mean()
sigma = ratio.rolling(LOOKBACK).std()
z = ((ratio - mu) / sigma).dropna()

axes[0].plot(z.index, z.values, linewidth=0.7)
axes[0].axhline(ENTRY, color='green', linestyle='--', alpha=0.5, label='Entry')
axes[0].axhline(-ENTRY, color='green', linestyle='--', alpha=0.5)
axes[0].axhline(0, color='gray', alpha=0.3)
axes[0].set_ylabel(f"Z-score ({a}/{b})")
axes[0].legend()
axes[0].set_title("Ch.10 — Pairs Trading Z-Score")

# Cumulative PnL for all pairs
for name, pnl in results.items():
    cum = (1 + pnl).cumprod()
    axes[1].plot(cum.index, cum.values, label=name)
axes[1].set_ylabel("Growth of $1")
axes[1].legend()
axes[1].set_title("Cumulative PnL")
plt.tight_layout()
plt.savefig("ch10_pairs.png", dpi=150)
plt.show()
