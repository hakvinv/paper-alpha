"""
Chapter 1: Volatility Targeting
Moreira & Muir, 'Volatility-Managed Portfolios', Journal of Finance, 2017.
1,200+ citations.

Scale equity exposure inversely to recent volatility using EWMA.
sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_{t-1}  (lambda=0.94)
w_t = sigma_target / sigma_ann_t, capped at 2.0
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

spy = yf.download("SPY", start="2000-01-01", auto_adjust=True)["Close"]
if isinstance(spy, pd.DataFrame):
    spy = spy.iloc[:, 0]
ret = spy.pct_change().dropna()

lam = 0.94
vol_target = 0.10
max_lev = 2.0

# EWMA variance
var = pd.Series(index=ret.index, dtype=float)
var.iloc[0] = ret.iloc[:20].var()
for i in range(1, len(ret)):
    var.iloc[i] = lam*var.iloc[i-1] + (1-lam)*ret.iloc[i-1]**2

vol = np.sqrt(var) * np.sqrt(252)    # annualize
w = (vol_target / vol).clip(upper=max_lev)

# Strategy return: yesterday's weight * today's return
strat = w.shift(1) * ret

# Performance comparison
for name, r in [("Buy & Hold", ret), ("Vol Targeted", strat)]:
    r = r.dropna()
    ann_r = r.mean() * 252
    ann_v = r.std() * np.sqrt(252)
    sharpe = ann_r / ann_v
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"{name:15s} Ret={ann_r:.1%} Vol={ann_v:.1%} "
          f"Sharpe={sharpe:.2f} MaxDD={mdd:.1%}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax1.plot(vol.index, vol*100, color='red', linewidth=0.5)
ax1.axhline(vol_target*100, color='blue', linestyle='--', alpha=0.5)
ax1.set_ylabel('Ann. vol (%)')
ax1.set_title('Volatility Targeting: SPY')

ax2.fill_between(w.index, 0, w, alpha=0.3)
ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('Position size')
ax2.set_xlabel('Date')

plt.tight_layout()
plt.savefig('fig_ch01_vol_targeting.png', dpi=150)
plt.show()
