"""
Chapter 3: Value
Fama & French, 'The Cross-Section of Expected Stock Returns',
Journal of Finance, 1992. 25,000+ citations.

HML = Book equity / Market cap. Retail implementation: VTV (value) vs
VUG (growth) vs SPY. The value premium was negative over 2005-2025.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = yf.download(['VTV', 'VUG', 'SPY'],
                    start="2005-01-01", auto_adjust=True)
prices = data['Close']
if isinstance(prices.columns, pd.MultiIndex):
    prices = prices.droplevel(0, axis=1)

r_val = prices['VTV'].pct_change().dropna()
r_gro = prices['VUG'].pct_change().dropna()
r_spy = prices['SPY'].pct_change().dropna()

idx = r_val.index.intersection(r_gro.index).intersection(r_spy.index)
hml = r_val[idx] - r_gro[idx]

# Rolling 3-year premium
rolling_hml = hml.rolling(756).mean() * 252
print(f"HML annual return: {hml.mean()*252:.1%}")
print(f"Rolling 3yr range: {rolling_hml.min():.1%} to "
      f"{rolling_hml.max():.1%}")

# Backtest stats
for name, r in [("SPY", r_spy[idx]), ("Value (VTV)", r_val[idx]),
                ("Growth (VUG)", r_gro[idx]), ("Value-Growth", hml)]:
    ann_r = r.mean() * 252
    ann_v = r.std() * np.sqrt(252)
    sharpe = ann_r / ann_v
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"{name:15s} Ret={ann_r:.1%} Vol={ann_v:.1%} "
          f"Sharpe={sharpe:.2f} MaxDD={mdd:.1%}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for name, r in [("SPY", r_spy[idx]), ("Value (VTV)", r_val[idx]),
                ("Growth (VUG)", r_gro[idx])]:
    cum = (1 + r).cumprod()
    ax.plot(cum.index, cum, label=name)
ax.set_yscale('log')
ax.set_ylabel('Growth of $1 (log)')
ax.set_title('Value vs Growth vs Market, 2005-present')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_ch03_value.png', dpi=150)
plt.show()
