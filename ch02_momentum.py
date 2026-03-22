"""
Chapter 2: Momentum
Jegadeesh & Titman, 'Returns to Buying Winners and Selling Losers',
Journal of Finance, 1993. 13,000+ citations.

12-month lookback, 1-month skip. Rank sector ETFs, long top 3 monthly.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sector ETFs as universe
tickers = ['XLK','XLF','XLE','XLV','XLI','XLY','XLP','XLU','XLB']
data = yf.download(tickers, start="2002-01-01", auto_adjust=True)['Close']
if isinstance(data.columns, pd.MultiIndex):
    data = data.droplevel(0, axis=1)

# Monthly prices
mp = data.resample('ME').last()
monthly_ret = mp.pct_change()

# 12-1 momentum signal
signal = mp.pct_change(12).shift(1)

# Build portfolio: long top 3 sectors each month
port = pd.Series(dtype=float)
bench = pd.Series(dtype=float)
for date in monthly_ret.index[13:]:
    sig = signal.loc[date].dropna().sort_values()
    if len(sig) < 6:
        continue
    top3 = sig.index[-3:]
    port[date] = monthly_ret.loc[date, top3].mean()
    bench[date] = monthly_ret.loc[date].mean()

# Annualize (monthly data)
for name, r in [("Equal weight", bench), ("Mom top 3", port)]:
    ann_r = r.mean() * 12
    ann_v = r.std() * np.sqrt(12)
    sharpe = ann_r / ann_v
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"{name:15s} Ret={ann_r:.1%} Vol={ann_v:.1%} "
          f"Sharpe={sharpe:.2f} MaxDD={mdd:.1%}")

# Plot
cum_b = (1 + bench).cumprod()
cum_p = (1 + port).cumprod()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cum_b.index, cum_b, color='gray', label='Equal Weight')
ax.plot(cum_p.index, cum_p, color='steelblue', label='Momentum Top 3')
ax.set_yscale('log')
ax.set_ylabel('Growth of $1 (log)')
ax.set_title('Sector Momentum: top 3 by 12-1 month return')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_ch02_momentum.png', dpi=150)
plt.show()
