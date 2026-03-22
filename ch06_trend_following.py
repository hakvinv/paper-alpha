"""
Chapter 6: Trend Following
Moskowitz, Ooi & Pedersen, 'Time Series Momentum', Journal of Financial
Economics, 2012. 2,000+ citations.

Binary signal: if 12-month return > 0, hold; else go to cash.
Multi-asset: SPY, TLT, GLD. Equal weight among assets with positive trend.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers = ['SPY', 'TLT', 'GLD']
data = yf.download(tickers, start="2005-01-01",
                    auto_adjust=True)['Close']
if isinstance(data.columns, pd.MultiIndex):
    data = data.droplevel(0, axis=1)

mp = data.resample('ME').last()
mr = mp.pct_change().dropna()
sig = mp.pct_change(12).shift(1)
n = len(tickers)

tf = pd.Series(dtype=float)
bh = pd.Series(dtype=float)
for date in mr.index[13:]:
    tf_r = sum(mr.loc[date, tk]/n for tk in tickers
               if sig.loc[date, tk] > 0)
    bh_r = mr.loc[date, tickers].mean()
    tf[date] = tf_r
    bh[date] = bh_r

for name, r in [("Buy & Hold (equal wt)", bh), ("Trend Following (12m)", tf)]:
    ann_r = r.mean() * 12
    ann_v = r.std() * np.sqrt(12)
    sharpe = ann_r / ann_v
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"{name:25s} Ret={ann_r:.1%} Vol={ann_v:.1%} "
          f"Sharpe={sharpe:.2f} MaxDD={mdd:.1%}")

fig, ax = plt.subplots(figsize=(10, 6))
cum_bh = (1 + bh).cumprod()
cum_tf = (1 + tf).cumprod()
ax.plot(cum_bh.index, cum_bh, color='gray', label='Buy & Hold (equal wt SPY/TLT/GLD)')
ax.plot(cum_tf.index, cum_tf, color='steelblue', label='Trend Following (12m signal)')
ax.set_yscale('log')
ax.set_ylabel('Growth of $1 (log)')
ax.set_title('Trend Following: 3-asset, monthly rebalancing')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_ch06_trend.png', dpi=150)
plt.show()
