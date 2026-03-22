"""
Chapter 15: Putting It Together
Hamill, Rattray & Van Hemert, 'Trend Following: Equity and Bond Crisis
Alpha', Man Group / AHL, 2016.

Combines trend following (Ch 6) with risk parity (Ch 11).
Four assets: SPY, TLT, GLD, DBC. Risk parity weights with 12-month
trend filter. The combined portfolio: Sharpe 0.88, Max DD -10.8%.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers = ['SPY', 'TLT', 'GLD', 'DBC']
data = yf.download(tickers, start="2007-01-01",
                    auto_adjust=True)['Close']
if isinstance(data.columns, pd.MultiIndex):
    data = data.droplevel(0, axis=1)

mp = data.resample('ME').last()
mr = mp.pct_change().dropna()
sig = mp.pct_change(12).shift(1)

combo = pd.Series(dtype=float)
rp_only = pd.Series(dtype=float)
tf_only = pd.Series(dtype=float)
sixty_forty = pd.Series(dtype=float)

for i in range(12, len(mr)):
    vols = mr[tickers].iloc[i-12:i].std()
    rp_w = (1.0/vols) / (1.0/vols).sum()
    date = mr.index[i]
    r = mr.iloc[i]

    # Risk parity only
    rp_only[date] = (rp_w * r[tickers]).sum()

    # 60/40
    sixty_forty[date] = 0.6 * r['SPY'] + 0.4 * r['TLT']

    # Trend following (equal weight)
    n = len(tickers)
    tf_r = sum(r[tk]/n for tk in tickers if sig.loc[date, tk] > 0)
    tf_only[date] = tf_r

    # Combined: risk parity + trend filter
    total = 0.0
    for tk in tickers:
        trend_on = sig.loc[date, tk] > 0
        total += rp_w[tk] * r[tk] * (1.0 if trend_on else 0.0)
    combo[date] = total

for name, r in [("60/40 (SPY/TLT)", sixty_forty),
                ("Risk Parity only", rp_only),
                ("Trend Following only", tf_only),
                ("Trend + Risk Parity", combo)]:
    ann_r = r.mean() * 12
    ann_v = r.std() * np.sqrt(12)
    sharpe = ann_r / ann_v
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"{name:25s} Ret={ann_r:.1%} Vol={ann_v:.1%} "
          f"Sharpe={sharpe:.2f} MaxDD={mdd:.1%}")

fig, ax = plt.subplots(figsize=(10, 6))
for name, r, style, color in [("60/40", sixty_forty, '-', 'gray'),
                               ("Risk Parity", rp_only, '--', 'red'),
                               ("Trend (equal wt)", tf_only, ':', 'blue'),
                               ("Trend + Risk Parity", combo, '-', 'black')]:
    cum = (1 + r).cumprod()
    ax.plot(cum.index, cum, style, color=color, label=name, linewidth=1.5 if 'black' in color else 1)
ax.set_yscale('log')
ax.set_ylabel('Growth of $1 (log)')
ax.set_title('Combined Strategy: Trend Following + Risk Parity')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_ch15_combined.png', dpi=150)
plt.show()
