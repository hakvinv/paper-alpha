"""
Chapter 9: Short-Term Reversal
Lehmann, 'Fads, Martingales, and Market Efficiency', QJE, 1990.
Lo & MacKinlay, 'When Are Contrarian Profits Due to Stock Market
Overreaction?', Review of Financial Studies, 1990.

Over 1-4 week horizons, assets that went down bounce back, and assets
that went up pull back. Sector ETF implementation with weekly rebalancing.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers = ['XLK','XLF','XLE','XLV','XLI','XLY','XLP','XLU','XLB']
data = yf.download(tickers, start="2004-01-01", auto_adjust=True)['Close']
if isinstance(data.columns, pd.MultiIndex):
    data = data.droplevel(0, axis=1)

wp = data.resample('W-FRI').last()
wr = wp.pct_change().dropna()

port_long = pd.Series(dtype=float)  # long last week's losers
port_ls = pd.Series(dtype=float)    # long losers, short winners
bench = pd.Series(dtype=float)

for i in range(1, len(wr)):
    date = wr.index[i]
    prev = wr.iloc[i-1].dropna().sort_values()
    if len(prev) < 6:
        continue
    losers = prev.index[:3]  # bottom 3
    winners = prev.index[-3:]  # top 3
    port_long[date] = wr.loc[date, losers].mean()
    port_ls[date] = wr.loc[date, losers].mean() - wr.loc[date, winners].mean()
    bench[date] = wr.loc[date].mean()

for name, r in [("Equal Weight Sectors", bench), ("Reversal Long-Only", port_long),
                ("Reversal L/S", port_ls)]:
    ann_r = r.mean() * 52
    ann_v = r.std() * np.sqrt(52)
    sharpe = ann_r / ann_v
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"{name:25s} Ret={ann_r:.1%} Vol={ann_v:.1%} "
          f"Sharpe={sharpe:.2f} MaxDD={mdd:.1%}")

fig, ax = plt.subplots(figsize=(10, 6))
for name, r, style, color in [("Equal Weight", bench, '-', 'gray'),
                               ("Reversal Long (last week losers)", port_long, '-', 'steelblue'),
                               ("Reversal L/S", port_ls, '--', 'red')]:
    cum = (1 + r).cumprod()
    ax.plot(cum.index, cum, style, color=color, label=name)
ax.set_yscale('log')
ax.set_ylabel('Growth of $1 (log)')
ax.set_title('Short-Term Reversal: Sector ETFs, weekly rebalancing')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_ch09_reversal.png', dpi=150)
plt.show()
