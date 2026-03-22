"""
Chapter 8: Betting Against Beta
Frazzini & Pedersen, 'Betting Against Beta', Journal of Financial Economics,
2014. 3,500+ citations.

Sort sector ETFs by trailing 12-month beta to SPY. Long low-beta sectors,
short high-beta sectors. The long-only low-beta portfolio (Sharpe 0.85)
outperforms the long-short BAB factor (which lost money 2006-2025).
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sector_tickers = ['XLK','XLF','XLE','XLV','XLI','XLY','XLP','XLU','XLB']
all_tickers = sector_tickers + ['SPY']
data = yf.download(all_tickers, start="2005-01-01", auto_adjust=True)['Close']
if isinstance(data.columns, pd.MultiIndex):
    data = data.droplevel(0, axis=1)

mp = data.resample('ME').last()
mr = mp.pct_change().dropna()

port_lo = pd.Series(dtype=float)  # long-only low beta
port_ls = pd.Series(dtype=float)  # long-short BAB

for i in range(12, len(mr)):
    date = mr.index[i]
    window = mr.iloc[i-12:i]
    spy_ret = window['SPY']
    betas = {}
    for tk in sector_tickers:
        if tk in window.columns:
            cov = np.cov(window[tk].values, spy_ret.values)
            betas[tk] = cov[0,1] / cov[1,1] if cov[1,1] > 0 else 1.0

    if len(betas) < 6:
        continue

    sorted_b = sorted(betas.items(), key=lambda x: x[1])
    low3 = [t[0] for t in sorted_b[:3]]
    high3 = [t[0] for t in sorted_b[-3:]]

    port_lo[date] = mr.loc[date, low3].mean()
    port_ls[date] = mr.loc[date, low3].mean() - mr.loc[date, high3].mean()

spy_ret_monthly = mr['SPY'].loc[port_lo.index]

for name, r in [("SPY", spy_ret_monthly), ("Low Beta (long only)", port_lo),
                ("BAB L/S (low-high)", port_ls)]:
    ann_r = r.mean() * 12
    ann_v = r.std() * np.sqrt(12)
    sharpe = ann_r / ann_v
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"{name:25s} Ret={ann_r:.1%} Vol={ann_v:.1%} "
          f"Sharpe={sharpe:.2f} MaxDD={mdd:.1%}")

fig, ax = plt.subplots(figsize=(10, 6))
for name, r, style, color in [("SPY", spy_ret_monthly, '-', 'gray'),
                               ("Low Beta Sectors", port_lo, '-', 'steelblue'),
                               ("BAB L/S", port_ls, '--', 'red')]:
    cum = (1 + r).cumprod()
    ax.plot(cum.index, cum, style, color=color, label=name)
ax.set_yscale('log')
ax.set_ylabel('Growth of $1 (log)')
ax.set_title('Betting Against Beta: Sector ETFs sorted by trailing beta')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_ch08_bab.png', dpi=150)
plt.show()
