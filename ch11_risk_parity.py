"""
Chapter 11: Risk Parity
Qian, 'Risk Parity Portfolios: Efficient Portfolios Through True
Diversification', Panagora Asset Management, 2005.

w_i = (1/sigma_i) / sum(1/sigma_j)
Four assets: SPY, TLT, GLD, DBC. Monthly rebalancing.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers = ['SPY', 'TLT', 'GLD', 'DBC']
data = yf.download(tickers, start="2007-01-01", auto_adjust=True)['Close']
if isinstance(data.columns, pd.MultiIndex):
    data = data.droplevel(0, axis=1)

mp = data.resample('ME').last()
mr = mp.pct_change().dropna()

rp = pd.Series(dtype=float)
ew = pd.Series(dtype=float)
sixty_forty = pd.Series(dtype=float)

for i in range(12, len(mr)):
    vols = mr[tickers].iloc[i-12:i].std()
    rp_w = (1.0/vols) / (1.0/vols).sum()
    date = mr.index[i]
    r = mr.iloc[i]
    rp[date] = (rp_w * r[tickers]).sum()
    ew[date] = r[tickers].mean()
    sixty_forty[date] = 0.6 * r['SPY'] + 0.4 * r['TLT']

for name, r in [("60/40 (SPY/TLT)", sixty_forty),
                ("Equal Weight (4 assets)", ew),
                ("Risk Parity (inv vol)", rp)]:
    ann_r = r.mean() * 12
    ann_v = r.std() * np.sqrt(12)
    sharpe = ann_r / ann_v
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"{name:30s} Ret={ann_r:.1%} Vol={ann_v:.1%} "
          f"Sharpe={sharpe:.2f} MaxDD={mdd:.1%}")

fig, ax = plt.subplots(figsize=(10, 6))
for name, r, color in [("60/40", sixty_forty, 'gray'),
                        ("Equal Weight", ew, 'red'),
                        ("Risk Parity", rp, 'steelblue')]:
    cum = (1 + r).cumprod()
    ax.plot(cum.index, cum, color=color, label=name)
ax.set_yscale('log')
ax.set_ylabel('Growth of $1 (log)')
ax.set_title('Risk Parity vs 60/40 vs Equal Weight (SPY, TLT, GLD, DBC)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_ch11_risk_parity.png', dpi=150)
plt.show()
