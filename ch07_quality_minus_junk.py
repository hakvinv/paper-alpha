"""
Chapter 7: Quality Minus Junk
Asness, Frazzini & Pedersen, 'Quality Minus Junk', Review of Accounting
Studies, 2019.

Quality = z(Profitability) + z(Growth) + z(Safety) + z(Payout)
Retail implementation: QUAL (iShares MSCI USA Quality Factor ETF) vs SPY.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = yf.download(['SPY', 'QUAL'], start="2013-08-01", auto_adjust=True)['Close']
if isinstance(data.columns, pd.MultiIndex):
    data = data.droplevel(0, axis=1)

ret = data.pct_change().dropna()
qmj = ret['QUAL'] - ret['SPY']

for name, r in [("SPY", ret['SPY']), ("Quality (QUAL)", ret['QUAL']),
                ("QMJ (QUAL-SPY)", qmj)]:
    ann_r = r.mean() * 252
    ann_v = r.std() * np.sqrt(252)
    sharpe = ann_r / ann_v
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"{name:20s} Ret={ann_r:.1%} Vol={ann_v:.1%} "
          f"Sharpe={sharpe:.2f} MaxDD={mdd:.1%}")

fig, ax = plt.subplots(figsize=(10, 6))
for tk, label in [('SPY','SPY'), ('QUAL','Quality (QUAL)')]:
    cum = (1 + ret[tk]).cumprod()
    ax.plot(cum.index, cum, label=label)
ax.set_yscale('log')
ax.set_ylabel('Growth of $1 (log)')
ax.set_title('Quality (QUAL) vs SPY')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_ch07_quality.png', dpi=150)
plt.show()
