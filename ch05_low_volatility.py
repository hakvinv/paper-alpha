"""
Chapter 5: Low Volatility Anomaly
Baker, Bradley & Wurgler, 'Benchmarks as Limits to Arbitrage: Understanding
the Low-Volatility Anomaly', Financial Analysts Journal, 2011.

SPLV (S&P 500 Low Volatility ETF) vs SPHB (S&P 500 High Beta ETF) vs SPY.
BAB = SPLV - SPHB (betting against beta).
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers = ['SPY', 'SPLV', 'SPHB']
data = yf.download(tickers, start="2011-06-01", auto_adjust=True)['Close']
if isinstance(data.columns, pd.MultiIndex):
    data = data.droplevel(0, axis=1)

ret = data.pct_change().dropna()
bab = ret['SPLV'] - ret['SPHB']

for name, r in [("SPY", ret['SPY']), ("Low Volatility (SPLV)", ret['SPLV']),
                ("High Beta (SPHB)", ret['SPHB']), ("BAB (SPLV-SPHB)", bab)]:
    ann_r = r.mean() * 252
    ann_v = r.std() * np.sqrt(252)
    sharpe = ann_r / ann_v
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"{name:25s} Ret={ann_r:.1%} Vol={ann_v:.1%} "
          f"Sharpe={sharpe:.2f} MaxDD={mdd:.1%}")

fig, ax = plt.subplots(figsize=(10, 6))
for tk, label in [('SPY','SPY'), ('SPLV','Low Volatility (SPLV)'), ('SPHB','High Beta (SPHB)')]:
    cum = (1 + ret[tk]).cumprod()
    ax.plot(cum.index, cum, label=label)
ax.set_yscale('log')
ax.set_ylabel('Growth of $1 (log)')
ax.set_title('Low Volatility Anomaly: SPLV vs SPHB vs SPY')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_ch05_low_vol.png', dpi=150)
plt.show()
