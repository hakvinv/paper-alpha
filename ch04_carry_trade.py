"""
Chapter 4: Carry Trade
Lustig, Roussanov & Verdelhan, 'Common Risk Factors in Currency Markets',
Review of Financial Studies, 2011. 2,800+ citations.

High-interest-rate currencies outperform low-interest-rate currencies.
ETF proxy: FXA (AUD) vs FXY (JPY).
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CurrencyShares ETFs: FXA = AUD, FXY = JPY
data = yf.download(['FXA', 'FXY'], start="2007-06-01",
                    auto_adjust=True)['Close']
if isinstance(data.columns, pd.MultiIndex):
    data = data.droplevel(0, axis=1)

r_aud = data['FXA'].pct_change().dropna()
r_jpy = data['FXY'].pct_change().dropna()
idx = r_aud.index.intersection(r_jpy.index)

carry = r_aud[idx] - r_jpy[idx]

# Note: ETFs capture exchange rate movement only.
# The actual interest rate differential (2-3% historically)
# adds to the carry return but isn't reflected in ETF prices.

for name, r in [("Carry (AUD-JPY)", carry),
                ("Equal weight FX", (r_aud[idx] + r_jpy[idx])/2)]:
    ann_r = r.mean() * 252
    ann_v = r.std() * np.sqrt(252)
    sharpe = ann_r / ann_v
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"{name:20s} Ret={ann_r:.1%} Vol={ann_v:.1%} "
          f"Sharpe={sharpe:.2f} MaxDD={mdd:.1%}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
cum_carry = (1 + carry).cumprod()
cum_eq = (1 + (r_aud[idx]+r_jpy[idx])/2).cumprod()
ax.plot(cum_carry.index, cum_carry, color='steelblue', label='Carry (long AUD / short JPY)')
ax.plot(cum_eq.index, cum_eq, color='gray', label='Equal weight FX')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
ax.set_ylabel('Growth of $1')
ax.set_title('AUD/JPY Carry Trade via CurrencyShares ETFs')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_ch04_carry.png', dpi=150)
plt.show()
