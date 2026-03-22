"""
Chapter 14: Volatility Risk Premium
Ilmanen, 'Expected Returns: An Investor's Guide to Harvesting Market
Rewards', Wiley, 2011. Chapter 15.

Implied volatility (VIX) is systematically higher than realized volatility
by about 4 percentage points. This is the volatility risk premium (VRP).

This chapter has no backtest because options strategies require
broker-specific execution. Instead, we measure the VRP directly.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# VIX (implied vol) and SPY (to compute realized vol)
spy = yf.download("SPY", start="2000-01-01", auto_adjust=True)["Close"]
if isinstance(spy, pd.DataFrame):
    spy = spy.iloc[:, 0]
vix = yf.download("^VIX", start="2000-01-01", auto_adjust=True)["Close"]
if isinstance(vix, pd.DataFrame):
    vix = vix.iloc[:, 0]

ret = spy.pct_change().dropna()
realized_vol = ret.rolling(21).std() * np.sqrt(252) * 100  # annualized %

idx = realized_vol.index.intersection(vix.index)
rv = realized_vol[idx]
iv = vix[idx]
vrp = iv - rv

print("Volatility Risk Premium Analysis")
print("=" * 50)
print(f"Mean VIX (implied):     {iv.mean():.1f}%")
print(f"Mean realized vol:      {rv.mean():.1f}%")
print(f"Mean VRP (IV - RV):     {vrp.mean():.1f}%")
print(f"VRP positive {(vrp > 0).mean():.0%} of the time")
print(f"\nThis {vrp.mean():.1f}% gap is the insurance premium options sellers collect.")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(iv.index, iv.rolling(63).mean(), label='VIX (implied)', color='red', linewidth=0.8)
ax.plot(rv.index, rv.rolling(63).mean(), label='Realized vol (21d)', color='blue', linewidth=0.8)
ax.fill_between(vrp.index, 0, vrp.rolling(63).mean(), alpha=0.2, color='green', label='VRP')
ax.set_ylabel('Volatility (%)')
ax.set_title('Volatility Risk Premium: VIX vs Realized Volatility')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_ch14_vrp.png', dpi=150)
plt.show()
