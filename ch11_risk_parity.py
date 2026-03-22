"""
Chapter 11 — Risk Parity
Paper: Qian, "Risk Parity Portfolios: Efficient Portfolios Through True
       Diversification", Panagora Asset Management, 2005.

Inverse-volatility weighting across SPY, TLT, GLD, DBC.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity, plot_drawdown

# ── Parameters ────────────────────────────────────────────────────────────────
TICKERS = ['SPY', 'TLT', 'GLD', 'DBC']
START = "2007-01-01"
VOL_WINDOW = 12  # months for volatility estimate

# ── Data ──────────────────────────────────────────────────────────────────────
prices = download(TICKERS, start=START)
mp = prices.resample('ME').last()
mr = mp.pct_change().dropna()

# ── Build portfolios ──────────────────────────────────────────────────────────
rp_port = pd.Series(dtype=float)
ew_port = pd.Series(dtype=float)
sixty_forty = pd.Series(dtype=float)

for i in range(VOL_WINDOW, len(mr)):
    date = mr.index[i]
    # Inverse-vol weights from trailing 12-month vol
    vols = mr[TICKERS].iloc[i-VOL_WINDOW:i].std()
    rp_w = (1.0 / vols) / (1.0 / vols).sum()

    # Risk parity return
    rp_port[date] = (rp_w * mr.iloc[i][TICKERS]).sum()

    # Equal weight
    ew_port[date] = mr.iloc[i][TICKERS].mean()

    # 60/40 SPY/TLT
    sixty_forty[date] = 0.6 * mr.iloc[i]['SPY'] + 0.4 * mr.iloc[i]['TLT']

# ── Stats ─────────────────────────────────────────────────────────────────────
print_stats({
    "60/40 (SPY/TLT)": sixty_forty,
    "Equal Weight (4 assets)": ew_port,
    "Risk Parity (inv vol)": rp_port,
}, freq="monthly")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plot_equity({
    "60/40": sixty_forty,
    "Equal Weight": ew_port,
    "Risk Parity": rp_port,
}, title="Ch.11 — Risk Parity vs 60/40")
plt.savefig("ch11_risk_parity.png", dpi=150)

plot_drawdown({
    "60/40": sixty_forty,
    "Risk Parity": rp_port,
}, title="Drawdown Comparison")
plt.savefig("ch11_drawdown.png", dpi=150)
plt.show()
