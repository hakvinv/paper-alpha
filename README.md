# Paper Alpha

> 15 academic trading strategies. Fully implemented. No paywalls.
>
> ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
> ![License](https://img.shields.io/badge/License-MIT-green?style=flat)
> ![Data](https://img.shields.io/badge/Data-Yahoo%20Finance-purple?style=flat)
> ![Strategies](https://img.shields.io/badge/Strategies-15-orange?style=flat)
>
> Each chapter takes a landmark academic paper and turns it into clean, runnable Python — from volatility targeting to machine learning alphas. Free data, no Bloomberg terminal required.
>
> ---
>
> ## Quickstart
>
> ```bash
> git clone https://github.com/hakvinv/paper-alpha.git
> cd paper-alpha
> pip install -r requirements.txt
> python ch01_volatility_targeting.py
> ```
>
> ---
>
> ## Strategies
>
> | # | File | Strategy | Paper |
> |---|------|----------|-------|
> | 01 | `ch01_volatility_targeting.py` | EWMA Vol Targeting | Moreira & Muir (2017) |
> | 02 | `ch02_momentum.py` | 12-1 Sector Momentum | Jegadeesh & Titman (1993) |
> | 03 | `ch03_value.py` | Value vs Growth via ETFs | Fama & French (1992) |
> | 04 | `ch04_carry.py` | FX Carry Trade AUD/JPY | Lustig et al. (2011) |
> | 05 | `ch05_low_volatility.py` | Low-Vol Anomaly SPLV/SPHB | Baker et al. (2011) |
> | 06 | `ch06_trend_following.py` | 3-Asset Trend Following | Moskowitz et al. (2012) |
> | 07 | `ch07_quality.py` | Quality Minus Junk | Asness et al. (2019) |
> | 08 | `ch08_betting_against_beta.py` | Betting Against Beta | Frazzini & Pedersen (2014) |
> | 09 | `ch09_reversal.py` | Weekly Sector Reversal | Lehmann (1990) |
> | 10 | `ch10_pairs_trading.py` | Z-Score Pairs: KO/PEP, XOM/CVX | Gatev et al. (2006) |
> | 11 | `ch11_risk_parity.py` | Inverse-Vol Risk Parity | Qian (2005) |
> | 12 | `ch12_factor_timing.py` | Value Spread Analysis | Asness (2016) |
> | 13 | `ch13_ml_alpha.py` | XGBoost on ETF Features | Gu, Kelly & Xiu (2020) |
> | 14 | `ch14_volatility_risk_premium.py` | VRP + Conservative Short-Vol | Ilmanen (2011) |
> | 15 | `ch15_combined.py` | Trend + Risk Parity Combined | Hamill et al. (2016) |
>
> ---
>
> ## Dependencies
>
> | Package | Purpose |
> |---------|---------|
> | `yfinance` | Free market data |
> | `pandas` | Data manipulation |
> | `numpy` | Numerical computing |
> | `matplotlib` | Plotting |
> | `scipy` | Statistics |
> | `xgboost` | Ch. 13 (falls back to sklearn) |
>
> ---
>
> ## Notes
>
> All scripts pull free data from Yahoo Finance — no paid subscriptions, no API keys. If your numbers differ slightly from the book, Yahoo Finance may have revised historical data since publication (dividend adjustments, splits).
>
> ---
>
> *Built by [Hakvin Vosteen](https://github.com/hakvinv)*
