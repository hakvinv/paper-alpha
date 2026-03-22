# Paper Alpha — 15 Academic Strategies, Exposed

Complete, runnable Python code for every chapter in **Paper Alpha**.

## Quickstart

```bash
git clone https://github.com/hakvinv/paper-alpha.git
cd paper-alpha
pip install -r requirements.txt
python ch01_volatility_targeting.py
```

## Files

| File | Chapter | Strategy |
|------|---------|----------|
| `utils.py` | — | Shared utilities: data download, stats, plotting, costs |
| `ch01_volatility_targeting.py` | 1 | EWMA vol targeting (Moreira & Muir 2017) |
| `ch02_momentum.py` | 2 | 12-1 sector momentum (Jegadeesh & Titman 1993) |
| `ch03_value.py` | 3 | Value vs growth via ETFs (Fama & French 1992) |
| `ch04_carry.py` | 4 | FX carry trade AUD/JPY (Lustig et al. 2011) |
| `ch05_low_volatility.py` | 5 | Low-vol anomaly SPLV/SPHB (Baker et al. 2011) |
| `ch06_trend_following.py` | 6 | 3-asset trend following (Moskowitz et al. 2012) |
| `ch07_quality.py` | 7 | Quality minus junk QUAL (Asness et al. 2019) |
| `ch08_betting_against_beta.py` | 8 | BAB via sector betas (Frazzini & Pedersen 2014) |
| `ch09_reversal.py` | 9 | Weekly sector reversal (Lehmann 1990) |
| `ch10_pairs_trading.py` | 10 | Z-score pairs: KO/PEP, XOM/CVX, GS/MS (Gatev et al. 2006) |
| `ch11_risk_parity.py` | 11 | Inverse-vol risk parity (Qian 2005) |
| `ch12_factor_timing.py` | 12 | Value spread analysis (Asness 2016) |
| `ch13_ml_alpha.py` | 13 | XGBoost on ETF features (Gu, Kelly & Xiu 2020) |
| `ch14_volatility_risk_premium.py` | 14 | VRP analysis + conservative short-vol (Ilmanen 2011) |
| `ch15_combined.py` | 15 | Trend + risk parity combined (Hamill et al. 2016) |

## Data

All scripts use free data from Yahoo Finance via `yfinance`. No paid subscriptions required.

If your numbers differ from the book, Yahoo Finance has updated its historical data since publication (dividend adjustments, stock splits).

## Dependencies

- **yfinance** — market data
- **pandas** — data manipulation
- **numpy** — numerical computing
- **matplotlib** — plotting
- **scipy** — statistics
- **xgboost** — Ch.13 only (falls back to sklearn if not installed)

## Author

Hakvin Vosteen — vosteen@uni-bremen.de
