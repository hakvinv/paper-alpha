"""
Chapter 13 — Machine Learning Alpha
Paper: Gu, Kelly & Xiu, "Empirical Asset Pricing via Machine Learning",
       Review of Financial Studies, 2020.

Simplified retail version: XGBoost on ETF features from Yahoo Finance.
This is explicitly a toy model — the real paper uses 94 firm characteristics,
CRSP/Compustat data, and V100 GPUs. This script shows the concept.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download, print_stats, plot_equity

# ── Data: sector ETFs + features ──────────────────────────────────────────────
TICKERS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
START = "2005-01-01"

prices = download(TICKERS, start=START)
ret = prices.pct_change()

# ── Feature engineering ───────────────────────────────────────────────────────
# For each ETF: momentum (12-1), short-term reversal (1w), volatility (1m)
features_list = []
targets_list = []

for tk in TICKERS:
    p = prices[tk].dropna()
    r = ret[tk].dropna()

    feat = pd.DataFrame(index=r.index)
    feat['mom_12_1'] = p.pct_change(252).shift(21)   # 12-month mom, skip 1 month
    feat['reversal_1w'] = r.rolling(5).sum().shift(1)  # 1-week return
    feat['vol_1m'] = r.rolling(21).std().shift(1)       # 1-month realized vol
    feat['vol_3m'] = r.rolling(63).std().shift(1)       # 3-month realized vol
    feat['mom_1m'] = p.pct_change(21).shift(1)           # 1-month momentum

    # Target: next month return (21 trading days)
    target = r.rolling(21).sum().shift(-21)

    feat['ticker'] = tk
    feat['target'] = target
    features_list.append(feat)

data = pd.concat(features_list).dropna()
feature_cols = ['mom_12_1', 'reversal_1w', 'vol_1m', 'vol_3m', 'mom_1m']

# ── Train/test split ──────────────────────────────────────────────────────────
split_date = "2015-01-01"
train = data.loc[:split_date]
test = data.loc[split_date:]

print(f"Train: {len(train)} obs, Test: {len(test)} obs")

# ── Model: try XGBoost if available, else fall back to sklearn ────────────────
try:
    from xgboost import XGBClassifier
    model_name = "XGBoost"
    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        use_label_encoder=False, eval_metric='logloss'
    )
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    model_name = "GradientBoosting (sklearn)"
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42
    )

# Binary target: up or down
train_y = (train['target'] > 0).astype(int)
test_y = (test['target'] > 0).astype(int)

model.fit(train[feature_cols], train_y)

# ── Predictions and accuracy ──────────────────────────────────────────────────
train_pred = model.predict(train[feature_cols])
test_pred = model.predict(test[feature_cols])

train_acc = (train_pred == train_y).mean()
test_acc = (test_pred == test_y).mean()

print(f"\n{model_name} accuracy:")
print(f"  Train: {train_acc:.1%}")
print(f"  Test:  {test_acc:.1%}")
print(f"  (Random baseline: 50%)")

# ── Feature importance ────────────────────────────────────────────────────────
if hasattr(model, 'feature_importances_'):
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nFeature importance:")
    for feat, val in imp.items():
        print(f"  {feat:15s} {val:.3f}")

# ── Simple signal portfolio ───────────────────────────────────────────────────
test_data = test.copy()
test_data['pred'] = test_pred

# Monthly rebalancing: long predicted-up sectors, equal weight
monthly_test = test_data.groupby(test_data.index).apply(
    lambda x: x[x['pred'] == 1]['target'].mean() if (x['pred'] == 1).any() else 0
)

print(f"\nNote: 52-55% accuracy on direction is typical for this approach.")
print("The paper's neural network achieved ~0.4% monthly R² — tiny but meaningful")
print("when diversified across 10,000+ stocks. With 9 ETFs, it's noise.")

# ── Plot feature importance ───────────────────────────────────────────────────
if hasattr(model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(8, 4))
    imp.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f"Ch.13 — {model_name} Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("ch13_ml_importance.png", dpi=150)
    plt.show()
