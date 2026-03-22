"""
Chapter 13: Machine Learning Alpha
Gu, Kelly & Xiu, 'Empirical Asset Pricing via Machine Learning',
Review of Financial Studies, 2020. 3,000+ citations.

The paper tested every major ML method on 94 firm characteristics.
Neural networks achieved out-of-sample R^2 of ~0.4% per month.

This chapter provides a minimal XGBoost example for experimentation.
The full strategy requires CRSP/Compustat ($10K-50K/year) and $5M+ capital.
"""
import numpy as np
import pandas as pd
import yfinance as yf

def ml_alpha_demo():
    """
    Minimal ML alpha example using XGBoost on sector ETF features.
    This is for learning purposes only -- expect 52-55% accuracy.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("pip install xgboost to run this demo")
        return

    tickers = ['XLK','XLF','XLE','XLV','XLI','XLY','XLP','XLU','XLB','SPY']
    data = yf.download(tickers, start="2010-01-01", auto_adjust=True)['Close']
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)

    # Features: lagged returns at various horizons
    ret = data.pct_change()
    features = pd.DataFrame(index=data.index)
    for tk in tickers[:-1]:  # exclude SPY (target)
        features[f'{tk}_1m'] = ret[tk].rolling(21).mean()
        features[f'{tk}_3m'] = ret[tk].rolling(63).mean()
        features[f'{tk}_vol'] = ret[tk].rolling(21).std()

    # Target: SPY direction next month
    target = (ret['SPY'].rolling(21).mean().shift(-21) > 0).astype(int)

    df = features.join(target.rename('target')).dropna()
    X = df.drop('target', axis=1)
    y = df['target']

    # Train/test split: first 80% train, last 20% test
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                          use_label_encoder=False, eval_metric='logloss',
                          verbosity=0)
    model.fit(X_train, y_train)

    acc_train = (model.predict(X_train) == y_train).mean()
    acc_test = (model.predict(X_test) == y_test).mean()

    print(f"ML Alpha Demo (XGBoost on sector ETF features)")
    print(f"=" * 50)
    print(f"Train accuracy: {acc_train:.1%}")
    print(f"Test accuracy:  {acc_test:.1%}")
    print(f"\nExpected: 52-55% accuracy. This is not tradeable alpha at retail")
    print(f"scale. The paper's result required 94 features, 10,000+ stocks,")
    print(f"and V100 GPU compute time.")

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\nTop 5 features:")
    for feat, val in imp.head(5).items():
        print(f"  {feat}: {val:.3f}")

if __name__ == "__main__":
    ml_alpha_demo()
