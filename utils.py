"""
Paper Alpha — Shared utilities for all 15 chapters.
Hakvin Vosteen | github.com/hakvinv/paper-alpha

Common helpers: data download, backtest stats, plotting, transaction costs.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ── Data helpers ──────────────────────────────────────────────────────────────

def download(tickers, start="2000-01-01", end=None):
    """Download adjusted close prices from Yahoo Finance. Returns DataFrame."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]] if isinstance(tickers, str) else data
        if "Close" in prices.columns:
            prices = prices["Close"]
    # Flatten if single ticker
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers if isinstance(tickers, str) else tickers[0])
    return prices.dropna(how="all")


# ── Backtest statistics ──────────────────────────────────────────────────────

def backtest_stats(returns, name="Strategy", freq="daily"):
    """Compute annualized Return, Vol, Sharpe, Max DD, Win Rate from a return Series."""
    r = returns.dropna()
    if freq == "daily":
        factor = 252
    elif freq == "monthly":
        factor = 12
    elif freq == "weekly":
        factor = 52
    else:
        factor = 252

    ann_ret = r.mean() * factor
    ann_vol = r.std() * np.sqrt(factor)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0.0
    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    win_rate = (r > 0).mean()

    return {
        "Name": name,
        "Return": f"{ann_ret:.1%}",
        "Volatility": f"{ann_vol:.1%}",
        "Sharpe": f"{sharpe:.2f}",
        "Max DD": f"{max_dd:.1%}",
        "Win Rate": f"{win_rate:.1%}",
    }


def print_stats(returns_dict, freq="daily"):
    """Print a formatted stats table for multiple return series."""
    rows = []
    for name, ret in returns_dict.items():
        rows.append(backtest_stats(ret, name=name, freq=freq))
    df = pd.DataFrame(rows).set_index("Name")
    print("\n" + df.to_string() + "\n")
    return df


# ── Transaction cost model ───────────────────────────────────────────────────

def apply_costs(returns, weights, commission_per_dollar=0.0001, slippage_bps=3):
    """
    Apply realistic transaction costs.
    - commission_per_dollar: IB commission as fraction of trade value (~$0.005/share ≈ 1 bp)
    - slippage_bps: slippage in basis points per trade
    """
    turnover = weights.diff().abs().sum(axis=1) if isinstance(weights, pd.DataFrame) \
        else weights.diff().abs()
    cost_per_rebal = turnover * (commission_per_dollar + slippage_bps / 10_000)
    return returns - cost_per_rebal


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_equity(returns_dict, title="Growth of $1", log=True, figsize=(10, 5)):
    """Plot cumulative growth curves for multiple return series."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, ret in returns_dict.items():
        cum = (1 + ret.dropna()).cumprod()
        ax.plot(cum.index, cum.values, label=name)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log)" if log else "Growth of $1")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_drawdown(returns_dict, title="Drawdown", figsize=(10, 3)):
    """Plot drawdown curves."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, ret in returns_dict.items():
        cum = (1 + ret.dropna()).cumprod()
        dd = cum / cum.cummax() - 1
        ax.fill_between(dd.index, dd.values, alpha=0.3, label=name)
    ax.set_ylabel("Drawdown")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Quick demo
    prices = download("SPY", start="2020-01-01")
    ret = prices["SPY"].pct_change().dropna()
    print_stats({"SPY Buy & Hold": ret})
