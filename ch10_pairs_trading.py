"""
Chapter 10: Pairs Trading
Gatev, Goetzmann & Rouwenhorst, 'Pairs Trading: Performance of a
Relative-Value Arbitrage Rule', Review of Financial Studies, 2006.
2,200+ citations.

z_t = (P_A/P_B - mu_60) / sigma_60
Entry: |z| > 2.0.  Exit: |z| < 0.5.  Stop-loss: |z| > 4.0.
Three classic pairs: KO/PEP, XOM/CVX, GS/MS.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pairs_backtest(tk_a, tk_b, start="2005-01-01"):
    """Run pairs trading backtest on two tickers."""
    data = yf.download([tk_a, tk_b], start=start, auto_adjust=True)['Close']
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)
    data = data.dropna()

    ratio = data[tk_a] / data[tk_b]
    mu = ratio.rolling(60).mean()
    sigma = ratio.rolling(60).std()
    z = (ratio - mu) / sigma

    pos = pd.Series(0.0, index=data.index)
    ret_a = data[tk_a].pct_change()
    ret_b = data[tk_b].pct_change()

    in_trade = 0  # 0=flat, 1=long spread, -1=short spread
    trades = 0

    for i in range(61, len(data)):
        if in_trade == 0:
            if z.iloc[i] < -2.0:
                in_trade = 1   # long A, short B
                trades += 1
            elif z.iloc[i] > 2.0:
                in_trade = -1  # short A, long B
                trades += 1
        else:
            if abs(z.iloc[i]) < 0.5:
                in_trade = 0
            elif abs(z.iloc[i]) > 4.0:
                in_trade = 0  # stop-loss
        pos.iloc[i] = in_trade

    strat_ret = pos.shift(1) * (ret_a - ret_b)
    strat_ret = strat_ret.dropna()

    ann_r = strat_ret.mean() * 252
    ann_v = strat_ret.std() * np.sqrt(252)
    sharpe = ann_r / ann_v if ann_v > 0 else 0
    cum = (1 + strat_ret).cumprod()
    mdd = (cum / cum.cummax() - 1).min()

    return {'return': ann_r, 'vol': ann_v, 'sharpe': sharpe,
            'mdd': mdd, 'trades': trades, 'z': z, 'cum': cum}

if __name__ == "__main__":
    pairs = [('KO','PEP'), ('XOM','CVX'), ('GS','MS')]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for tk_a, tk_b in pairs:
        print(f"\n{tk_a} / {tk_b}:")
        r = pairs_backtest(tk_a, tk_b)
        print(f"  Return={r['return']:.1%}  Vol={r['vol']:.1%}  "
              f"Sharpe={r['sharpe']:.2f}  MaxDD={r['mdd']:.1%}  "
              f"Trades={r['trades']}")

        axes[0].plot(r['z'].index, r['z'], label=f'{tk_a}/{tk_b}', linewidth=0.5)
        axes[1].plot(r['cum'].index, r['cum'], label=f'{tk_a} / {tk_b}')

    axes[0].axhline(2, color='green', linestyle='--', alpha=0.3)
    axes[0].axhline(-2, color='green', linestyle='--', alpha=0.3)
    axes[0].set_ylabel(f'Z-score')
    axes[0].set_title('Pairs Trading: Z-score of price ratio')
    axes[0].legend(fontsize=8)

    axes[1].axhline(1, color='gray', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Growth of $1')
    axes[1].set_title('Cumulative PnL')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('fig_ch10_pairs.png', dpi=150)
    plt.show()
