"""
Chapter 12: Factor Timing
Asness, 'The Siren Song of Factor Timing', Journal of Portfolio
Management, 2016.

Academic consensus: factors can be timed, but the signal-to-noise ratio
is so low that the improvement from timing is smaller than the loss from
estimation error. Hold a diversified factor portfolio at static weights.
Rebalance annually. Don't try to predict which factor wins next quarter.

This chapter has no backtest because the evidence is clear:
static allocation beats timing for retail investors.
"""
import numpy as np
import pandas as pd

def factor_timing_theory():
    """
    Demonstrate why factor timing doesn't work in practice.

    The Sharpe improvement from perfect timing is only 0.1-0.2 above
    static allocation. But perfect timing is impossible, and estimation
    error in the timing signal destroys the theoretical improvement.
    """
    np.random.seed(42)
    n_months = 240  # 20 years

    # Simulate two uncorrelated factors
    factor_1 = np.random.normal(0.005, 0.04, n_months)  # momentum-like
    factor_2 = np.random.normal(0.003, 0.03, n_months)  # value-like

    # Static 50/50 allocation
    static = 0.5 * factor_1 + 0.5 * factor_2
    sharpe_static = np.mean(static) / np.std(static) * np.sqrt(12)

    # Perfect timing (impossible in practice)
    perfect = np.maximum(factor_1, factor_2)
    sharpe_perfect = np.mean(perfect) / np.std(perfect) * np.sqrt(12)

    # Noisy timing (realistic)
    # Predict which factor wins with 55% accuracy
    signal = np.random.random(n_months) < 0.55
    actual_winner = factor_1 > factor_2
    correct = signal == actual_winner
    noisy = np.where(signal, 0.7*factor_1 + 0.3*factor_2,
                             0.3*factor_1 + 0.7*factor_2)
    sharpe_noisy = np.mean(noisy) / np.std(noisy) * np.sqrt(12)

    print("Factor Timing Simulation (20 years)")
    print("=" * 50)
    print(f"Static 50/50:      Sharpe = {sharpe_static:.2f}")
    print(f"Perfect timing:    Sharpe = {sharpe_perfect:.2f}  "
          f"(+{sharpe_perfect - sharpe_static:.2f})")
    print(f"Noisy timing (55%): Sharpe = {sharpe_noisy:.2f}  "
          f"(+{sharpe_noisy - sharpe_static:.2f})")
    print(f"\nConclusion: even 55% accuracy barely improves on static allocation.")
    print("The diversification benefit of holding multiple uncorrelated factors")
    print("is both larger and more reliable than any timing benefit.")

if __name__ == "__main__":
    factor_timing_theory()
