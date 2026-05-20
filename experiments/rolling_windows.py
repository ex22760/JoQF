"""
experiments/rolling_windows.py
Phase 1: Rolling window backtest across 3 windows.
Baseline specification: EWMA+VIX vol, NBER labels.
Outputs results to results/ and prints summary table.
"""

import os, sys, pickle
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (ROLLING_WINDOWS, BASELINE_VOL_SPEC,
                    BASELINE_LABEL_SCHEME, PRIMARY_ASSET, RESULTS_DIR)
from pipeline import run_experiment, ExperimentResult


def run_rolling_windows(ticker:      str  = PRIMARY_ASSET,
                         vol_spec:    str  = BASELINE_VOL_SPEC,
                         label_scheme: str = BASELINE_LABEL_SCHEME,
                         verbose:     bool = True) -> list:
    """
    Run experiment across all three rolling windows.
    Returns list of ExperimentResult objects.
    """
    results = []
    for window in ROLLING_WINDOWS:
        result = run_experiment(
            ticker       = ticker,
            window       = window,
            vol_spec     = vol_spec,
            label_scheme = label_scheme,
            verbose      = verbose,
        )
        results.append(result)
    return results


def print_rolling_summary(results: list):
    """Print Table 1: rolling window performance summary."""
    print("\n" + "="*80)
    print("TABLE 1: ROLLING WINDOW RESULTS (Baseline spec: EWMA+VIX, NBER labels)")
    print("="*80)
    print(f"{'Window':<20} {'Sharpe':>7} {'BL Sharpe':>9} {'Δ Sharpe':>8} "
          f"{'MaxDD':>7} {'BL MaxDD':>8} {'CAGR':>7} {'Bear Wt':>8} {'BL Wt':>7}")
    print("-"*80)
    for r in results:
        dsharpe = r.sharpe - r.baseline_sharpe
        print(f"{r.window_name:<20} "
              f"{r.sharpe:>7.3f} {r.baseline_sharpe:>9.3f} "
              f"{dsharpe:>+8.3f} "
              f"{r.max_dd:>7.2%} {r.baseline_max_dd:>8.2%} "
              f"{r.cagr:>7.2%} "
              f"{r.bear_mean_weight:>8.3f} {r.baseline_bear_weight:>7.3f}")
    print("="*80)

    # Bear day statistics
    print("\nBear day statistics:")
    for r in results:
        pct = r.n_bear_days / r.n_test_days * 100
        print(f"  {r.window_name}: {r.n_bear_days} bear days "
              f"({pct:.1f}% of {r.n_test_days} test days)")


if __name__ == "__main__":
    print("PHASE 1: ROLLING WINDOW BACKTEST")
    print("Ticker: S&P 500 (^GSPC)")
    print("Vol spec: EWMA+VIX (baseline)")
    print("Labels: NBER (baseline)")

    results = run_rolling_windows()
    print_rolling_summary(results)

    # Save aggregate CSV
    rows = []
    for r in results:
        rows.append({
            "ticker":       r.ticker,
            "window":       r.window_name,
            "vol_spec":     r.vol_spec,
            "label_scheme": r.label_scheme,
            "sharpe":       r.sharpe,
            "baseline_sharpe": r.baseline_sharpe,
            "delta_sharpe": r.sharpe - r.baseline_sharpe,
            "max_dd":       r.max_dd,
            "baseline_max_dd": r.baseline_max_dd,
            "cagr":         r.cagr,
            "macro_f1":     r.ensemble_macro_f1,
            "bear_recall":  r.ensemble_bear_recall,
            "bear_mean_wt": r.bear_mean_weight,
            "n_bear_days":  r.n_bear_days,
            "n_test_days":  r.n_test_days,
        })
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "rolling_windows_baseline.csv")
    df.to_csv(out, index=False)
    print(f"\nResults saved: {out}")
