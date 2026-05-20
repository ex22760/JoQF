"""
experiments/cross_market.py
Phase 3: Cross-market generalisation.
Tests the framework on 5 additional assets using:
  - Best vol spec from Phase 2 AND baseline (ewma_vix) for robustness
  - GMM/HMM retrained per asset
  - Supervised architecture fixed (hyperparameters from S&P 500 grid search)
  - Window 3 (2019-2026) as primary, all 3 windows for robustness
"""

import os, sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import (ROLLING_WINDOWS, CROSS_MARKET_ASSETS, BEST_VOL_SPEC,
                    BASELINE_VOL_SPEC, BASELINE_LABEL_SCHEME,
                    RESULTS_DIR)
from pipeline import run_experiment


def run_cross_market(vol_spec:     str  = BASELINE_VOL_SPEC,
                      label_scheme: str  = BASELINE_LABEL_SCHEME,
                      windows:      list = None,
                      verbose:      bool = True) -> list:
    """
    Run experiment across all cross-market assets and windows.
    Returns list of ExperimentResult objects.
    """
    if windows is None:
        windows = ROLLING_WINDOWS   # all 3 by default

    results = []
    assets  = list(CROSS_MARKET_ASSETS.values())
    total   = len(assets) * len(windows)
    done    = 0

    for asset_name, ticker in CROSS_MARKET_ASSETS.items():
        for window in windows:
            done += 1
            print(f"\n[{done}/{total}] asset={asset_name} ({ticker}), "
                  f"window={window['name']}, vol={vol_spec}")
            try:
                result = run_experiment(
                    ticker       = ticker,
                    window       = window,
                    vol_spec     = vol_spec,
                    label_scheme = label_scheme,
                    verbose      = verbose,
                )
                results.append(result)
            except Exception as e:
                print(f"  WARNING: {ticker} failed — {e}")

    return results


def print_cross_market_table(results: list,
                              sp500_results: list = None):
    """Print Table 4: cross-market generalisation results."""
    print("\n" + "="*85)
    print("TABLE 4: CROSS-MARKET GENERALISATION")
    print("="*85)
    print(f"{'Asset':<10} {'Ticker':<8} {'Window':<20} {'Sharpe':>7} "
          f"{'BL Sharpe':>9} {'Δ Sharpe':>8} {'MaxDD':>7} {'F1':>6}")
    print("-"*85)

    ticker_to_name = {v: k for k, v in
                      {**{"^GSPC": "S&P500"}, **CROSS_MARKET_ASSETS}.items()}

    # S&P 500 first (if provided)
    if sp500_results:
        for r in sp500_results:
            print(f"{'S&P500':<10} {r.ticker:<8} {r.window_name:<20} "
                  f"{r.sharpe:>7.3f} {r.baseline_sharpe:>9.3f} "
                  f"{r.sharpe-r.baseline_sharpe:>+8.3f} "
                  f"{r.max_dd:>7.2%} {r.ensemble_macro_f1:>6.3f}")
        print()

    for ticker, asset_name in {v: k for k,v in CROSS_MARKET_ASSETS.items()}.items():
        asset_results = [r for r in results if r.ticker == ticker]
        for r in asset_results:
            print(f"{asset_name:<10} {ticker:<8} {r.window_name:<20} "
                  f"{r.sharpe:>7.3f} {r.baseline_sharpe:>9.3f} "
                  f"{r.sharpe-r.baseline_sharpe:>+8.3f} "
                  f"{r.max_dd:>7.2%} {r.ensemble_macro_f1:>6.3f}")
        print()

    # Summary: fraction of experiments where ensemble outperforms
    all_r     = results + (sp500_results or [])
    n_pos     = sum(1 for r in all_r if r.sharpe > r.baseline_sharpe)
    n_total   = len(all_r)
    print(f"\nEnsemble outperforms baseline: {n_pos}/{n_total} experiments "
          f"({n_pos/n_total*100:.0f}%)")
    print("="*85)


if __name__ == "__main__":
    print("PHASE 3: CROSS-MARKET GENERALISATION")
    print("Architecture fixed from S&P 500 grid search")
    print("GMM/HMM retrained per asset")
    print("Running both baseline (ewma_vix) and best (gjr_garch) vol specs")

    from config import BEST_VOL_SPEC

    # Run baseline spec
    print("\n--- Baseline vol spec (ewma_vix) ---")
    results_baseline = run_cross_market(vol_spec=BASELINE_VOL_SPEC)

    # Run best spec from Phase 2
    print("\n--- Best vol spec from Phase 2 (gjr_garch) ---")
    results_best = run_cross_market(vol_spec=BEST_VOL_SPEC)

    print("\n=== BASELINE (ewma_vix) ===")
    print_cross_market_table(results_baseline)

    print("\n=== BEST VOL SPEC (gjr_garch) ===")
    print_cross_market_table(results_best)

    # Save
    rows = []
    for r in results_baseline:
        rows.append({
            "ticker":         r.ticker,
            "window":         r.window_name,
            "vol_spec":       r.vol_spec,
            "label_scheme":   r.label_scheme,
            "sharpe":         r.sharpe,
            "baseline_sharpe": r.baseline_sharpe,
            "delta_sharpe":   r.sharpe - r.baseline_sharpe,
            "max_dd":         r.max_dd,
            "cagr":           r.cagr,
            "macro_f1":       r.ensemble_macro_f1,
            "bear_recall":    r.ensemble_bear_recall,
            "n_test_days":    r.n_test_days,
            "n_bear_days":    r.n_bear_days,
        })
    df  = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "cross_market.csv")
    df.to_csv(out, index=False)
    print(f"\nResults saved: {out}")