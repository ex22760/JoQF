"""
experiments/vol_robustness.py
Phase 2: Volatility model robustness.
Runs all 5 vol specs across all 3 rolling windows.
Framed as robustness test, not "GARCH is better."
"""

import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (ROLLING_WINDOWS, VOL_SPECS, BASELINE_LABEL_SCHEME,
                    BASELINE_VOL_SPEC, PRIMARY_ASSET, RESULTS_DIR)
from pipeline import run_experiment


def run_vol_robustness(ticker:       str  = PRIMARY_ASSET,
                        label_scheme: str  = BASELINE_LABEL_SCHEME,
                        verbose:      bool = True) -> list:
    """
    Run all vol specs × all windows. Returns list of ExperimentResults.
    Total runs: 5 specs × 3 windows = 15 experiments.
    """
    results = []
    total   = len(VOL_SPECS) * len(ROLLING_WINDOWS)
    done    = 0

    for vol_spec in VOL_SPECS:
        for window in ROLLING_WINDOWS:
            done += 1
            print(f"\n[{done}/{total}] vol={vol_spec}, window={window['name']}")
            result = run_experiment(
                ticker       = ticker,
                window       = window,
                vol_spec     = vol_spec,
                label_scheme = label_scheme,
                verbose      = verbose,
            )
            results.append(result)

    return results


def print_vol_robustness_table(results: list):
    """Print Table 3: performance by volatility specification."""
    print("\n" + "="*90)
    print("TABLE 3: ROBUSTNESS TO VOLATILITY SPECIFICATION")
    print("(NBER labels, all 3 rolling windows)")
    print("="*90)
    print(f"{'Vol Spec':<12} {'Window':<20} {'Sharpe':>7} {'BL Sharpe':>9} "
          f"{'Δ Sharpe':>8} {'MaxDD':>7} {'MacroF1':>8} {'BearRec':>8}")
    print("-"*90)

    for vol_spec in ["ewma", "vix", "ewma_vix", "garch", "gjr_garch"]:
        spec_results = [r for r in results if r.vol_spec == vol_spec]
        for r in spec_results:
            marker = " ◄ baseline" if vol_spec == "ewma_vix" else ""
            print(f"{vol_spec:<12} {r.window_name:<20} "
                  f"{r.sharpe:>7.3f} {r.baseline_sharpe:>9.3f} "
                  f"{r.sharpe - r.baseline_sharpe:>+8.3f} "
                  f"{r.max_dd:>7.2%} "
                  f"{r.ensemble_macro_f1:>8.3f} "
                  f"{r.ensemble_bear_recall:>8.3f}{marker}")
        print()

    # Summary: mean across windows per spec
    print("\nMean across all 3 windows:")
    print(f"{'Vol Spec':<12} {'Mean Δ Sharpe':>14} {'Mean MaxDD':>11} "
          f"{'Mean F1':>8} {'Consistent?':>12}")
    print("-"*55)
    for vol_spec in ["ewma", "vix", "ewma_vix", "garch", "gjr_garch"]:
        spec_r = [r for r in results if r.vol_spec == vol_spec]
        if not spec_r:
            continue
        deltas  = [r.sharpe - r.baseline_sharpe for r in spec_r]
        max_dds = [r.max_dd for r in spec_r]
        f1s     = [r.ensemble_macro_f1 for r in spec_r]
        consistent = "YES" if all(d > 0 for d in deltas) else "NO"
        print(f"{vol_spec:<12} {np.mean(deltas):>+14.3f} "
              f"{np.mean(max_dds):>11.2%} "
              f"{np.mean(f1s):>8.3f} {consistent:>12}")
    print("="*90)


if __name__ == "__main__":
    print("PHASE 2: VOLATILITY MODEL ROBUSTNESS")
    print("Note: GARCH framed as robustness check, not improvement claim")

    results = run_vol_robustness()
    print_vol_robustness_table(results)

    # Save CSV
    rows = []
    for r in results:
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
            "n_bear_days":    r.n_bear_days,
        })
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "vol_robustness.csv")
    df.to_csv(out, index=False)
    print(f"\nResults saved: {out}")
