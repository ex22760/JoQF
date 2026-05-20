"""
run_all.py
Master script — runs all experiments in order.
Results saved to results/ after each phase so you can
interrupt and resume without restarting from scratch.

Estimated total runtime:
  Phase 1 (rolling windows, baseline):   ~2 hours
  Phase 2 (vol robustness, 5 specs × 3): ~8 hours
  Phase 3 (cross-market, 5 assets × 3):  ~6 hours
  Phase 5 (significance):                ~5 minutes
  Total:                                 ~16 hours
Run overnight or in tmux/screen session.

Usage:
  python run_all.py              # run everything
  python run_all.py --phase 1   # run specific phase only
  python run_all.py --phase 2
"""

import os, sys, argparse, time

# Ensure the JF root directory is always on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import RESULTS_DIR


def run_phase_1():
    print("\n" + "#"*70)
    print("PHASE 1: ROLLING WINDOW BACKTEST (baseline specification)")
    print("#"*70)
    from experiments.rolling_windows import run_rolling_windows, print_rolling_summary
    results = run_rolling_windows()
    print_rolling_summary(results)
    return results


def run_phase_1b():
    print("\n" + "#"*70)
    print("PHASE 1b: LABEL ROBUSTNESS (NBER vs drawdown-threshold)")
    print("#"*70)
    import pandas as pd
    from config import ROLLING_WINDOWS, LABEL_SCHEMES, PRIMARY_ASSET
    from pipeline import run_experiment

    all_results = []
    for label_scheme in LABEL_SCHEMES:
        for window in ROLLING_WINDOWS:
            r = run_experiment(
                ticker       = PRIMARY_ASSET,
                window       = window,
                vol_spec     = "ewma_vix",
                label_scheme = label_scheme,
            )
            all_results.append(r)

    # Print comparison
    print("\nTABLE 2: ROBUSTNESS TO REGIME DEFINITION")
    print(f"{'Label':<12} {'Window':<20} {'Sharpe':>7} {'BL Sharpe':>9} "
          f"{'Δ Sharpe':>8} {'MaxDD':>7} {'F1':>6}")
    print("-"*70)
    for r in all_results:
        print(f"{r.label_scheme:<12} {r.window_name:<20} "
              f"{r.sharpe:>7.3f} {r.baseline_sharpe:>9.3f} "
              f"{r.sharpe-r.baseline_sharpe:>+8.3f} "
              f"{r.max_dd:>7.2%} {r.ensemble_macro_f1:>6.3f}")

    import pandas as pd
    df  = pd.DataFrame([{
        "label_scheme": r.label_scheme,
        "window":       r.window_name,
        "sharpe":       r.sharpe,
        "delta_sharpe": r.sharpe - r.baseline_sharpe,
        "max_dd":       r.max_dd,
        "macro_f1":     r.ensemble_macro_f1,
    } for r in all_results])
    df.to_csv(os.path.join(RESULTS_DIR, "label_robustness.csv"), index=False)
    return all_results


def run_phase_2():
    print("\n" + "#"*70)
    print("PHASE 2: VOLATILITY MODEL ROBUSTNESS")
    print("Framing: stress test of framework, not performance improvement claim")
    print("#"*70)
    from experiments.vol_robustness import (run_vol_robustness,
                                              print_vol_robustness_table)
    results = run_vol_robustness()
    print_vol_robustness_table(results)
    return results


def run_phase_3():
    print("\n" + "#"*70)
    print("PHASE 3: CROSS-MARKET GENERALISATION")
    print("GMM/HMM retrained per asset; supervised architecture fixed")
    print("#"*70)
    from experiments.cross_market import (run_cross_market,
                                           print_cross_market_table)
    results = run_cross_market()
    print_cross_market_table(results)
    return results


def run_phase_4():
    print("\n" + "#"*70)
    print("PHASE 4: MULTI-ASSET ALLOCATION")
    print("Bear: dynamic min-var over TLT/GLD/SHY | Neutral/Bull: HJB")
    print("#"*70)
    from experiments.multi_asset_alloc import run_phase4
    return run_phase4()


def run_phase_5():
    print("\n" + "#"*70)
    print("PHASE 5: STATISTICAL SIGNIFICANCE")
    print("#"*70)
    from experiments.significance import run_significance_analysis
    return run_significance_analysis()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all JF paper experiments")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (0=all, 1, 2, 3, 5)")
    args = parser.parse_args()

    t_total = time.time()

    if args.phase == 0 or args.phase == 1:
        run_phase_1()
        run_phase_1b()

    if args.phase == 0 or args.phase == 2:
        run_phase_2()

    if args.phase == 0 or args.phase == 3:
        run_phase_3()

    if args.phase == 0 or args.phase == 4:
        run_phase_4()

    if args.phase == 6:
        from experiments.momentum_filter import run_phase6
        run_phase6()

    if args.phase == 0 or args.phase == 5:
        run_phase_5()

    if args.phase == 0 or args.phase == 7:
        from experiments.phase7_analysis import run_phase7
        run_phase7()

    if args.phase == 0 or args.phase == 8:
        from experiments.phase8_analysis import run_phase8
        run_phase8()

    print(f"\n{'='*70}")
    print(f"ALL PHASES COMPLETE. Total runtime: "
          f"{(time.time()-t_total)/3600:.1f} hours")
    print(f"Results in: {RESULTS_DIR}")
    print(f"{'='*70}")
