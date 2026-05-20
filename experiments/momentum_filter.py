"""
experiments/momentum_filter.py
Phase 6: Momentum Persistence Gate + GJR-GARCH Vol Spec

Tests two targeted improvements motivated by the quiet bear analysis:

IMPROVEMENT 1: Momentum Persistence Gate
  The quiet bear analysis confirmed that the ensemble confidently
  misclassifies low-volatility within-bear periods as neutral/bull,
  increasing equity weight despite negative momentum. The fix: gate any
  equity weight increase on the SIGN of rolling short-term momentum.

  Rule: if the ensemble is within an ongoing bear episode (any bear signal
  in the last MOM_WINDOW days) AND short-term momentum is still negative,
  block any increase in equity weight above its current level.

IMPROVEMENT 2: GJR-GARCH Volatility Specification
  Phase 2 showed GJR-GARCH produces the smallest Sharpe gap (-0.177 vs
  -0.258 baseline) by maintaining elevated conditional volatility during
  quiet within-bear periods via the leverage effect.

EXPERIMENT DESIGN:
  Four conditions on S&P 500, Windows 1 and 3:
    A: EWMA+VIX, no gate     (standard baseline)
    B: EWMA+VIX, gate        (gate only)
    C: GJR-GARCH, no gate    (vol spec only)
    D: GJR-GARCH, gate       (both improvements)
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg

MOM_GATE_WINDOW  = 20
MOM_GATE_PTHRESH = 0.3


def apply_momentum_gate(w_final: np.ndarray,
                         p_bear:  np.ndarray,
                         muann:   np.ndarray) -> np.ndarray:
    """
    Gate equity weight increases during ongoing bear episodes with
    negative momentum. Called from run_experiment via gate_fn param.
    """
    N = len(w_final)
    w_gated    = w_final.copy()
    bear_sig   = p_bear > MOM_GATE_PTHRESH
    bear_ep    = np.zeros(N, dtype=bool)
    mom_neg    = np.zeros(N, dtype=bool)

    for t in range(N):
        s = max(0, t - MOM_GATE_WINDOW + 1)
        bear_ep[t] = bear_sig[s:t+1].any()

    for t in range(MOM_GATE_WINDOW, N):
        mom_neg[t] = float(np.mean(muann[t-MOM_GATE_WINDOW:t])) < 0

    gate_active = bear_ep & mom_neg
    prev_w = w_final[0]
    for t in range(N):
        if gate_active[t]:
            w_gated[t] = min(w_gated[t], prev_w)
        prev_w = w_gated[t]

    n = gate_active.sum()
    print(f"    Gate active: {n}/{N} days ({n/N*100:.1f}%)")
    return w_gated


def run_phase6(verbose: bool = True) -> pd.DataFrame:
    from pipeline import run_experiment

    print("\n" + "="*70)
    print("PHASE 6: MOMENTUM PERSISTENCE GATE + GJR-GARCH")
    print("="*70)

    test_windows = [w for w in cfg.ROLLING_WINDOWS
                    if w["name"] in ("Window1_2000s", "Window3_2020s")]

    conditions = [
        {"vol": "ewma_vix",  "gate": False, "label": "A_baseline"},
        {"vol": "ewma_vix",  "gate": True,  "label": "B_gate_only"},
        {"vol": "gjr_garch", "gate": False, "label": "C_gjrgarch_only"},
        {"vol": "gjr_garch", "gate": True,  "label": "D_gate_gjrgarch"},
    ]

    all_results = []

    for cond in conditions:
        print(f"\n{'='*55}")
        print(f"Condition {cond['label']}: vol={cond['vol']}, "
              f"gate={'ON' if cond['gate'] else 'OFF'}")
        print(f"{'='*55}")

        gate_fn = apply_momentum_gate if cond["gate"] else None

        for window in test_windows:
            print(f"\n  Window: {window['name']}")
            result = run_experiment(
                ticker       = cfg.PRIMARY_ASSET,
                window       = window,
                vol_spec     = cond["vol"],
                label_scheme = cfg.BASELINE_LABEL_SCHEME,
                verbose      = verbose,
                gate_fn      = gate_fn,
                save_result  = False,  # don't overwrite Phase 1-5 pkl files
            )
            if result is not None:
                all_results.append({
                    "condition":       cond["label"],
                    "vol_spec":        cond["vol"],
                    "gate":            cond["gate"],
                    "window":          window["name"],
                    "sharpe":          result.sharpe,
                    "baseline_sharpe": result.baseline_sharpe,
                    "delta_sharpe":    result.sharpe - result.baseline_sharpe,
                    "max_dd":          result.max_dd,
                    "baseline_max_dd": result.baseline_max_dd,
                    "delta_max_dd":    result.max_dd - result.baseline_max_dd,
                    "cagr":            result.cagr,
                })

    if not all_results:
        print("No results produced.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    print("\n" + "="*70)
    print("TABLE: MOMENTUM GATE + GJR-GARCH RESULTS")
    print("="*70)
    print(f"{'Condition':<22} {'Window':<22} {'Sharpe':>7} "
          f"{'BL Sharpe':>9} {'Δ Sharpe':>9} {'MaxDD':>8}")
    print("-"*70)
    for _, r in df.iterrows():
        print(f"{r['condition']:<22} {r['window']:<22} "
              f"{r['sharpe']:>7.3f} {r['baseline_sharpe']:>9.3f} "
              f"{r['delta_sharpe']:>+9.3f} {r['max_dd']:>8.2%}")

    cond_a = df[df["condition"] == "A_baseline"].set_index("window")
    print(f"\n  Improvement vs condition A:")
    for label in ["B_gate_only", "C_gjrgarch_only", "D_gate_gjrgarch"]:
        sub = df[df["condition"] == label].set_index("window")
        for w in sub.index:
            if w in cond_a.index:
                gap_a = cond_a.loc[w, "delta_sharpe"]
                gap_n = sub.loc[w, "delta_sharpe"]
                print(f"    {label} | {w}: {gap_a:+.3f} → {gap_n:+.3f} "
                      f"(Δ gap: {gap_n-gap_a:+.3f})")

    out = os.path.join(cfg.RESULTS_DIR, "phase6_momentum_gate.csv")
    df.to_csv(out, index=False)
    print(f"\nResults saved: {out}")
    return df


if __name__ == "__main__":
    run_phase6()
