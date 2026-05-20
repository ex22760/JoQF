"""
experiments/phase8_analysis.py
Phase 8: Confidence Threshold Robustness + Transaction Cost / Turnover Analysis

PART A: Confidence Threshold Robustness
Tests whether the quiet bear confident misclassification mechanism survives
across confidence thresholds ct in [0.50, 0.55, 0.60, 0.65, 0.70].

For each threshold:
  - Recompute ensemble weights using that threshold
  - Rerun quiet bear analysis on drawdown-threshold bear days
  - Report weight differential and forward DD cost

All computation from saved daily_ensemble_proba — no retraining needed.

PART B: Transaction Cost and Turnover Analysis
Reports:
  - Number of rebalances per year
  - Mean weight traded per rebalance
  - Total annualised turnover (sum of |Δw| per year)
  - TC drag in basis points (already embedded in backtest)
  - Cost-adjusted Sharpe at c=2bps, 5bps, 10bps
  - Comparison ensemble vs baseline turnover

Output:
  results/phase8_threshold_robustness.csv
  results/phase8_turnover.csv
  figures/fig8_threshold_robustness.png
  figures/fig8_turnover_comparison.png
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (RESULTS_DIR, FIGURES_DIR, ANNUALIZATION,
                    ALPHA_BEAR, ALPHA_BULL, BEAR_FLOOR,
                    CONFLICT_SCALE, TC, BAND_MULT, BAND_ROLL, WARMUP)

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(ticker="^GSPC", label_scheme="nber"):
    results = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
            try:
                obj = pickle.load(f)
                if (hasattr(obj, "daily_ensemble_proba") and
                    obj.daily_ensemble_proba is not None and
                    getattr(obj, "ticker", "") == ticker and
                    getattr(obj, "label_scheme", "") == label_scheme):
                    results.append(obj)
            except Exception:
                pass
    return results


def recompute_weights_with_threshold(r, threshold: float) -> np.ndarray:
    """
    Recompute ensemble equity weights using a different confidence threshold.
    Uses saved daily_ensemble_proba and daily_baseline_weights.
    Returns array of recomputed ensemble weights.
    """
    proba    = np.asarray(r.daily_ensemble_proba)   # (N, 3)
    w_hjb    = np.asarray(r.daily_baseline_weights) # (N,) baseline weight
    sigma    = np.asarray(r.daily_sigma)

    p_bear = proba[:, 0]
    p_bull = proba[:, 2]
    conf   = proba.max(axis=1)

    # Equal-weight fallback when confidence below threshold
    low_conf = conf < threshold
    proba_adj = proba.copy()
    proba_adj[low_conf] = 1.0 / 3.0
    p_bear_adj = proba_adj[:, 0]
    p_bull_adj = proba_adj[:, 2]
    conf_adj   = proba_adj.max(axis=1)

    # Regime-conditional weight
    w_regime = np.clip(
        w_hjb * (1 + ALPHA_BULL * p_bull_adj - ALPHA_BEAR * p_bear_adj),
        0, 1)

    # Confidence-weighted blend
    w_blend = np.clip(conf_adj * w_regime + (1 - conf_adj) * w_hjb, 0, 1)

    return w_blend


def quiet_bear_test_weights(bear_mask, sigma, w_ens, w_base, returns, window=20):
    """Quick quiet bear test given weight arrays."""
    bear_mask = np.asarray(bear_mask, dtype=bool)
    if bear_mask.sum() < 5:
        return None
    sigma_arr  = np.asarray(sigma)
    sigma_med  = np.median(sigma_arr[bear_mask])
    quiet_bear = bear_mask & (sigma_arr <= sigma_med)
    if quiet_bear.sum() < 3:
        return None
    w_diff = (np.asarray(w_ens) - np.asarray(w_base))[quiet_bear]
    t_stat, p_val = stats.ttest_1samp(w_diff, 0)

    ret_arr = np.asarray(returns)
    dd_base, dd_ens = [], []
    for i in np.where(quiet_bear)[0]:
        end = min(i + window, len(ret_arr))
        cb = np.cumprod(1 + np.asarray(w_base)[i:end] * ret_arr[i:end])
        ce = np.cumprod(1 + np.asarray(w_ens)[i:end]  * ret_arr[i:end])
        if len(cb) > 1:
            dd_base.append(float(cb.min() - 1))
            dd_ens.append(float(ce.min() - 1))

    return {
        "n_quiet":  int(quiet_bear.sum()),
        "w_diff":   float(w_diff.mean()),
        "p_value":  float(p_val),
        "delta_dd": float(np.mean(dd_ens) - np.mean(dd_base)) if dd_base else np.nan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART A: Confidence Threshold Robustness
# ══════════════════════════════════════════════════════════════════════════════

def run_threshold_robustness(results_nber, results_dd):
    print("\n" + "="*70)
    print("PART A: CONFIDENCE THRESHOLD ROBUSTNESS")
    print("="*70)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    rows = []

    # Use drawdown results for primary test (richer bear days)
    test_results = results_dd if results_dd else results_nber
    defn = "drawdown" if results_dd else "nber"

    print(f"  Using {defn} labels, {len(test_results)} windows")
    print(f"\n  {'Threshold':>10} {'Window':<22} {'N quiet':>7} "
          f"{'W Diff':>8} {'p-val':>7} {'ΔDD':>8} {'Consistent':>11}")
    print("  " + "-"*70)

    for r in test_results:
        w_base = np.asarray(r.daily_baseline_weights)
        sigma  = np.asarray(r.daily_sigma)
        ret    = np.asarray(r.daily_returns)

        # Drawdown bear mask
        price_idx   = np.cumprod(1 + ret)
        running_max = np.maximum.accumulate(price_idx)
        drawdown    = (price_idx - running_max) / running_max
        dd_bear     = drawdown < -0.15

        if dd_bear.sum() < 5:
            continue

        for thresh in thresholds:
            w_ens_thresh = recompute_weights_with_threshold(r, thresh)
            res = quiet_bear_test_weights(dd_bear, sigma,
                                          w_ens_thresh, w_base, ret)
            if res:
                consistent = "✓" if res["w_diff"] < 0 else "✗"
                print(f"  {thresh:>10.2f} {r.window_name:<22} "
                      f"{res['n_quiet']:>7} {res['w_diff']:>+8.4f} "
                      f"{res['p_value']:>7.4f} {res['delta_dd']:>+8.4f} "
                      f"{consistent:>11}")
                rows.append({
                    "threshold":   thresh,
                    "window":      r.window_name,
                    "label":       defn,
                    "n_quiet":     res["n_quiet"],
                    "w_diff":      res["w_diff"],
                    "p_value":     res["p_value"],
                    "delta_dd":    res["delta_dd"],
                    "consistent":  res["w_diff"] < 0,
                })

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n  Summary by threshold:")
        for thresh in thresholds:
            sub = df[df["threshold"] == thresh]
            wins = sub["consistent"].sum()
            print(f"    ct={thresh:.2f}: {wins}/{len(sub)} consistent, "
                  f"mean w_diff={sub['w_diff'].mean():+.4f}, "
                  f"mean ΔDD={sub['delta_dd'].mean():+.4f}")

        # Figure: threshold vs mean weight differential
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        for wname in df["window"].unique():
            sub = df[df["window"] == wname]
            ax1.plot(sub["threshold"], sub["w_diff"], "o-",
                     label=wname, linewidth=2, markersize=6)
        ax1.axhline(0, color="black", linestyle="--", linewidth=1)
        ax1.axvline(0.55, color="grey", linestyle=":", linewidth=1,
                    label="Baseline threshold")
        ax1.set_xlabel("Confidence threshold $c_t$", fontsize=11)
        ax1.set_ylabel("Mean W. Diff. (ensemble $-$ baseline)", fontsize=11)
        ax1.set_title("Quiet bear weight differential\nacross confidence thresholds",
                      fontsize=10)
        ax1.legend(fontsize=9)
        ax1.set_xlim(0.48, 0.72)

        for wname in df["window"].unique():
            sub = df[df["window"] == wname]
            ax2.plot(sub["threshold"], sub["delta_dd"] * 100, "s-",
                     label=wname, linewidth=2, markersize=6)
        ax2.axhline(0, color="black", linestyle="--", linewidth=1)
        ax2.axvline(0.55, color="grey", linestyle=":", linewidth=1,
                    label="Baseline threshold")
        ax2.set_xlabel("Confidence threshold $c_t$", fontsize=11)
        ax2.set_ylabel("Mean $\\Delta$DD (bp)", fontsize=11)
        ax2.set_title("20-day forward drawdown differential\nacross confidence thresholds",
                      fontsize=10)
        ax2.legend(fontsize=9)
        ax2.set_xlim(0.48, 0.72)

        fig.suptitle("Quiet bear mechanism robustness to confidence threshold\n"
                     "S\\&P 500, drawdown-threshold bear definition",
                     fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "fig8_threshold_robustness.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: fig8_threshold_robustness.png")

        out = os.path.join(RESULTS_DIR, "phase8_threshold_robustness.csv")
        df.to_csv(out, index=False)
        print(f"  Saved: {out}")
        return df

    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# PART B: Transaction Cost and Turnover Analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_turnover_stats(weights: np.ndarray, n_days: int) -> dict:
    """Compute turnover statistics from a weight series."""
    diffs     = np.abs(np.diff(weights))
    n_rebal   = int((diffs > 1e-6).sum())
    total_to  = float(diffs.sum())
    ann_to    = total_to * ANNUALIZATION / n_days
    mean_trade = float(diffs[diffs > 1e-6].mean()) if (diffs > 1e-6).any() else 0.0
    rebal_pa  = n_rebal * ANNUALIZATION / n_days
    return {
        "n_rebal":    n_rebal,
        "rebal_pa":   rebal_pa,
        "total_to":   total_to,
        "ann_to":     ann_to,
        "mean_trade": mean_trade,
    }


def adjusted_sharpe(returns: np.ndarray, tc_bps: float,
                    weights: np.ndarray) -> float:
    """Recompute Sharpe with a different TC assumption."""
    diffs    = np.abs(np.diff(weights))
    tc_daily = np.zeros(len(returns))
    n        = min(len(diffs), len(returns))
    tc_daily[:n] = diffs[:n] * tc_bps * 1e-4
    adj_ret  = returns - tc_daily
    if adj_ret.std() < 1e-10:
        return np.nan
    return float(adj_ret.mean() / adj_ret.std() * np.sqrt(ANNUALIZATION))


def run_turnover_analysis(results_nber):
    print("\n" + "="*70)
    print("PART B: TRANSACTION COST AND TURNOVER ANALYSIS")
    print("="*70)

    rows = []
    tc_rows = []

    for r in results_nber:
        w_ens  = np.asarray(r.daily_weights)
        w_base = np.asarray(r.daily_baseline_weights)
        ret    = np.asarray(r.daily_returns)
        n_days = len(w_ens)
        wname  = r.window_name

        ens_to  = compute_turnover_stats(w_ens,  n_days)
        base_to = compute_turnover_stats(w_base, n_days)

        print(f"\n  {wname}:")
        print(f"    {'Metric':<30} {'Ensemble':>12} {'Baseline':>12}")
        print(f"    {'-'*55}")
        print(f"    {'Rebalances per year':<30} "
              f"{ens_to['rebal_pa']:>12.1f} {base_to['rebal_pa']:>12.1f}")
        print(f"    {'Ann. turnover (weight units)':<30} "
              f"{ens_to['ann_to']:>12.4f} {base_to['ann_to']:>12.4f}")
        print(f"    {'Mean trade size':<30} "
              f"{ens_to['mean_trade']:>12.4f} {base_to['mean_trade']:>12.4f}")
        print(f"    {'TC drag (bps, embedded)':<30} "
              f"{getattr(r, 'tc_bps', np.nan):>12.2f} "
              f"{getattr(r, 'baseline_tc_bps', np.nan):>12.2f}")

        rows.append({
            "window":        wname,
            "strategy":      "ensemble",
            "rebal_pa":      ens_to["rebal_pa"],
            "ann_turnover":  ens_to["ann_to"],
            "mean_trade":    ens_to["mean_trade"],
            "sharpe":        r.sharpe,
        })
        rows.append({
            "window":        wname,
            "strategy":      "baseline",
            "rebal_pa":      base_to["rebal_pa"],
            "ann_turnover":  base_to["ann_to"],
            "mean_trade":    base_to["mean_trade"],
            "sharpe":        r.baseline_sharpe,
        })

        # TC sensitivity
        print(f"\n    TC sensitivity (Sharpe at different cost assumptions):")
        print(f"    {'TC (bps)':>10} {'Ens Sharpe':>12} {'BL Sharpe':>12} "
              f"{'Δ Sharpe':>10}")
        for tc in [1, 2, 5, 10, 20]:
            ens_ret  = w_ens[:-1]  * ret[1:] + (1-w_ens[:-1])  * 0
            base_ret = w_base[:-1] * ret[1:] + (1-w_base[:-1]) * 0
            s_ens  = adjusted_sharpe(ens_ret,  tc, w_ens)
            s_base = adjusted_sharpe(base_ret, tc, w_base)
            print(f"    {tc:>10} {s_ens:>12.3f} {s_base:>12.3f} "
                  f"{s_ens-s_base:>+10.3f}")
            tc_rows.append({
                "window": wname, "tc_bps": tc,
                "ens_sharpe": s_ens, "base_sharpe": s_base,
                "delta_sharpe": s_ens - s_base,
            })

    if rows:
        # Turnover figure
        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        windows = df["window"].unique()
        x = np.arange(len(windows))
        w = 0.35

        for ax, metric, label in [
            (axes[0], "rebal_pa",     "Rebalances per year"),
            (axes[1], "ann_turnover", "Annualised turnover (weight units)"),
            (axes[2], "mean_trade",   "Mean trade size"),
        ]:
            ens_vals  = [df[(df["window"]==wn) & (df["strategy"]=="ensemble")][metric].values[0]
                         for wn in windows]
            base_vals = [df[(df["window"]==wn) & (df["strategy"]=="baseline")][metric].values[0]
                         for wn in windows]
            ax.bar(x - w/2, ens_vals,  w, label="Ensemble",  color="#1f77b4", alpha=0.8)
            ax.bar(x + w/2, base_vals, w, label="Baseline",  color="#ff7f0e", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([wn.replace("_", "\n") for wn in windows], fontsize=8)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(label, fontsize=9)
            ax.legend(fontsize=8)

        fig.suptitle("Transaction cost and turnover: ensemble vs HJB baseline\n"
                     "S\\&P 500, NBER labels, all three windows", fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "fig8_turnover_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: fig8_turnover_comparison.png")

        out = os.path.join(RESULTS_DIR, "phase8_turnover.csv")
        df.to_csv(out, index=False)
        print(f"  Saved: {out}")

        df_tc = pd.DataFrame(tc_rows)
        out2  = os.path.join(RESULTS_DIR, "phase8_tc_sensitivity.csv")
        df_tc.to_csv(out2, index=False)
        print(f"  Saved: {out2}")

        return df, df_tc

    return pd.DataFrame(), pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_phase8():
    print("\n" + "#"*70)
    print("PHASE 8: CONFIDENCE THRESHOLD ROBUSTNESS + TC/TURNOVER ANALYSIS")
    print("#"*70)

    results_nber = load_results(ticker="^GSPC", label_scheme="nber")
    results_dd   = load_results(ticker="^GSPC", label_scheme="drawdown")

    print(f"Loaded: {len(results_nber)} NBER, {len(results_dd)} drawdown results")

    if not results_nber and not results_dd:
        print("ERROR: No results with daily_ensemble_proba found.")
        print("Rerun Phase 1 with updated pipeline.py")
        return

    # Filter to Windows with bear days
    results_nber_bear = [r for r in results_nber
                         if r.window_name in ("Window1_2000s", "Window3_2020s")]
    results_dd_all    = results_dd  # all 3 windows have drawdown bear days

    # Part A
    df_thresh = run_threshold_robustness(results_nber_bear, results_dd_all)

    # Part B
    df_to, df_tc = run_turnover_analysis(results_nber)

    print("\n" + "="*70)
    print("PHASE 8 COMPLETE")
    if len(df_thresh) > 0:
        total = len(df_thresh)
        consistent = df_thresh["consistent"].sum()
        print(f"Threshold robustness: {consistent}/{total} conditions consistent")
    print("="*70)

    return df_thresh, df_to, df_tc


if __name__ == "__main__":
    run_phase8()
