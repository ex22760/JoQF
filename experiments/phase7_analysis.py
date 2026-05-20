"""
experiments/phase7_analysis.py
Phase 7: Alternative Bear Definitions + Confidence Analysis

Two components:

PART A: Alternative Bear Definitions
Tests whether the quiet bear confidence collapse mechanism survives across
four independent bear definitions, addressing the NBER label limitation:

  1. NBER labels          — already confirmed, included for comparison
  2. Drawdown-threshold   — bear = peak-to-trough decline > 15%
  3. Volatility-threshold — bear = realised vol > 90th percentile
  4. HMM states           — bear = HMM causal forward-filter bear state
  5. Pagan-Sossounov      — bear = peak-trough rule (already in labels)

For each definition: identify quiet bear days (below-median vol within
bear episodes), compare ensemble vs baseline equity weight, compute
subsequent 20-day forward drawdown differential.

PART B: Confidence Analysis
Makes the mechanism statistically visible:

  1. Entropy analysis      — Shannon entropy on quiet bear vs all bear days
  2. Confidence histograms — confidence score distribution by regime
  3. Calibration analysis  — reliability curves for ensemble probabilities
  4. Quiet bear entropy    — show entropy is systematically higher on quiet bears

All analysis runs on S&P 500 NBER experiments (Windows 1 and 3) using
the daily series saved in pkl files.

Output:
  results/phase7_bear_definitions.csv   — alternative bear definition results
  results/phase7_confidence.csv         — confidence/entropy analysis
  figures/fig7_confidence_histograms.png
  figures/fig7_reliability_curves.png
  figures/fig7_entropy_by_regime.png
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import RESULTS_DIR, FIGURES_DIR, ANNUALIZATION

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results_with_series(ticker="^GSPC", label_scheme="nber"):
    results = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
            try:
                obj = pickle.load(f)
                if (hasattr(obj, "daily_weights") and
                    obj.daily_weights is not None and
                    getattr(obj, "ticker", "") == ticker and
                    getattr(obj, "label_scheme", "") == label_scheme):
                    results.append(obj)
            except Exception:
                pass
    print(f"Loaded {len(results)} {ticker} {label_scheme} results with daily series")
    return results


def shannon_entropy(proba_matrix):
    """Shannon entropy of probability distribution. Shape (N, K)."""
    eps = 1e-10
    p = np.clip(proba_matrix, eps, 1)
    return -np.sum(p * np.log(p), axis=1)


def quiet_bear_test(bear_mask, sigma, w_ens, w_base, returns, window=20):
    """
    Run quiet bear analysis given a bear mask and daily series.
    Returns dict of results or None if insufficient data.
    """
    bear_mask = np.asarray(bear_mask, dtype=bool)
    if bear_mask.sum() < 5:
        return None

    sigma_arr = np.asarray(sigma)
    bear_sigma = sigma_arr[bear_mask]
    sigma_med  = np.median(bear_sigma)
    quiet_bear = bear_mask & (sigma_arr <= sigma_med)
    n_quiet    = quiet_bear.sum()

    if n_quiet < 3:
        return None

    w_diff = (np.asarray(w_ens) - np.asarray(w_base))[quiet_bear]
    t_stat, p_val = stats.ttest_1samp(w_diff, 0)

    # 20-day forward drawdown
    ret_arr = np.asarray(returns)
    dd_base, dd_ens = [], []
    idx = np.where(quiet_bear)[0]
    for i in idx:
        end = min(i + window, len(ret_arr))
        cum_base = np.cumprod(1 + np.asarray(w_base)[i:end] * ret_arr[i:end] +
                              (1 - np.asarray(w_base)[i:end]) * 0.0)
        cum_ens  = np.cumprod(1 + np.asarray(w_ens)[i:end]  * ret_arr[i:end] +
                              (1 - np.asarray(w_ens)[i:end])  * 0.0)
        if len(cum_base) > 1:
            dd_base.append(float(cum_base.min() - 1))
            dd_ens.append(float(cum_ens.min() - 1))

    return {
        "n_bear":    int(bear_mask.sum()),
        "n_quiet":   int(n_quiet),
        "pct_quiet": n_quiet / bear_mask.sum() * 100,
        "w_diff":    float(w_diff.mean()),
        "p_value":   float(p_val),
        "bl_dd":     float(np.mean(dd_base)) if dd_base else np.nan,
        "ens_dd":    float(np.mean(dd_ens))  if dd_ens  else np.nan,
        "delta_dd":  float(np.mean(dd_ens) - np.mean(dd_base)) if dd_base else np.nan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART A: Alternative Bear Definitions
# ══════════════════════════════════════════════════════════════════════════════

def run_alternative_bear_definitions(results):
    print("\n" + "="*70)
    print("PART A: ALTERNATIVE BEAR DEFINITIONS")
    print("="*70)

    rows = []

    for r in results:
        w_ens  = np.asarray(r.daily_weights)
        w_base = np.asarray(r.daily_baseline_weights)
        sigma  = np.asarray(r.daily_sigma)
        labels = np.asarray(r.daily_labels)
        ret    = np.asarray(r.daily_returns)
        window_name = r.window_name

        # Need ensemble probabilities for HMM definition
        # These are not saved in pkl — use labels as proxy for HMM
        # (HMM states are used to create the labels field for nber scheme)

        print(f"\n  {r.ticker} | {window_name}")

        # ── Definition 1: NBER (baseline, already confirmed) ─────────────────
        nber_bear = labels == 0
        res = quiet_bear_test(nber_bear, sigma, w_ens, w_base, ret)
        if res:
            rows.append({"definition": "NBER", "window": window_name, **res})
            print(f"    NBER:         n_bear={res['n_bear']:4d}  n_quiet={res['n_quiet']:4d}  "
                  f"w_diff={res['w_diff']:+.4f}  p={res['p_value']:.4f}  "
                  f"ΔDD={res['delta_dd']:+.4f}")

        # ── Definition 2: Drawdown-threshold (>15% peak-to-trough) ───────────
        price_idx = np.cumprod(1 + ret)
        running_max = np.maximum.accumulate(price_idx)
        drawdown = (price_idx - running_max) / running_max
        dd_bear  = drawdown < -0.15

        res = quiet_bear_test(dd_bear, sigma, w_ens, w_base, ret)
        if res:
            rows.append({"definition": "Drawdown>15%", "window": window_name, **res})
            print(f"    Drawdown>15%: n_bear={res['n_bear']:4d}  n_quiet={res['n_quiet']:4d}  "
                  f"w_diff={res['w_diff']:+.4f}  p={res['p_value']:.4f}  "
                  f"ΔDD={res['delta_dd']:+.4f}")

        # ── Definition 3: Volatility-threshold (>90th percentile) ────────────
        vol_90   = np.percentile(sigma, 90)
        vol_bear = sigma > vol_90

        res = quiet_bear_test(vol_bear, sigma, w_ens, w_base, ret)
        if res:
            rows.append({"definition": "Vol>P90", "window": window_name, **res})
            print(f"    Vol>P90:      n_bear={res['n_bear']:4d}  n_quiet={res['n_quiet']:4d}  "
                  f"w_diff={res['w_diff']:+.4f}  p={res['p_value']:.4f}  "
                  f"ΔDD={res['delta_dd']:+.4f}")

        # ── Definition 4: Pagan-Sossounov (labels == 0 already captures
        #    the NBER bear; PS labels bear as part of the label construction.
        #    Here we reconstruct bear episodes as sustained drawdown > 15%
        #    lasting > 70 trading days — consistent with PS rule) ─────────────
        ps_bear = np.zeros(len(ret), dtype=bool)
        in_episode = False
        peak_i = 0
        for t in range(1, len(price_idx)):
            if price_idx[t] > price_idx[peak_i]:
                if in_episode:
                    # End of bear episode
                    if t - peak_i > 70:
                        ps_bear[peak_i:t] = True
                    in_episode = False
                peak_i = t
            elif (price_idx[t] / price_idx[peak_i] - 1) < -0.15:
                in_episode = True
        if in_episode and len(price_idx) - peak_i > 70:
            ps_bear[peak_i:] = True

        res = quiet_bear_test(ps_bear, sigma, w_ens, w_base, ret)
        if res:
            rows.append({"definition": "Pagan-Sossounov", "window": window_name, **res})
            print(f"    Pagan-Sosson: n_bear={res['n_bear']:4d}  n_quiet={res['n_quiet']:4d}  "
                  f"w_diff={res['w_diff']:+.4f}  p={res['p_value']:.4f}  "
                  f"ΔDD={res['delta_dd']:+.4f}")

    # Summary
    if rows:
        df = pd.DataFrame(rows)
        print("\n" + "="*70)
        print("SUMMARY: QUIET BEAR MECHANISM ACROSS BEAR DEFINITIONS")
        print("="*70)
        for defn in df["definition"].unique():
            sub = df[df["definition"] == defn]
            wins = (sub["w_diff"] < 0).sum()
            dd_wins = (sub["delta_dd"] > 0).sum()
            print(f"  {defn:<20} mean w_diff={sub['w_diff'].mean():+.4f}  "
                  f"wins(ensemble>base)={wins}/{len(sub)}  "
                  f"DD worse={dd_wins}/{len(sub)}")

        out = os.path.join(RESULTS_DIR, "phase7_bear_definitions.csv")
        df.to_csv(out, index=False)
        print(f"\nSaved: {out}")
        return df
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# PART B: Confidence Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_confidence_analysis(results):
    print("\n" + "="*70)
    print("PART B: CONFIDENCE ANALYSIS")
    print("="*70)

    all_conf   = []
    all_labels = []
    all_entropy = []
    all_quiet_bear = []
    all_bear_mask  = []

    conf_rows = []

    for r in results:
        if not hasattr(r, "daily_ensemble_proba") or r.daily_ensemble_proba is None:
            print(f"  WARNING: {r.window_name} has no daily_ensemble_proba — skipping confidence analysis")
            continue

        proba  = np.asarray(r.daily_ensemble_proba)   # (N, 3)
        conf   = proba.max(axis=1)
        labels = np.asarray(r.daily_labels)
        sigma  = np.asarray(r.daily_sigma)
        wname  = r.window_name

        entropy = shannon_entropy(proba)

        # Quiet bear mask
        bear_mask  = labels == 0
        if bear_mask.sum() > 0:
            sigma_med  = np.median(sigma[bear_mask])
            quiet_bear = bear_mask & (sigma <= sigma_med)
        else:
            quiet_bear = np.zeros(len(labels), dtype=bool)

        all_conf.extend(conf.tolist())
        all_labels.extend(labels.tolist())
        all_entropy.extend(entropy.tolist())
        all_quiet_bear.extend(quiet_bear.tolist())
        all_bear_mask.extend(bear_mask.tolist())

        # Confidence stats by regime
        for regime, name in [(0, "bear"), (1, "neutral"), (2, "bull")]:
            mask = labels == regime
            if mask.sum() > 0:
                conf_rows.append({
                    "window":  wname,
                    "regime":  name,
                    "n":       int(mask.sum()),
                    "mean_conf":   float(conf[mask].mean()),
                    "median_conf": float(np.median(conf[mask])),
                    "mean_entropy": float(entropy[mask].mean()),
                })

        # Quiet bear vs non-quiet bear entropy
        if bear_mask.sum() > 0 and quiet_bear.sum() > 0:
            non_quiet_bear = bear_mask & ~quiet_bear
            print(f"\n  {wname}:")
            print(f"    Quiet bear entropy:     {entropy[quiet_bear].mean():.4f} "
                  f"(n={quiet_bear.sum()})")
            if non_quiet_bear.sum() > 0:
                print(f"    Non-quiet bear entropy: {entropy[non_quiet_bear].mean():.4f} "
                      f"(n={non_quiet_bear.sum()})")
            print(f"    Bull entropy:           {entropy[labels==2].mean():.4f}")
            print(f"    Neutral entropy:        {entropy[labels==1].mean():.4f}")
            print(f"    Mean conf (quiet bear): {conf[quiet_bear].mean():.4f}")
            print(f"    Mean conf (all bear):   {conf[bear_mask].mean():.4f}")

    if not conf_rows:
        print("  No ensemble probability series available.")
        print("  Re-run pipeline with daily_ensemble_proba saving enabled.")
        print("  Skipping figures — running entropy proxy from weights instead.")
        _run_weight_based_entropy_proxy(results)
        return pd.DataFrame()

    # ── Confidence histogram by regime ────────────────────────────────────────
    all_conf   = np.array(all_conf)
    all_labels = np.array(all_labels)
    all_entropy = np.array(all_entropy)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    regime_names = ["Bear", "Neutral", "Bull"]
    colours      = ["#d62728", "#7f7f7f", "#2ca02c"]
    for ax, regime_id, name, col in zip(axes, [0, 1, 2], regime_names, colours):
        mask = all_labels == regime_id
        if mask.sum() > 0:
            ax.hist(all_conf[mask], bins=30, color=col, alpha=0.75,
                    edgecolor="white", linewidth=0.5)
            ax.axvline(0.55, color="black", linestyle="--", linewidth=1,
                       label="0.55 threshold")
            ax.set_title(f"{name} days (n={mask.sum()})", fontsize=11)
            ax.set_xlabel("Ensemble confidence score", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(fontsize=8)
    fig.suptitle("Ensemble confidence score distribution by true regime\n"
                 "S\\&P 500, NBER labels, Windows 1 and 3", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_confidence_histograms.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: fig7_confidence_histograms.png")

    # ── Entropy comparison: quiet bear vs non-quiet bear vs bull ──────────────
    quiet_bear_arr = np.array(all_quiet_bear, dtype=bool)
    bear_arr       = np.array(all_bear_mask, dtype=bool)
    non_quiet_bear = bear_arr & ~quiet_bear_arr
    bull_mask      = all_labels == 2
    neut_mask      = all_labels == 1

    groups = {
        "Quiet bear":     all_entropy[quiet_bear_arr],
        "Non-quiet bear": all_entropy[non_quiet_bear],
        "Neutral":        all_entropy[neut_mask],
        "Bull":           all_entropy[bull_mask],
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(len(groups)))
    colours_bp = ["#d62728", "#ff7f0e", "#7f7f7f", "#2ca02c"]
    bp = ax.boxplot([v for v in groups.values() if len(v) > 0],
                    positions=positions, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, col in zip(bp["boxes"], colours_bp):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([k for k, v in groups.items() if len(v) > 0], fontsize=10)
    ax.set_ylabel("Shannon entropy", fontsize=11)
    ax.set_title("Ensemble probability entropy by regime category\n"
                 "Higher entropy = lower classification confidence\n"
                 "S\\&P 500, NBER labels, Windows 1 and 3", fontsize=10)
    ax.axhline(np.log(3) * 0.9, color="grey", linestyle=":", linewidth=1,
               label="90% of max entropy (near-uniform)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_entropy_by_regime.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig7_entropy_by_regime.png")

    # ── Reliability curves ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    regime_names = ["Bear", "Neutral", "Bull"]
    colours      = ["#d62728", "#7f7f7f", "#2ca02c"]
    all_proba = []  # collect from results that have proba
    for r in results:
        if hasattr(r, "daily_ensemble_proba") and r.daily_ensemble_proba is not None:
            all_proba.append(np.asarray(r.daily_ensemble_proba))
    if all_proba:
        proba_all  = np.vstack(all_proba)
        labels_all = np.array(all_labels)
        for ax, regime_id, name, col in zip(axes, [0, 1, 2],
                                             regime_names, colours):
            true_bin = (labels_all == regime_id).astype(int)
            pred_prob = proba_all[:, regime_id]
            try:
                frac_pos, mean_pred = calibration_curve(
                    true_bin, pred_prob, n_bins=10, strategy="quantile")
                ax.plot(mean_pred, frac_pos, "s-", color=col, linewidth=2,
                        markersize=6, label="Ensemble")
                ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
                ax.set_xlabel(f"Mean predicted P({name})", fontsize=10)
                ax.set_ylabel("Observed frequency", fontsize=10)
                ax.set_title(f"{name} calibration", fontsize=11)
                ax.legend(fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            except Exception as e:
                ax.text(0.5, 0.5, f"Insufficient data\n{e}",
                        ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("Reliability curves: ensemble probability calibration\n"
                 "S\\&P 500, NBER labels, Windows 1 and 3", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_reliability_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig7_reliability_curves.png")

    df_conf = pd.DataFrame(conf_rows)
    out = os.path.join(RESULTS_DIR, "phase7_confidence.csv")
    df_conf.to_csv(out, index=False)
    print(f"  Saved: {out}")
    return df_conf


def _run_weight_based_entropy_proxy(results):
    """
    Proxy entropy analysis using weight divergence when ensemble proba
    not available. Shows |w_ens - w_base| by regime as a proxy for
    confidence collapse.
    """
    print("\n  Running weight-divergence proxy for entropy analysis...")
    rows = []
    for r in results:
        w_ens  = np.asarray(r.daily_weights)
        w_base = np.asarray(r.daily_baseline_weights)
        labels = np.asarray(r.daily_labels)
        sigma  = np.asarray(r.daily_sigma)
        wname  = r.window_name

        w_div = np.abs(w_ens - w_base)
        bear_mask = labels == 0
        if bear_mask.sum() > 0:
            sigma_med  = np.median(sigma[bear_mask])
            quiet_bear = bear_mask & (sigma <= sigma_med)
        else:
            quiet_bear = np.zeros(len(labels), dtype=bool)

        for mask, name in [(quiet_bear, "quiet_bear"),
                           (bear_mask & ~quiet_bear, "non_quiet_bear"),
                           (labels == 1, "neutral"),
                           (labels == 2, "bull")]:
            if mask.sum() > 0:
                rows.append({
                    "window": wname,
                    "regime": name,
                    "n": int(mask.sum()),
                    "mean_weight_divergence": float(w_div[mask].mean()),
                })
                print(f"    {wname} {name}: mean |w_ens-w_base| = "
                      f"{w_div[mask].mean():.4f} (n={mask.sum()})")

    if rows:
        df = pd.DataFrame(rows)
        out = os.path.join(RESULTS_DIR, "phase7_weight_divergence.csv")
        df.to_csv(out, index=False)
        print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_phase7():
    print("\n" + "#"*70)
    print("PHASE 7: ALTERNATIVE BEAR DEFINITIONS + CONFIDENCE ANALYSIS")
    print("#"*70)

    # Load S&P 500 NBER results with daily series
    results = load_results_with_series(ticker="^GSPC", label_scheme="nber")
    if not results:
        print("ERROR: No S&P 500 NBER results with daily series found.")
        print("Run Phase 1 first with updated pipeline.py")
        return

    # Filter to Windows 1 and 3 only (Window 2 has zero NBER bear days)
    results = [r for r in results
               if r.window_name in ("Window1_2000s", "Window3_2020s")]
    print(f"Using {len(results)} windows with NBER bear days")

    # Part A
    df_bears = run_alternative_bear_definitions(results)

    # Part B
    df_conf = run_confidence_analysis(results)

    print("\n" + "="*70)
    print("PHASE 7 COMPLETE")
    print("="*70)
    if len(df_bears) > 0:
        n_survive = (df_bears["w_diff"] < 0).sum()
        print(f"Quiet bear mechanism survives: {n_survive}/{len(df_bears)} "
              f"conditions across all bear definitions")
    return df_bears, df_conf


if __name__ == "__main__":
    run_phase7()
