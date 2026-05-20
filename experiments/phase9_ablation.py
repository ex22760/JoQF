"""
experiments/phase9_ablation.py
Phase 9: Ensemble Ablation Study
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

import config as cfg
from config import RESULTS_DIR, FIGURES_DIR, ANNUALIZATION

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_gspc_results(label_scheme="nber", vol_spec="ewma_vix"):
    results = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
            try:
                r = pickle.load(f)
                if (getattr(r, "ticker", "") == "^GSPC" and
                    getattr(r, "label_scheme", "") == label_scheme and
                    getattr(r, "vol_spec", "") == vol_spec and
                    getattr(r, "window_name", "") in
                    ("Window1_2000s", "Window3_2020s") and
                    hasattr(r, "daily_ensemble_proba") and
                    r.daily_ensemble_proba is not None):
                    results.append(r)
            except Exception:
                pass
    return results


def sharpe_from_weights(w_ens, w_base, ret, r_daily, sigma, sigma_med,
                         tc=5e-4, band_mult=1.5, band_roll=252):
    ret = np.asarray(ret)
    r_d = np.asarray(r_daily)
    sig = np.asarray(sigma)
    sig_med = float(np.median(sig))

    w_curr = 0.0
    port_r = np.zeros(len(ret))
    for t in range(len(ret)):
        ratio = sig[t] / (sig_med + 1e-8)
        ratio = np.clip(ratio, 0.5, 2.0)
        band  = band_mult * np.sqrt(tc) * ratio * 0.02
        diff  = w_ens[t] - w_curr
        if abs(diff) > band:
            w_curr = float(np.clip(w_curr + np.sign(diff) * band, 0, 1))
        port_r[t] = w_curr * ret[t] + (1 - w_curr) * r_d[t] - tc * abs(w_ens[t] - w_curr)

    s = pd.Series(port_r)
    sr = s.mean() / (s.std() + 1e-10) * np.sqrt(ANNUALIZATION)
    eq = (1 + s).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    return float(sr), float(mdd)


def quiet_bear_weight_diff(bear_mask, sigma, w_ens, w_base):
    bear_mask = np.asarray(bear_mask, dtype=bool)
    if bear_mask.sum() < 5:
        return np.nan, np.nan
    sigma_arr = np.asarray(sigma)
    sigma_med = np.median(sigma_arr[bear_mask])
    quiet     = bear_mask & (sigma_arr <= sigma_med)
    if quiet.sum() < 3:
        return np.nan, np.nan
    w_diff = (np.asarray(w_ens) - np.asarray(w_base))[quiet]
    _, p   = stats.ttest_1samp(w_diff, 0)
    return float(w_diff.mean()), float(p)


def run_ablation_on_result(r):
    proba  = np.asarray(r.daily_ensemble_proba)
    w_base = np.asarray(r.daily_baseline_weights)
    ret    = np.asarray(r.daily_returns)
    sigma  = np.asarray(r.daily_sigma)
    labels = np.asarray(r.daily_labels)
    wname  = r.window_name

    has_per_model = (hasattr(r, "gmm_proba") and r.gmm_proba is not None)

    if has_per_model:
        model_probas = {
            "GMM":  np.asarray(r.gmm_proba),
            "HMM":  np.asarray(r.hmm_proba),
            "RF":   np.asarray(r.rf_proba),
            "LSTM": np.asarray(r.lstm_proba),
        }
        model_weights = {
            "GMM":  getattr(r, "w_gmm", 0.25),
            "HMM":  getattr(r, "w_hmm", 0.25),
            "RF":   getattr(r, "w_rf",  0.25),
            "LSTM": getattr(r, "w_lstm", 0.25),
        }
    else:
        print(f"  WARNING: per-model probabilities not saved for {wname}")
        model_probas  = {"GMM": proba, "HMM": proba,
                         "RF": proba,  "LSTM": proba}
        model_weights = {"GMM": 0.25, "HMM": 0.25,
                         "RF": 0.25, "LSTM": 0.25}

    ablation_configs = {
        "Full (GMM+HMM+RF+LSTM)": ["GMM", "HMM", "RF", "LSTM"],
        "No GMM":                  ["HMM", "RF", "LSTM"],
        "No HMM":                  ["GMM", "RF", "LSTM"],
        "No RF":                   ["GMM", "HMM", "LSTM"],
        "No LSTM":                 ["GMM", "HMM", "RF"],
        "Unsupervised only":       ["GMM", "HMM"],
        "Supervised only":         ["RF", "LSTM"],
    }

    r_daily_approx = w_base * np.asarray(ret)
    bear_mask = labels == 0

    rows = []
    for cond_name, models in ablation_configs.items():
        total_w = sum(model_weights[m] for m in models)
        abl_proba = sum(
            model_weights[m] / total_w * model_probas[m]
            for m in models
        )

        conf = abl_proba.max(axis=1)
        p_bear = abl_proba[:, 0]
        p_bull = abl_proba[:, 2]

        low_conf = conf < 0.55
        abl_adj  = abl_proba.copy()
        abl_adj[low_conf] = 1.0 / 3.0
        p_bear_adj = abl_adj[:, 0]
        p_bull_adj = abl_adj[:, 2]
        conf_adj   = abl_adj.max(axis=1)

        w_regime = np.clip(
            w_base * (1 + cfg.ALPHA_BULL * p_bull_adj
                      - cfg.ALPHA_BEAR * p_bear_adj), 0, 1)
        w_ens = np.clip(conf_adj * w_regime + (1 - conf_adj) * w_base, 0, 1)

        sr, mdd = sharpe_from_weights(
            w_ens, w_base, ret, r_daily_approx, sigma, sigma.mean())

        qb_diff, qb_p = quiet_bear_weight_diff(
            bear_mask, sigma, w_ens, w_base)

        rows.append({
            "window":     wname,
            "condition":  cond_name,
            "models":     "+".join(models),
            "sharpe":     sr,
            "baseline_sharpe": r.baseline_sharpe,
            "delta_sharpe": sr - r.baseline_sharpe,
            "max_dd":     mdd,
            "baseline_max_dd": r.baseline_max_dd,
            "qb_diff":    qb_diff,
            "qb_p":       qb_p,
        })

        print(f"    {cond_name:<28} Sharpe={sr:.3f} "
              f"(Δ={sr-r.baseline_sharpe:+.3f})  "
              f"QBdiff={qb_diff:+.4f}" if not np.isnan(qb_diff)
              else f"    {cond_name:<28} Sharpe={sr:.3f} "
              f"(Δ={sr-r.baseline_sharpe:+.3f})  "
              f"QBdiff=N/A")

    return rows


def run_phase9():
    print("\n" + "#"*70)
    print("PHASE 9: ENSEMBLE ABLATION STUDY")
    print("#"*70)

    results = load_gspc_results()
    if not results:
        print("ERROR: No S&P 500 NBER results with daily_ensemble_proba found.")
        return pd.DataFrame()

    print(f"Loaded {len(results)} results for ablation")

    all_rows = []
    for r in results:
        print(f"\n  Window: {r.window_name}")
        rows = run_ablation_on_result(r)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    print("\n" + "="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)
    print(f"{'Condition':<28} {'W1 ΔSharpe':>11} {'W3 ΔSharpe':>11} "
          f"{'W1 QB diff':>11} {'W3 QB diff':>11}")
    print("-"*65)
    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond]
        w1 = sub[sub["window"] == "Window1_2000s"]
        w3 = sub[sub["window"] == "Window3_2020s"]
        ds1 = f"{w1['delta_sharpe'].values[0]:+.3f}" if len(w1) else "N/A"
        ds3 = f"{w3['delta_sharpe'].values[0]:+.3f}" if len(w3) else "N/A"
        qb1 = f"{w1['qb_diff'].values[0]:+.4f}" if len(w1) and not np.isnan(w1['qb_diff'].values[0]) else "N/A"
        qb3 = f"{w3['qb_diff'].values[0]:+.4f}" if len(w3) and not np.isnan(w3['qb_diff'].values[0]) else "N/A"
        print(f"  {cond:<28} {ds1:>11} {ds3:>11} {qb1:>11} {qb3:>11}")

    _plot_ablation(df)

    out = os.path.join(RESULTS_DIR, "phase9_ablation.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


def _plot_ablation(df):
    conditions = list(df["condition"].unique())
    windows    = ["Window1_2000s", "Window3_2020s"]
    w_labels   = ["Window 1\n(2000-09)", "Window 3\n(2020-26)"]
    colours    = plt.cm.tab10(np.linspace(0, 0.7, len(conditions)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(windows))
    width = 0.12

    # Left panel: Sharpe gap
    for i, (cond, col) in enumerate(zip(conditions, colours)):
        vals = []
        for w in windows:
            sub = df[(df["condition"] == cond) & (df["window"] == w)]
            vals.append(sub["delta_sharpe"].values[0] if len(sub) else 0)
        offset = (i - len(conditions)/2) * width + width/2
        axes[0].bar(x + offset, vals, width, label=cond, color=col, alpha=0.85)

    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(w_labels, fontsize=10)
    axes[0].set_xlabel("Rolling window", fontsize=10)
    axes[0].set_ylabel("$\\Delta$Sharpe (ablated $-$ baseline)", fontsize=10)
    axes[0].legend(fontsize=7, loc="lower right", ncol=2)
    axes[0].grid(True, alpha=0.25, axis="y")

    # Right panel: Quiet bear weight diff
    for i, (cond, col) in enumerate(zip(conditions, colours)):
        vals = []
        for w in windows:
            sub = df[(df["condition"] == cond) & (df["window"] == w)]
            v   = sub["qb_diff"].values[0] if len(sub) else np.nan
            vals.append(v if not np.isnan(v) else 0)
        offset = (i - len(conditions)/2) * width + width/2
        axes[1].bar(x + offset, vals, width, label=cond, color=col, alpha=0.85)

    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(w_labels, fontsize=10)
    axes[1].set_xlabel("Rolling window", fontsize=10)
    axes[1].set_ylabel("Quiet bear weight differential\n(ensemble $-$ baseline; more negative = stronger mechanism)", fontsize=10)
    axes[1].legend(fontsize=7, loc="lower right", ncol=2)
    axes[1].grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig10_ablation.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    run_phase9()