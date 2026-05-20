"""
experiments/significance.py
Phase 5: Statistical significance across all experimental conditions.

Tests:
  1. Panel Jobson-Korkie: Sharpe improvement across N experiments
  2. Bootstrap confidence intervals on Sharpe differences
  3. Diebold-Mariano test on forecast accuracy
  4. Fraction of conditions where ensemble outperforms (sign test)
"""

import os, sys, pickle
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import RESULTS_DIR, ANNUALIZATION


# ── Jobson-Korkie test ────────────────────────────────────────────────────────

def jobson_korkie_test(sharpe_a: float,
                        sharpe_b: float,
                        n:        int) -> tuple:
    """
    Jobson & Korkie (1981) test for equality of Sharpe ratios.
    Corrected by Memmel (2003).
    Returns (z_stat, p_value_two_tailed).
    """
    se = np.sqrt((1 + 0.5 * sharpe_a**2) / n)
    z  = (sharpe_a - sharpe_b) / se
    p  = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def panel_jobson_korkie(results: list) -> dict:
    """
    Panel Jobson-Korkie: pool Sharpe differences across N experiments.
    Under H0: mean Sharpe improvement = 0.
    Returns test statistic and p-value for the panel.
    """
    deltas = [r.sharpe - r.baseline_sharpe for r in results]
    n_exp  = len(deltas)
    mean_d = np.mean(deltas)
    se_d   = np.std(deltas, ddof=1) / np.sqrt(n_exp)
    t_stat = mean_d / se_d if se_d > 0 else np.nan
    p_val  = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n_exp-1)))
    return {
        "n_experiments":    n_exp,
        "mean_delta_sharpe": mean_d,
        "se":               se_d,
        "t_stat":           t_stat,
        "p_value":          p_val,
        "n_positive":       sum(1 for d in deltas if d > 0),
        "n_negative":       sum(1 for d in deltas if d < 0),
    }


# ── Bootstrap confidence intervals ───────────────────────────────────────────

def bootstrap_sharpe_ci(returns_a: pd.Series,
                          returns_b: pd.Series,
                          n_boot:    int = 10000,
                          ci:        float = 0.95,
                          seed:      int = 42) -> dict:
    """
    Bootstrap confidence interval for Sharpe ratio difference (a - b).
    Returns dict with mean, lower, upper, and p_value.
    """
    rng     = np.random.default_rng(seed)
    n       = len(returns_a)
    ra, rb  = returns_a.values, returns_b.values
    diffs   = []

    for _ in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        sa_b = ra[idx]
        sb_b = rb[idx]
        sr_a = (sa_b.mean() * ANNUALIZATION /
                (sa_b.std() * np.sqrt(ANNUALIZATION) + 1e-10))
        sr_b = (sb_b.mean() * ANNUALIZATION /
                (sb_b.std() * np.sqrt(ANNUALIZATION) + 1e-10))
        diffs.append(sr_a - sr_b)

    diffs = np.array(diffs)
    alpha = 1 - ci
    lower = float(np.percentile(diffs, alpha/2 * 100))
    upper = float(np.percentile(diffs, (1-alpha/2) * 100))
    p_val = float(np.mean(diffs <= 0)) * 2   # two-tailed

    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower":  lower,
        "ci_upper":  upper,
        "p_value":   p_val,
        "ci_level":  ci,
    }


# ── Diebold-Mariano test ──────────────────────────────────────────────────────

def diebold_mariano(y_true:  np.ndarray,
                     pred_a:  np.ndarray,
                     pred_b:  np.ndarray) -> tuple:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.
    Uses squared prediction error as loss function.
    H0: E[d_t] = 0, where d_t = e_a_t^2 - e_b_t^2.
    Returns (dm_stat, p_value_two_tailed).
    """
    e_a = (y_true - pred_a) ** 2
    e_b = (y_true - pred_b) ** 2
    d   = e_a - e_b
    n   = len(d)
    d_bar = d.mean()
    # Newey-West variance with lag=1
    gamma_0 = np.var(d, ddof=1)
    gamma_1 = np.cov(d[:-1], d[1:])[0, 1] if n > 2 else 0
    nw_var  = (gamma_0 + 2 * gamma_1) / n
    dm_stat = d_bar / np.sqrt(max(nw_var, 1e-10))
    p_val   = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    return float(dm_stat), p_val


# ── Sign test ─────────────────────────────────────────────────────────────────

def sign_test(results: list) -> dict:
    """
    Non-parametric sign test: fraction of experiments where ensemble wins.
    H0: P(ensemble > baseline) = 0.5 (no consistent advantage).
    """
    wins   = sum(1 for r in results if r.sharpe > r.baseline_sharpe)
    n      = len(results)
    # Binomial test
    # binom_test renamed to binomtest in scipy >= 1.7
    try:
        p_val = float(stats.binomtest(wins, n, p=0.5, alternative="greater").pvalue)
    except AttributeError:
        p_val = float(stats.binom_test(wins, n, p=0.5, alternative="greater"))
    return {
        "n_experiments": n,
        "n_wins":        wins,
        "win_rate":      wins / n,
        "p_value":       p_val,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def load_all_results() -> list:
    """Load all ExperimentResult objects from RESULTS_DIR."""
    results = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".pkl"):
            with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
                try:
                    obj = pickle.load(f)
                    if hasattr(obj, "sharpe"):
                        results.append(obj)
                except Exception:
                    pass
    print(f"Loaded {len(results)} experiment results")
    return results


def run_significance_analysis(results: list = None):
    if results is None:
        results = load_all_results()

    print("\n" + "="*70)
    print("TABLE 6: STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*70)

    # ── Panel JK ──────────────────────────────────────────────────────────────
    print("\n1. Panel Jobson-Korkie (all experiments):")
    pjk = panel_jobson_korkie(results)
    print(f"   N experiments:     {pjk['n_experiments']}")
    print(f"   Mean Δ Sharpe:     {pjk['mean_delta_sharpe']:+.4f}")
    print(f"   SE:                {pjk['se']:.4f}")
    print(f"   t-statistic:       {pjk['t_stat']:.3f}")
    print(f"   p-value:           {pjk['p_value']:.4f}")
    print(f"   Wins/Losses:       {pjk['n_positive']}/{pjk['n_negative']}")

    # By subgroup
    for subgroup_key in ["vol_spec", "label_scheme", "window_name"]:
        groups = {}
        for r in results:
            g = getattr(r, subgroup_key, "unknown")
            if g not in groups: groups[g] = []
            groups[g].append(r)

        print(f"\n   Breakdown by {subgroup_key}:")
        for g, grp_results in sorted(groups.items()):
            pjk_g = panel_jobson_korkie(grp_results)
            print(f"     {g:<15} n={pjk_g['n_experiments']:>2}  "
                  f"mean_Δ={pjk_g['mean_delta_sharpe']:+.3f}  "
                  f"p={pjk_g['p_value']:.3f}  "
                  f"wins={pjk_g['n_positive']}/{pjk_g['n_experiments']}")

    # ── Sign test ──────────────────────────────────────────────────────────────
    print("\n2. Sign test (non-parametric):")
    st = sign_test(results)
    print(f"   Win rate:    {st['n_wins']}/{st['n_experiments']} "
          f"= {st['win_rate']:.1%}")
    print(f"   p-value:     {st['p_value']:.4f} "
          f"({'significant' if st['p_value'] < 0.05 else 'not significant'} at 5%)")

    # ── MaxDD analysis ────────────────────────────────────────────────────────
    print("\n3. MaxDD improvement (positive finding):")
    max_dd_deltas = [r.max_dd - r.baseline_max_dd for r in results
                     if not (np.isnan(r.max_dd) or np.isnan(r.baseline_max_dd))]
    max_dd_pjk = panel_jobson_korkie([type("R", (), {
        "sharpe": r.max_dd, "baseline_sharpe": r.baseline_max_dd})()
        for r in results
        if not (np.isnan(r.max_dd) or np.isnan(r.baseline_max_dd))])
    print(f"   N experiments:     {len(max_dd_deltas)}")
    print(f"   Mean Δ MaxDD:      {np.mean(max_dd_deltas):+.4f} "
          f"({'improvement' if np.mean(max_dd_deltas) < 0 else 'worsening'})")
    print(f"   SE:                {np.std(max_dd_deltas, ddof=1)/np.sqrt(len(max_dd_deltas)):.4f}")
    wins_dd = sum(1 for d in max_dd_deltas if d < 0)
    print(f"   Wins (lower MaxDD): {wins_dd}/{len(max_dd_deltas)} "
          f"({wins_dd/len(max_dd_deltas)*100:.0f}%)")

    # MaxDD by window
    print("\n   MaxDD improvement by window:")
    for window in ["Window1_2000s", "Window2_2010s", "Window3_2020s"]:
        wr = [r for r in results if r.window_name == window
              and not (np.isnan(r.max_dd) or np.isnan(r.baseline_max_dd))]
        if wr:
            deltas_w = [r.max_dd - r.baseline_max_dd for r in wr]
            wins_w   = sum(1 for d in deltas_w if d < 0)
            print(f"     {window:<20} mean_Δ={np.mean(deltas_w):+.4f}  "
                  f"wins={wins_w}/{len(deltas_w)}")

    # MaxDD by ticker (cross-market)
    tickers = sorted(set(r.ticker for r in results))
    if len(tickers) > 1:
        print("\n   MaxDD improvement by asset:")
        for ticker in tickers:
            tr = [r for r in results if r.ticker == ticker
                  and not (np.isnan(r.max_dd) or np.isnan(r.baseline_max_dd))]
            if tr:
                deltas_t = [r.max_dd - r.baseline_max_dd for r in tr]
                wins_t   = sum(1 for d in deltas_t if d < 0)
                print(f"     {ticker:<10} mean_Δ={np.mean(deltas_t):+.4f}  "
                      f"wins={wins_t}/{len(deltas_t)}")

    # Bear weight reduction
    print("\n4. Bear weight reduction (mechanism check):")
    bear_reductions = []
    for r in results:
        if not (np.isnan(r.bear_mean_weight) or np.isnan(r.baseline_bear_weight)
                or r.baseline_bear_weight == 0):
            reduction = (r.baseline_bear_weight - r.bear_mean_weight) / r.baseline_bear_weight
            bear_reductions.append(reduction)
    if bear_reductions:
        print(f"   Mean bear weight reduction: {np.mean(bear_reductions)*100:.1f}%")
        print(f"   Range: {min(bear_reductions)*100:.1f}% to {max(bear_reductions)*100:.1f}%")
        print(f"   Consistent (>0): {sum(1 for r in bear_reductions if r > 0)}/{len(bear_reductions)}")

    # ── Volatility spec comparison ─────────────────────────────────────────────
    print("\n5. Are Sharpe differences driven by volatility spec?")
    vol_results = {}
    for r in results:
        if r.vol_spec not in vol_results:
            vol_results[r.vol_spec] = []
        vol_results[r.vol_spec].append(r.sharpe - r.baseline_sharpe)

    print(f"   {'Vol Spec':<12} {'Mean Δ':>8} {'Std':>6} {'N':>4}")
    for spec, deltas in sorted(vol_results.items()):
        print(f"   {spec:<12} {np.mean(deltas):>+8.4f} "
              f"{np.std(deltas):>6.4f} {len(deltas):>4}")

    # F-test: are vol spec means significantly different?
    if len(vol_results) >= 2:
        groups_anova = [d for d in vol_results.values() if len(d) >= 2]
        if len(groups_anova) >= 2:
            f_stat, p_anova = stats.f_oneway(*groups_anova)
            print(f"\n   One-way ANOVA across vol specs:")
            print(f"   F={f_stat:.3f}, p={p_anova:.4f} "
                  f"({'significant' if p_anova < 0.05 else 'not significant'})")

    print("="*70)

    return {
        "panel_jk":  pjk,
        "sign_test": st,
    }


if __name__ == "__main__":
    run_significance_analysis()