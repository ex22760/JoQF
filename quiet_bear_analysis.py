"""
quiet_bear_analysis.py
Tests the core hypothesis:

  "Volatility-targeting strategies misclassify low-volatility episodes
   within sustained bear markets as recovery signals, generating premature
   re-entry into equities and amplifying maximum drawdown."

Requires pipeline.py to have been updated to save daily weight series.
Run after rerunning Phase 1 with updated pipeline.

Steps:
  1. Load result pkl files (must contain daily_weights, daily_baseline_weights,
     daily_labels, daily_sigma fields)
  2. Identify quiet bear days: NBER bear days with below-median volatility
  3. Compare ensemble vs baseline equity weights on quiet bear days
  4. Test whether baseline weight - ensemble weight on quiet bear days
     predicts subsequent drawdown
  5. Repeat across all windows and assets

Output: Table 7 — Quiet Bear Day Analysis
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import RESULTS_DIR, ANNUALIZATION


def load_results_with_series() -> list:
    """Load pkl files that contain daily series."""
    results = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
            try:
                obj = pickle.load(f)
                if hasattr(obj, "daily_weights"):
                    results.append(obj)
            except Exception:
                pass
    print(f"Loaded {len(results)} results with daily series")
    return results


def identify_quiet_bear_days(labels: pd.Series,
                              sigma:  pd.Series) -> pd.Series:
    """
    Quiet bear days: days classified as bear (label=0) where
    volatility is below the median volatility across all bear days.
    Returns boolean Series.
    """
    bear_mask  = labels == 0
    if bear_mask.sum() == 0:
        return pd.Series(False, index=labels.index)
    bear_sigma = sigma[bear_mask]
    sigma_med  = bear_sigma.median()
    quiet_bear = bear_mask & (sigma <= sigma_med)
    return quiet_bear


def test_quiet_bear_hypothesis(result) -> dict:
    """
    For a single experiment result with daily series:
    1. Identify quiet bear days
    2. Compute mean weight difference (baseline - ensemble) on quiet bear days
    3. Compute subsequent 20-day return after quiet bear days for each strategy
    4. Test whether weight difference predicts subsequent drawdown
    """
    if not hasattr(result, "daily_weights"):
        return None

    labels   = result.daily_labels    # pd.Series, 0=bear 1=neutral 2=bull
    sigma    = result.daily_sigma     # pd.Series, annualised vol
    w_ens    = result.daily_weights   # pd.Series, ensemble equity weight
    w_base   = result.daily_baseline_weights  # pd.Series, baseline equity weight
    ret      = result.daily_returns   # pd.Series, daily returns

    if labels is None or len(labels) == 0:
        return None

    quiet_bear = identify_quiet_bear_days(labels, sigma)
    n_quiet    = quiet_bear.sum()

    if n_quiet == 0:
        return {"ticker": result.ticker, "window": result.window_name,
                "n_quiet_bear": 0, "note": "no quiet bear days"}

    # Weight difference on quiet bear days
    w_diff_qb = (w_base - w_ens)[quiet_bear]  # positive = baseline more invested

    # All bear days weight difference
    bear_mask  = labels == 0
    w_diff_all = (w_base - w_ens)[bear_mask] if bear_mask.sum() > 0 else pd.Series()

    # Subsequent 20-day max drawdown after quiet bear days
    # For each quiet bear day, compute the max drawdown over next 20 days
    subsequent_dd_base = []
    subsequent_dd_ens  = []

    # Build equity curves from weights and returns
    eq_ens  = (1 + w_ens  * ret + (1 - w_ens)  * 0.0).cumprod()
    eq_base = (1 + w_base * ret + (1 - w_base) * 0.0).cumprod()

    idx = labels.index
    for i, date in enumerate(idx):
        if not quiet_bear.loc[date]:
            continue
        end_i = min(i + 20, len(idx) - 1)
        future_base = eq_base.iloc[i:end_i+1]
        future_ens  = eq_ens.iloc[i:end_i+1]
        if len(future_base) > 1:
            dd_base = float((future_base / future_base.iloc[0]).min() - 1)
            dd_ens  = float((future_ens  / future_ens.iloc[0]).min() - 1)
            subsequent_dd_base.append(dd_base)
            subsequent_dd_ens.append(dd_ens)

    mean_dd_base = np.mean(subsequent_dd_base) if subsequent_dd_base else np.nan
    mean_dd_ens  = np.mean(subsequent_dd_ens)  if subsequent_dd_ens  else np.nan

    # t-test: is baseline weight significantly higher than ensemble on quiet bear days?
    if len(w_diff_qb) > 1:
        t_stat, p_val = stats.ttest_1samp(w_diff_qb.values, 0)
    else:
        t_stat, p_val = np.nan, np.nan

    return {
        "ticker":            result.ticker,
        "window":            result.window_name,
        "vol_spec":          result.vol_spec,
        "n_bear_days":       int(bear_mask.sum()),
        "n_quiet_bear":      int(n_quiet),
        "pct_quiet":         n_quiet / bear_mask.sum() * 100 if bear_mask.sum() > 0 else 0,
        "mean_w_diff_quiet": float(w_diff_qb.mean()),   # baseline - ensemble on quiet bears
        "mean_w_diff_all":   float(w_diff_all.mean()) if len(w_diff_all) > 0 else np.nan,
        "t_stat":            float(t_stat),
        "p_value":           float(p_val),
        "mean_dd_base_20d":  mean_dd_base,
        "mean_dd_ens_20d":   mean_dd_ens,
        "dd_diff_20d":       mean_dd_base - mean_dd_ens if not np.isnan(mean_dd_base) else np.nan,
    }


def run_quiet_bear_analysis():
    results = load_results_with_series()
    if not results:
        print("\nNo results with daily series found.")
        print("Update pipeline.py to save daily series and rerun Phase 1.")
        return

    # Primary definition: drawdown-threshold (all 3 windows, richer bear days)
    # Drawdown gives Window 2 bear days unlike NBER (0 NBER bear days 2010-2019)
    # Filter to S&P 500 drawdown labels as primary; all assets as robustness
    results_primary = [r for r in results
                       if r.label_scheme == "drawdown" and r.ticker == "^GSPC"]
    results_all     = [r for r in results if r.label_scheme == "nber"]

    # Use drawdown primary for main table; fall back to all NBER if no drawdown
    results = results_primary if results_primary else results_all
    print(f"Filtered to {len(results)} S&P 500 drawdown experiments "
          f"(primary definition — richer bear day sample than NBER)")

    print("\n" + "="*80)
    print("TABLE 7: QUIET BEAR DAY ANALYSIS (primary: drawdown-threshold labels)")
    print("Hypothesis: ensemble confidently misclassifies quiet bear days,")
    print("actively increasing equity weight and suffering worse forward drawdown")
    print("="*80)

    rows = []
    for r in results:
        row = test_quiet_bear_hypothesis(r)
        if row and row.get("n_quiet_bear", 0) > 0:
            rows.append(row)

    if not rows:
        print("No valid experiments with quiet bear days found.")
        return

    df = pd.DataFrame(rows)

    print(f"\n{'Ticker':<8} {'Window':<20} {'N Bear':>6} {'N Quiet':>7} "
          f"{'Pct':>5} {'W Diff':>8} {'p-val':>7} "
          f"{'BL DD':>8} {'Ens DD':>8} {'Δ DD':>7}")
    print("-"*80)

    for _, row in df.iterrows():
        print(f"{row['ticker']:<8} {row['window']:<20} "
              f"{row['n_bear_days']:>6} {row['n_quiet_bear']:>7} "
              f"{row['pct_quiet']:>5.1f}% "
              f"{row['mean_w_diff_quiet']:>+8.4f} "
              f"{row['p_value']:>7.4f} "
              f"{row['mean_dd_base_20d']:>8.4f} "
              f"{row['mean_dd_ens_20d']:>8.4f} "
              f"{row['dd_diff_20d']:>+7.4f}")

    print("\nSummary:")
    print(f"  Mean baseline weight advantage on quiet bear days: "
          f"{df['mean_w_diff_quiet'].mean():+.4f}")
    print(f"  Mean 20-day DD difference (baseline - ensemble): "
          f"{df['dd_diff_20d'].mean():+.4f}")
    print(f"  Consistent (baseline > ensemble weight on quiet bears): "
          f"{(df['mean_w_diff_quiet'] > 0).sum()}/{len(df)}")
    print(f"  Consistent (baseline suffers worse DD after quiet bears): "
          f"{(df['dd_diff_20d'] < 0).sum()}/{len(df)}")

    # Save
    out = os.path.join(RESULTS_DIR, "quiet_bear_analysis.csv")
    df.to_csv(out, index=False)
    print(f"\nResults saved: {out}")

    return df


if __name__ == "__main__":
    run_quiet_bear_analysis()