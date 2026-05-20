"""
volatility/factory.py
Volatility model factory — returns sigma_ann for any specification.
Swapping vol model requires changing one string in config.py.

Specifications:
  "ewma"      — EWMA realised volatility
  "vix"       — VIX implied volatility only
  "ewma_vix"  — blend (current baseline)
  "garch"     — GARCH(1,1)
  "gjr_garch" — GJR-GARCH(1,1) with leverage effect
"""

import numpy as np
import pandas as pd
import os, sys, warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (LOOKBACK_LT, ANNUALIZATION, VIX_WEIGHT,
                    GARCH_P, GARCH_Q, GARCH_DIST, GARCH_REFIT_FREQ,
                    WARMUP, VOL_SPECS)

warnings.filterwarnings("ignore")


def _check_arch():
    try:
        from arch import arch_model
        return True
    except ImportError:
        raise ImportError(
            "arch package required for GARCH. Install: pip install arch"
        )


# ── EWMA ──────────────────────────────────────────────────────────────────────

def _causal_ewm_std(x: pd.Series, span: int) -> pd.Series:
    alpha = 2.0 / (span + 1)
    result = np.full(len(x), np.nan)
    m = v = np.nan
    for i, val in enumerate(x.values):
        if np.isnan(val):
            result[i] = np.nan; continue
        if np.isnan(m):
            m, v, result[i] = val, 0.0, np.nan
        else:
            p = m
            m = (1-alpha)*m + alpha*val
            v = (1-alpha)*(v + alpha*(val-p)**2)
            result[i] = np.sqrt(v)
    return pd.Series(result, index=x.index)


def vol_ewma(logret: pd.Series, vix: pd.Series = None,
             train_end: str = None, **kwargs) -> pd.Series:
    """EWMA realised volatility, annualised."""
    return _causal_ewm_std(logret, LOOKBACK_LT) * np.sqrt(ANNUALIZATION)


def vol_vix(logret: pd.Series, vix: pd.Series = None,
            train_end: str = None, **kwargs) -> pd.Series:
    """VIX implied volatility only, annualised."""
    if vix is None:
        raise ValueError("VIX Series required for vol_spec='vix'")
    return (vix / 100.0).reindex(logret.index).ffill().clip(lower=1e-6)


def vol_ewma_vix(logret: pd.Series, vix: pd.Series = None,
                  train_end: str = None, **kwargs) -> pd.Series:
    """Blend of EWMA realised vol and VIX, annualised. (Current baseline)"""
    ewma = vol_ewma(logret)
    if vix is None:
        return ewma
    vix_ann = (vix / 100.0).reindex(logret.index).ffill()
    return ((1 - VIX_WEIGHT) * ewma + VIX_WEIGHT * vix_ann).clip(lower=1e-6)


# ── GARCH(1,1) ────────────────────────────────────────────────────────────────

def _fit_garch(returns_pct: pd.Series, model_type: str = "garch"):
    """
    Fit GARCH(1,1) or GJR-GARCH(1,1) on a returns series.
    Returns arch_model result object.
    """
    from arch import arch_model as arch_m

    if model_type == "garch":
        am = arch_m(returns_pct, vol="Garch", p=GARCH_P, q=GARCH_Q,
                    dist=GARCH_DIST, mean="Zero", rescale=False)
    elif model_type == "gjr_garch":
        am = arch_m(returns_pct, vol="EGARCH" if False else "Garch",
                    p=GARCH_P, o=1, q=GARCH_Q,
                    dist=GARCH_DIST, mean="Zero", rescale=False)
    else:
        raise ValueError(f"Unknown GARCH model type: {model_type}")

    res = am.fit(disp="off", show_warning=False)
    return res


def _garch_forecast_causal(logret:     pd.Series,
                            train_end:  str,
                            model_type: str = "garch") -> pd.Series:
    """
    Causal GARCH volatility forecast.
    - Fit on train period (up to train_end)
    - Re-fit annually on expanding window (GARCH_REFIT_FREQ)
    - Apply one-step-ahead forecast causally on all dates
    Returns annualised sigma Series.
    """
    from arch import arch_model as arch_m

    returns_pct = (logret * 100.0).dropna()
    result      = pd.Series(np.nan, index=logret.index)

    train_mask = returns_pct.index <= train_end
    if train_mask.sum() < 252:
        raise ValueError("Insufficient training data for GARCH fitting "
                         "(need at least 252 observations)")

    # Initial fit on training data
    r_train  = returns_pct[train_mask]
    res      = _fit_garch(r_train, model_type)
    params   = res.params
    last_fit = r_train.index[-1]

    # Build conditional vol series date by date
    # For efficiency: fit once, then use recursion for forecast
    # Refit every GARCH_REFIT_FREQ days
    h_t = float(res.conditional_volatility.iloc[-1] ** 2)   # last var

    all_dates = returns_pct.index
    fit_dates = set(returns_pct[train_mask].index)

    # Recursive GARCH(1,1): h_t = omega + alpha*eps_{t-1}^2 + beta*h_{t-1}
    if model_type == "garch":
        omega = float(params.get("omega", params.iloc[0]))
        alpha = float(params.get("alpha[1]", params.iloc[1]))
        beta  = float(params.get("beta[1]",  params.iloc[2]))
    else:
        # GJR parameters
        omega  = float(params.get("omega", params.iloc[0]))
        alpha  = float(params.get("alpha[1]", params.iloc[1]))
        gamma  = float(params.get("gamma[1]", params.iloc[2]))
        beta   = float(params.get("beta[1]",  params.iloc[3]))

    days_since_refit = 0
    eps_prev = 0.0

    for t, date in enumerate(all_dates):
        r_val = returns_pct.loc[date]

        # Refit on expanding window periodically after train_end
        if (date > last_fit and
                days_since_refit >= GARCH_REFIT_FREQ):
            try:
                r_exp = returns_pct.loc[:date]
                res_new = _fit_garch(r_exp, model_type)
                p = res_new.params
                if model_type == "garch":
                    omega = float(p.get("omega", p.iloc[0]))
                    alpha = float(p.get("alpha[1]", p.iloc[1]))
                    beta  = float(p.get("beta[1]",  p.iloc[2]))
                else:
                    omega = float(p.get("omega",    p.iloc[0]))
                    alpha = float(p.get("alpha[1]", p.iloc[1]))
                    gamma = float(p.get("gamma[1]", p.iloc[2]))
                    beta  = float(p.get("beta[1]",  p.iloc[3]))
                h_t = float(res_new.conditional_volatility.iloc[-1] ** 2)
                days_since_refit = 0
            except Exception:
                pass

        # Update variance
        if model_type == "garch":
            h_t = max(omega + alpha * eps_prev**2 + beta * h_t, 1e-8)
        else:
            # GJR: asymmetric effect for negative shocks
            indicator = 1.0 if eps_prev < 0 else 0.0
            h_t = max(omega + (alpha + gamma*indicator)*eps_prev**2
                      + beta*h_t, 1e-8)

        result.loc[date] = np.sqrt(h_t) * np.sqrt(ANNUALIZATION) / 100.0
        eps_prev = r_val
        if date > last_fit:
            days_since_refit += 1

    return result.clip(lower=1e-6)


def vol_garch(logret: pd.Series, vix: pd.Series = None,
              train_end: str = None, **kwargs) -> pd.Series:
    """GARCH(1,1) conditional volatility, annualised."""
    if train_end is None:
        raise ValueError("train_end required for GARCH vol spec")
    _check_arch()
    return _garch_forecast_causal(logret, train_end, model_type="garch")


def vol_gjr_garch(logret: pd.Series, vix: pd.Series = None,
                   train_end: str = None, **kwargs) -> pd.Series:
    """
    GJR-GARCH(1,1) conditional volatility, annualised.
    Captures leverage effect: negative shocks increase vol more than positive.
    Preferred over EGARCH: simpler, interpretable, directly tests asymmetry.
    """
    if train_end is None:
        raise ValueError("train_end required for GJR-GARCH vol spec")
    _check_arch()
    return _garch_forecast_causal(logret, train_end, model_type="gjr_garch")


# ── Factory dispatcher ────────────────────────────────────────────────────────

_VOL_REGISTRY = {
    "ewma":      vol_ewma,
    "vix":       vol_vix,
    "ewma_vix":  vol_ewma_vix,
    "garch":     vol_garch,
    "gjr_garch": vol_gjr_garch,
}


def get_vol_model(spec: str):
    """
    Return the volatility function for a given specification string.

    Usage:
        vol_fn = get_vol_model("garch")
        sigma_ann = vol_fn(logret, vix=vix_series, train_end="2009-12-31")
    """
    if spec not in _VOL_REGISTRY:
        raise ValueError(f"Unknown vol spec: '{spec}'. "
                         f"Choose from {list(_VOL_REGISTRY.keys())}")
    return _VOL_REGISTRY[spec]


def compute_sigma(logret:    pd.Series,
                  vix:       pd.Series = None,
                  spec:      str = "ewma_vix",
                  train_end: str = None) -> pd.Series:
    """
    Convenience function: compute sigma_ann for a given spec.
    This is the single call used in pipeline.py.
    """
    fn = get_vol_model(spec)
    return fn(logret, vix=vix, train_end=train_end)
