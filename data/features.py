"""
data/features.py
Causal feature engineering — all features depend only on F_t.
Returns a standardised feature DataFrame ready for model input.
"""

import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (LOOKBACK_ST, LOOKBACK_LT, ANNUALIZATION,
                    VIX_WEIGHT, MACRO_WEIGHT, ALPHA_SHORT,
                    RISK_AVERSION, WARMUP)


# ── Causal EWM helpers ────────────────────────────────────────────────────────

def causal_ewm_mean(x: pd.Series, span: int) -> pd.Series:
    """Exponentially weighted mean — strictly causal (no pandas ewm lookahead)."""
    alpha  = 2.0 / (span + 1)
    result = np.full(len(x), np.nan)
    val    = np.nan
    for i, v in enumerate(x.values):
        if np.isnan(v):
            result[i] = np.nan
            continue
        val       = v if np.isnan(val) else (1 - alpha) * val + alpha * v
        result[i] = val
    return pd.Series(result, index=x.index)


def causal_ewm_std(x: pd.Series, span: int) -> pd.Series:
    """Exponentially weighted standard deviation — strictly causal."""
    alpha    = 2.0 / (span + 1)
    result   = np.full(len(x), np.nan)
    ewm_mean = ewm_var = np.nan
    for i, v in enumerate(x.values):
        if np.isnan(v):
            result[i] = np.nan
            continue
        if np.isnan(ewm_mean):
            ewm_mean, ewm_var, result[i] = v, 0.0, np.nan
        else:
            prev     = ewm_mean
            ewm_mean = (1 - alpha) * ewm_mean + alpha * v
            ewm_var  = (1 - alpha) * (ewm_var + alpha * (v - prev) ** 2)
            result[i] = np.sqrt(ewm_var)
    return pd.Series(result, index=x.index)


# ── Macro signal ──────────────────────────────────────────────────────────────

def build_macro_signal(df: pd.DataFrame,
                        macro_cols: list = None,
                        lookback:   int  = LOOKBACK_LT) -> pd.Series:
    """
    Causal rolling z-score composite of macro indicators.
    Negative sign: high inflation/unemployment/rates = bearish for equities.
    Returns daily Series (0 if macro not available).
    """
    if macro_cols is None:
        macro_cols = [c for c in ["CPI", "Unemployment", "FedFunds"]
                      if c in df.columns]
    if not macro_cols:
        return pd.Series(0.0, index=df.index)

    z = df[macro_cols].rolling(lookback).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8),
        raw=False
    ).clip(-3, 3)

    weights = {"CPI": -0.4, "Unemployment": -0.3, "FedFunds": -0.3}
    signal  = sum(z[c] * weights.get(c, -0.33)
                  for c in z.columns if c in weights)
    return signal.fillna(0.0)


# ── Volatility via EWMA + VIX ─────────────────────────────────────────────────
# (Other specs in volatility/factory.py)

def build_ewma_vol(logret:  pd.Series,
                   span:    int   = LOOKBACK_LT) -> pd.Series:
    """EWMA realised volatility, annualised."""
    return causal_ewm_std(logret, span) * np.sqrt(ANNUALIZATION)


def build_ewma_vix_vol(logret:  pd.Series,
                        vix:     pd.Series,
                        ewma_w:  float = VIX_WEIGHT,
                        span:    int   = LOOKBACK_LT) -> pd.Series:
    """Blend of EWMA realised vol and VIX implied vol, annualised."""
    ewma = build_ewma_vol(logret, span)
    vix_ann = (vix / 100.0).reindex(ewma.index).ffill()
    return ((1 - ewma_w) * ewma + ewma_w * vix_ann).clip(lower=1e-6)


# ── HJB signal ────────────────────────────────────────────────────────────────

def build_hjb_signal(logret:       pd.Series,
                      sigma_ann:    pd.Series,
                      r_daily:      pd.Series,
                      gamma:        float = RISK_AVERSION) -> pd.Series:
    """
    Merton optimal risky weight: u* = (mu - r) / (gamma * sigma^2)
    Clipped to [0, 1]. First WARMUP days zeroed.
    """
    mu_st  = causal_ewm_mean(logret, LOOKBACK_ST) * ANNUALIZATION
    mu_lt  = causal_ewm_mean(logret, LOOKBACK_LT) * ANNUALIZATION
    mu_ann = ALPHA_SHORT * mu_st + (1 - ALPHA_SHORT) * mu_lt

    excess_mu = mu_ann - r_daily * ANNUALIZATION
    u_star    = (excess_mu / (gamma * sigma_ann ** 2)).clip(0.0, 1.0)
    u_star.iloc[:WARMUP] = 0.0
    return u_star


# ── Full feature matrix ───────────────────────────────────────────────────────

def build_features(df:          pd.DataFrame,
                   sigma_ann:   pd.Series) -> pd.DataFrame:
    """
    Build the full feature matrix used by all models.
    Requires sigma_ann to already be computed (by the vol factory).

    Features:
      return    — log return
      vol       — annualised sigma
      momentum  — blended EWMA return forecast
      macro     — composite macro z-score signal
      vix       — VIX level (if available)

    Returns
    -------
    pd.DataFrame, rows = trading days, columns = features
    All NaN rows dropped.
    """
    logret = np.log(df["Close"]).diff().rename("LogReturn")

    mu_st  = causal_ewm_mean(logret, LOOKBACK_ST) * ANNUALIZATION
    mu_lt  = causal_ewm_mean(logret, LOOKBACK_LT) * ANNUALIZATION

    macro_signal = build_macro_signal(df)
    mu_ann = (ALPHA_SHORT * mu_st + (1 - ALPHA_SHORT) * mu_lt
              + MACRO_WEIGHT * macro_signal)

    features = {
        "return":   logret,
        "vol":      sigma_ann,
        "momentum": mu_ann,
        "macro":    macro_signal,
    }
    if "VIX" in df.columns:
        features["vix"] = df["VIX"].astype(float)

    X = pd.DataFrame(features).dropna()
    return X


def build_r_daily(df: pd.DataFrame) -> pd.Series:
    """Daily risk-free rate from Fed Funds Rate column."""
    if "FedFunds" in df.columns:
        r = df["FedFunds"].astype(float)
        r = r / (100.0 if r.max() > 1.0 else 1.0) / ANNUALIZATION
        return r.reindex(df.index).ffill().fillna(0.0)
    return pd.Series(0.0, index=df.index)


def build_no_trade_band(sigma_ann:   pd.Series,
                         tc:          float,
                         band_mult:   float,
                         band_roll:   int,
                         warmup:      int) -> pd.Series:
    """
    Volatility-scaled no-trade band — causal, lagged 1 day.
    band_t = band_mult * sqrt(tc) * clip(sigma_t / sigma_median_t, 0.5, 2.0) * 0.02
    """
    sigma_roll = sigma_ann.rolling(band_roll, min_periods=LOOKBACK_LT).median()
    sigma_exp  = sigma_ann.expanding(min_periods=1).median()
    sigma_med  = sigma_roll.fillna(sigma_exp)

    band = (band_mult * np.sqrt(tc)
            * (sigma_ann / sigma_med).clip(0.5, 2.0)
            * 0.02)
    band = band.fillna(band.expanding(min_periods=1).median())
    band.iloc[:warmup] = 999.0

    # Lag 1 day — causal
    band = band.shift(1).fillna(band.expanding(min_periods=1).median())
    return band
