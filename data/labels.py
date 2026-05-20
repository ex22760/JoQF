"""
data/labels.py
Regime label construction under two schemes:
  1. NBER — recession dates + Pagan-Sossounov bull rule
  2. Drawdown — peak-to-trough threshold (>15% = bear)
Both return integer Series: 0=bear, 1=neutral, 2=bull
"""

import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (PAGAN_MIN_DAYS, PAGAN_MIN_RET,
                    DRAWDOWN_BEAR_THRESHOLD, LABEL_SCHEMES)


LABEL_MAP     = {"bear": 0, "neutral": 1, "bull": 2}
LABEL_MAP_INV = {0: "bear", 1: "neutral", 2: "bull"}


def pagan_sossounov_bull(px: pd.Series,
                          min_days: int = PAGAN_MIN_DAYS,
                          min_ret:  float = PAGAN_MIN_RET) -> pd.Series:
    """
    Pagan & Sossounov (2003) peak-trough bull market rule.
    A bull phase runs from trough to peak if:
      - duration >= min_days trading days
      - cumulative return >= min_ret
    Returns boolean Series (True = bull day).
    """
    prices = px.values
    n      = len(prices)
    is_bull = np.zeros(n, dtype=bool)
    window  = min_days // 2

    troughs, peaks = [], []
    for i in range(window, n - window):
        window_slice = prices[max(0, i-window):i+window+1]
        if prices[i] == window_slice.min():
            troughs.append(i)
        if prices[i] == window_slice.max():
            peaks.append(i)

    for t_i in troughs:
        next_peaks = [p for p in peaks if p > t_i]
        if not next_peaks:
            continue
        p_i = next_peaks[0]
        duration = p_i - t_i
        cum_ret  = prices[p_i] / prices[t_i] - 1
        if duration >= min_days and cum_ret >= min_ret:
            is_bull[t_i:p_i+1] = True

    return pd.Series(is_bull, index=px.index)


def make_nber_labels(px: pd.Series,
                     usrec: pd.Series) -> pd.Series:
    """
    NBER-based labels:
      bear    = NBER recession days
      bull    = Pagan-Sossounov uptrends during expansions
      neutral = all other expansion days

    Parameters
    ----------
    px     : daily price series
    usrec  : daily binary NBER indicator (1=recession)

    Returns
    -------
    pd.Series of int (0=bear, 1=neutral, 2=bull), same index as px
    """
    usrec_aligned = usrec.reindex(px.index).ffill().fillna(0).astype(int)

    # Pagan-Sossounov on expansion price only
    px_expansion = px.copy().astype(float)
    px_expansion[usrec_aligned == 1] = np.nan
    px_expansion = px_expansion.ffill()
    ps_bull = pagan_sossounov_bull(px_expansion)

    labels = pd.Series("neutral", index=px.index)
    labels[usrec_aligned == 1]                          = "bear"
    labels[(usrec_aligned == 0) & ps_bull]              = "bull"

    return labels.map(LABEL_MAP).astype(int)


def make_drawdown_labels(px: pd.Series,
                          bear_threshold: float = DRAWDOWN_BEAR_THRESHOLD,
                          min_days:       int   = PAGAN_MIN_DAYS,
                          min_ret:        float = PAGAN_MIN_RET) -> pd.Series:
    """
    Drawdown-threshold labels:
      bear    = any episode where price falls >bear_threshold from its
                rolling peak (peak-to-trough drawdown)
      bull    = Pagan-Sossounov uptrends outside bear episodes
      neutral = all remaining days

    This scheme captures non-recessionary bear markets (e.g. 2022 -25%)
    that NBER labels miss.

    Parameters
    ----------
    px              : daily price series
    bear_threshold  : peak-to-trough drawdown to classify as bear (0.15 = 15%)

    Returns
    -------
    pd.Series of int (0=bear, 1=neutral, 2=bull)
    """
    px = px.astype(float)
    rolling_peak = px.expanding().max()
    drawdown     = (px / rolling_peak) - 1.0    # 0 to -1

    in_bear = drawdown <= -bear_threshold

    # Smooth: require bear episode to last at least 5 days
    # (avoids classifying single-day crashes as sustained bears)
    bear_smooth = in_bear.rolling(window=5, min_periods=1).sum() >= 3

    # Pagan-Sossounov bull on non-bear days
    px_non_bear = px.copy()
    px_non_bear[bear_smooth] = np.nan
    px_non_bear = px_non_bear.ffill()
    ps_bull = pagan_sossounov_bull(px_non_bear, min_days=min_days, min_ret=min_ret)

    labels = pd.Series("neutral", index=px.index)
    labels[bear_smooth]                         = "bear"
    labels[(~bear_smooth) & ps_bull]            = "bull"

    return labels.map(LABEL_MAP).astype(int)


def get_labels(px:           pd.Series,
               scheme:       str,
               usrec:        pd.Series = None,
               bear_threshold: float   = DRAWDOWN_BEAR_THRESHOLD) -> pd.Series:
    """
    Dispatcher: returns integer label Series for the given scheme.

    Parameters
    ----------
    px     : daily price series
    scheme : "nber" or "drawdown"
    usrec  : NBER indicator (required for scheme="nber")
    """
    if scheme not in LABEL_SCHEMES:
        raise ValueError(f"Unknown label scheme: {scheme}. "
                         f"Choose from {LABEL_SCHEMES}")

    if scheme == "nber":
        if usrec is None:
            raise ValueError("usrec required for nber labelling")
        return make_nber_labels(px, usrec)

    elif scheme == "drawdown":
        return make_drawdown_labels(px, bear_threshold=bear_threshold)


def label_summary(labels: pd.Series) -> dict:
    """Print and return label distribution statistics."""
    counts = labels.value_counts().sort_index()
    total  = len(labels)
    summary = {}
    for code, name in LABEL_MAP_INV.items():
        n = counts.get(code, 0)
        summary[name] = {"n": n, "pct": n/total*100}
        print(f"  {name:>8}: {n:>5} days ({n/total*100:.1f}%)")
    return summary
