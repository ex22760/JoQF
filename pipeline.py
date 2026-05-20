"""
pipeline.py
Master experiment function. Called by every experiment script.
Single entry point — no experiment duplicates logic.

Usage:
    from pipeline import run_experiment, ExperimentResult

    result = run_experiment(
        ticker       = "^GSPC",
        window       = ROLLING_WINDOWS[2],   # Window 3 (2019-2026)
        vol_spec     = "ewma_vix",
        label_scheme = "nber",
    )
    print(result.sharpe, result.max_dd, result.cagr)
"""

import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    BASE_DIR, RESULTS_DIR, DATA_DIR,
    GLOBAL_SEED, SEEDS, K_REGIMES,
    RF_MAX_DEPTH, RF_N_ESTIMATORS,
    LSTM_HIDDEN, LSTM_DROPOUT, LSTM_SEQ_LEN,
    LSTM_LR, LSTM_EPOCHS, LSTM_PATIENCE, LSTM_BATCH,
    CONFIDENCE_THRESHOLD, CONFLICT_SCALE, SOFTMAX_TEMP,
    RISK_AVERSION, TC, BAND_MULT, BAND_ROLL,
    BEAR_FLOOR, ALPHA_BEAR, ALPHA_BULL, WARMUP,
    # MOM_PERSIST imported dynamically inside function to allow runtime override
    LOOKBACK_LT, ANNUALIZATION, BASELINE_VOL_SPEC,
    BASELINE_LABEL_SCHEME,
)

from data.loader    import assemble_dataset, load_usrec
from data.features  import (build_features, build_r_daily,
                             build_no_trade_band, build_hjb_signal)
from data.labels    import get_labels
from volatility.factory import compute_sigma
from models.base    import (set_seed, forward_filter_proba, LSTMRegime,
                             RegimeDataset, macro_f1, portfolio_metrics,
                             softmax_weights, max_drawdown, sharpe_ratio, cagr)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble        import RandomForestClassifier
from sklearn.mixture         import GaussianMixture
from sklearn.preprocessing   import StandardScaler
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.metrics         import f1_score
from hmmlearn.hmm            import GaussianHMM


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    # Identifiers
    ticker:       str
    window_name:  str
    train_end:    str
    test_start:   str
    test_end:     str
    vol_spec:     str
    label_scheme: str

    # Portfolio performance
    sharpe:       float = np.nan
    cagr:         float = np.nan
    max_dd:       float = np.nan
    tc_bps:       float = np.nan

    # Baseline performance
    baseline_sharpe: float = np.nan
    baseline_cagr:   float = np.nan
    baseline_max_dd: float = np.nan

    # Classification
    ensemble_macro_f1:  float = np.nan
    ensemble_bear_recall: float = np.nan
    gmm_macro_f1:       float = np.nan
    hmm_macro_f1:       float = np.nan
    rf_macro_f1:        float = np.nan
    lstm_macro_f1:      float = np.nan

    # Ensemble weights (derived)
    ensemble_weights: Dict[str, float] = field(default_factory=dict)

    # Per-regime diagnostics
    bear_mean_weight:     float = np.nan
    neutral_mean_weight:  float = np.nan
    bull_mean_weight:     float = np.nan
    baseline_bear_weight: float = np.nan

    # Metadata
    n_test_days:  int   = 0
    n_bear_days:  int   = 0
    runtime_s:    float = np.nan

    # Daily series (for quiet bear analysis)
    daily_weights:           object = None  # pd.Series ensemble equity weight
    daily_baseline_weights:  object = None  # pd.Series baseline equity weight
    daily_labels:            object = None  # pd.Series regime labels (0/1/2)
    daily_sigma:             object = None  # pd.Series annualised vol
    daily_returns:           object = None  # pd.Series daily returns
    daily_ensemble_proba:    object = None  # np.ndarray (N,3) ensemble probabilities
    daily_equity_curve:      object = None  # pd.Series ensemble equity curve
    daily_baseline_curve:    object = None  # pd.Series baseline equity curve
    # Per-model probabilities for ablation study
    gmm_proba:               object = None  # np.ndarray (N,3)
    hmm_proba:               object = None  # np.ndarray (N,3)
    rf_proba:                object = None  # np.ndarray (N,3)
    lstm_proba:              object = None  # np.ndarray (N,3)
    ensemble_model_weights:  object = None  # dict {gmm, hmm, rf, lstm}


# ── Unsupervised training ─────────────────────────────────────────────────────

def _train_unsupervised(X_train: np.ndarray,
                         label_order: list) -> tuple:
    """
    Fit GMM and HMM on training features.
    Returns (gmm, hmm, gmm_col_order, hmm_col_order).
    Column orders map component indices to (0=bear, 1=neutral, 2=bull).
    """
    set_seed(GLOBAL_SEED)

    # GMM
    gmm = GaussianMixture(n_components=K_REGIMES, covariance_type="full",
                          random_state=GLOBAL_SEED, n_init=5, max_iter=200)
    gmm.fit(X_train)
    gmm_labels  = gmm.predict(X_train)
    gmm_means   = [X_train[gmm_labels == k, 0].mean() for k in range(K_REGIMES)]
    gmm_col_order = np.argsort(gmm_means)   # sorted by return: bear, neutral, bull

    # HMM
    hmm = GaussianHMM(n_components=K_REGIMES, covariance_type="full",
                       n_iter=500, random_state=GLOBAL_SEED)
    hmm.fit(X_train)
    hmm_labels  = hmm.predict(X_train)
    hmm_means   = [X_train[hmm_labels == k, 0].mean() for k in range(K_REGIMES)]
    hmm_col_order = np.argsort(hmm_means)

    return gmm, hmm, gmm_col_order.tolist(), hmm_col_order.tolist()


# ── Supervised training ───────────────────────────────────────────────────────

def _train_rf(X_train: np.ndarray, y_train: np.ndarray,
              seed: int = GLOBAL_SEED) -> CalibratedClassifierCV:
    rf_base = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=seed,
    )
    rf_cal = CalibratedClassifierCV(rf_base, method="sigmoid", cv=5)
    rf_cal.fit(X_train, y_train)
    return rf_cal


def _train_lstm(X_train: np.ndarray, y_train: np.ndarray,
                X_val:   np.ndarray, y_val:   np.ndarray,
                seed:    int = GLOBAL_SEED) -> LSTMRegime:
    """Train LSTM with early stopping on validation macro-F1."""
    set_seed(seed)
    torch.manual_seed(seed)

    ds     = RegimeDataset(X_train, y_train, LSTM_SEQ_LEN)
    loader = DataLoader(ds, batch_size=LSTM_BATCH, shuffle=False)
    model  = LSTMRegime(X_train.shape[1], LSTM_HIDDEN,
                        dropout=LSTM_DROPOUT).to("cpu")
    opt    = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-5)
    crit   = nn.CrossEntropyLoss()
    Xv     = torch.tensor(X_val.astype(np.float32))

    best_f1, patience_cnt = 0.0, 0
    best_state = None

    for epoch in range(LSTM_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Evaluate
        model.eval()
        preds = np.full(len(X_val), 1, dtype=int)
        with torch.no_grad():
            for t in range(LSTM_SEQ_LEN, len(X_val)):
                seq = Xv[t-LSTM_SEQ_LEN:t].unsqueeze(0)
                preds[t] = model(seq).argmax(dim=1).item()
        val_f1 = macro_f1(y_val, preds)

        if val_f1 > best_f1:
            best_f1      = val_f1
            patience_cnt = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= LSTM_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _lstm_predict_proba(model: LSTMRegime,
                         X: np.ndarray) -> np.ndarray:
    """Causal LSTM probability predictions. Shape: (n, K_REGIMES)."""
    model.eval()
    proba = np.full((len(X), K_REGIMES), 1.0 / K_REGIMES)
    Xt    = torch.tensor(X.astype(np.float32))
    with torch.no_grad():
        for t in range(LSTM_SEQ_LEN, len(X)):
            seq = Xt[t-LSTM_SEQ_LEN:t].unsqueeze(0)
            proba[t] = torch.softmax(model(seq), dim=-1).numpy()[0]
    return proba


# ── Ensemble combination ──────────────────────────────────────────────────────

def _ensemble_probas(gmm_p, hmm_p, rf_p, lstm_p, weights):
    return (weights["gmm"]  * gmm_p
          + weights["hmm"]  * hmm_p
          + weights["rf"]   * rf_p
          + weights["lstm"] * lstm_p)


def _conflict_score(gmm_p, hmm_p, rf_p, lstm_p):
    hard     = np.stack([gmm_p.argmax(1), hmm_p.argmax(1),
                          rf_p.argmax(1),  lstm_p.argmax(1)], axis=1)
    majority = np.array([np.bincount(r.astype(int), minlength=K_REGIMES).argmax()
                          for r in hard])
    return 1.0 - (hard == majority[:, None]).mean(axis=1)


# ── Backtest ──────────────────────────────────────────────────────────────────

def _run_backtest(ens_proba:   np.ndarray,
                   conf:        np.ndarray,
                   conflict:    np.ndarray,
                   w_hjb:       np.ndarray,
                   base_band:   np.ndarray,
                   ret:         np.ndarray,
                   r_daily:     np.ndarray,
                   sigma_ann:   np.ndarray,
                   sigma_median: float,
                   muann_arr:   np.ndarray = None) -> tuple:
    """
    Run the dynamic allocation backtest.
    Returns (equity_curve, weights_actual, tc_paid_total_bps).
    """
    p_bear = ens_proba[:, 0]
    p_neut = ens_proba[:, 1]
    p_bull = ens_proba[:, 2]

    vol_ratio = np.clip(sigma_ann / (sigma_median + 1e-9), 0.5, 2.0)
    trend     = np.clip(w_hjb * 0.2, 0, 0.2)   # simplified momentum overlay

    w_mom  = np.clip(w_hjb * (1 + 0.2 * np.maximum(np.sign(w_hjb - 0.5), 0)), 0, 1)
    w_vol  = np.clip(w_hjb / vol_ratio, 0, 1)
    w_def  = np.full(len(w_hjb), BEAR_FLOOR)

    # Regime-conditional weight
    w_regime = np.clip(
        w_hjb * (1 + ALPHA_BULL * p_bull - ALPHA_BEAR * p_bear), 0, 1
    )

    # Strategy blend (confidence-weighted)
    w_blended_pb = p_bull * w_mom + p_neut * w_vol + p_bear * w_def
    w_final = np.clip(conf * w_blended_pb + (1 - conf) * w_hjb, 0, 1)

    # ── Momentum persistence filter ───────────────────────────────────────
    # During confirmed bear regimes (p_bear > 0.5), block any increase in
    # equity weight unless short-term momentum has recovered above threshold.
    # This prevents confidence collapse from triggering premature re-entry.
    # Read config dynamically so runtime overrides in experiment scripts work
    import config as _cfg
    _mom_thresh = _cfg.MOM_PERSIST_THRESHOLD
    _mom_win    = _cfg.MOM_PERSIST_WINDOW

    if _mom_thresh > 0.0 and muann_arr is not None:
        # Rolling short-term momentum: mean return over _mom_win days
        mom_st = np.full(len(w_hjb), np.nan)
        for t in range(_mom_win, len(w_hjb)):
            mom_st[t] = float(np.mean(muann_arr[t-_mom_win:t]))
        # Normalise by absolute max momentum for comparability
        mom_abs_max = float(np.nanpercentile(np.abs(mom_st), 95)) + 1e-8
        mom_norm = mom_st / mom_abs_max  # range roughly [-1, 1]

        # In bear regime: use a rolling bear state rather than instantaneous
        # p_bear threshold. On quiet bear days p_bear is LOW (confidence
        # has collapsed) so p_bear > 0.5 never fires. Instead track whether
        # the ensemble has been in a bear classification within the last
        # MOM_PERSIST_WINDOW days (sustained bear state).
        bear_class = p_bear > 0.3   # lower threshold — any bear signal
        # Rolling: sustained bear = bear signal in last N days
        sustained_bear = np.zeros(len(w_hjb), dtype=bool)
        for t in range(len(w_hjb)):
            start = max(0, t - _mom_win)
            sustained_bear[t] = bear_class[start:t+1].any()
        mom_insufficient = mom_norm < _mom_thresh
        filter_active = sustained_bear & mom_insufficient

        # For filtered days: cap w_final at its previous value
        # (implement as rolling cap during bear with insufficient momentum)
        w_filtered = w_final.copy()
        prev_w = w_hjb[0]  # initialise at baseline
        for t in range(len(w_filtered)):
            if filter_active[t] and not np.isnan(mom_norm[t]):
                # Do not allow increase above previous day's weight
                w_filtered[t] = min(w_filtered[t], prev_w)
            prev_w = w_filtered[t]
        w_final = w_filtered

    # Conflict-adjusted band
    conf_band = base_band * (1 + CONFLICT_SCALE * conflict)
    conf_band = np.nan_to_num(conf_band, nan=float(np.nanmedian(base_band)))

    # Simulate
    n       = len(ret)
    w_curr  = 0.0
    equity  = 1.0
    eq_list = []
    wt_list = []
    tc_paid = 0.0

    for t in range(n):
        diff = w_final[t] - w_curr
        if abs(diff) > conf_band[t]:
            new_w   = float(np.clip(w_curr + np.sign(diff) * conf_band[t], 0, 1))
            cost    = TC * abs(new_w - w_curr)
            tc_paid += cost
            equity  *= (1 - cost)
            w_curr   = new_w

        port_ret = w_curr * ret[t] + (1 - w_curr) * r_daily[t]
        equity  *= (1 + port_ret)
        eq_list.append(equity)
        wt_list.append(w_curr)

    tc_bps_annual = tc_paid / (n / ANNUALIZATION) * 1e4
    return np.array(eq_list), np.array(wt_list), tc_bps_annual


# ── Master pipeline ───────────────────────────────────────────────────────────

def run_experiment(ticker:       str,
                   window:       dict,
                   vol_spec:     str  = BASELINE_VOL_SPEC,
                   label_scheme: str  = BASELINE_LABEL_SCHEME,
                   save_result:  bool = True,
                   verbose:      bool = True,
                   gate_fn:      object = None) -> ExperimentResult:
    """
    Run a complete experiment for one (ticker, window, vol_spec, label_scheme).

    Parameters
    ----------
    ticker       : Yahoo Finance ticker (e.g. "^GSPC")
    window       : dict from config.ROLLING_WINDOWS
    vol_spec     : one of config.VOL_SPECS
    label_scheme : one of config.LABEL_SCHEMES
    save_result  : save ExperimentResult to RESULTS_DIR
    verbose      : print progress

    Returns
    -------
    ExperimentResult dataclass
    """
    t_start = time.time()
    set_seed(GLOBAL_SEED)

    wname      = window["name"]
    train_end  = window["train_end"]
    val_start  = window["val_start"]
    test_start = window["test_start"]
    test_end   = window["test_end"]

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {ticker} | {wname} | vol={vol_spec} | labels={label_scheme}")
        print(f"  Train: {window['train_start']} -> {train_end}")
        print(f"  Test:  {test_start} -> {test_end}")
        print(f"{'='*70}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    # Always download full history — slicing to window done below
    # This prevents stale cache across different window experiments
    start_dl = "1988-01-01"
    end_dl   = "2026-06-01"
    df    = assemble_dataset(ticker, start=start_dl, end=end_dl)
    usrec = load_usrec(start=start_dl, end=end_dl)

    px     = df["Close"].astype(float)
    logret = df["LogReturn"]
    ret    = df["Return"]
    vix    = df["VIX"] if "VIX" in df.columns else None
    r_daily = build_r_daily(df)

    # ── 2. Volatility ─────────────────────────────────────────────────────────
    if verbose: print("  Computing volatility...")
    sigma_ann = compute_sigma(logret, vix=vix, spec=vol_spec,
                               train_end=train_end)

    # ── 3. Features ───────────────────────────────────────────────────────────
    if verbose: print("  Building features...")
    X_df = build_features(df, sigma_ann)

    # ── 4. Labels ─────────────────────────────────────────────────────────────
    if verbose: print(f"  Building labels ({label_scheme})...")
    y_all = get_labels(px, scheme=label_scheme, usrec=usrec)
    y_all = y_all.reindex(X_df.index)

    # ── 5. Splits ─────────────────────────────────────────────────────────────
    X_preval_raw = X_df.loc[:val_start]
    X_val_raw    = X_df.loc[val_start:train_end]
    X_train_raw  = X_df.loc[:train_end]
    X_test_raw   = X_df.loc[test_start:test_end]

    scaler_pv = StandardScaler()
    X_pv  = scaler_pv.fit_transform(X_preval_raw)
    X_val = scaler_pv.transform(X_val_raw)

    scaler   = StandardScaler()
    X_train  = scaler.fit_transform(X_train_raw)
    X_test   = scaler.transform(X_test_raw)

    y_pv    = y_all.reindex(X_preval_raw.index).fillna(1).astype(int).values
    y_val_a = y_all.reindex(X_val_raw.index).fillna(1).astype(int).values
    y_train = y_all.reindex(X_train_raw.index).fillna(1).astype(int).values
    y_test  = y_all.reindex(X_test_raw.index).fillna(1).astype(int).values

    # ── 6. Train unsupervised models ──────────────────────────────────────────
    if verbose: print("  Training GMM + HMM...")
    gmm, hmm, gmm_ord, hmm_ord = _train_unsupervised(X_pv, [0,1,2])

    def get_stacked(X_sc):
        gp = gmm.predict_proba(X_sc)[:, gmm_ord]
        hp = forward_filter_proba(hmm, X_sc)[:, hmm_ord]
        return np.hstack([X_sc, gp, hp])

    X_pv_s   = get_stacked(X_pv)
    X_val_s  = get_stacked(X_val)
    X_tr_s   = get_stacked(X_train)
    X_te_s   = get_stacked(X_test)

    # ── 7. Train supervised models (pre-val data, evaluated on val) ───────────
    if verbose: print("  Training RF + LSTM (pre-val split, no leakage)...")
    rf_pv   = _train_rf(X_pv_s, y_pv, seed=GLOBAL_SEED)
    lstm_pv = _train_lstm(X_pv_s, y_pv, X_val_s, y_val_a, seed=GLOBAL_SEED)

    # Validation macro-F1 for softmax weights
    gmm_val_pred  = gmm.predict_proba(X_val)[:, gmm_ord].argmax(axis=1)
    hmm_val_pred  = forward_filter_proba(hmm, X_val)[:, hmm_ord].argmax(axis=1)
    rf_val_pred   = rf_pv.predict(X_val_s)
    lstm_val_pred = _lstm_predict_proba(lstm_pv, X_val_s).argmax(axis=1)

    val_f1s = {
        "gmm":  macro_f1(y_val_a, gmm_val_pred),
        "hmm":  macro_f1(y_val_a, hmm_val_pred),
        "rf":   macro_f1(y_val_a, rf_val_pred),
        "lstm": macro_f1(y_val_a, lstm_val_pred),
    }
    weights = softmax_weights(val_f1s, temperature=SOFTMAX_TEMP)
    if verbose:
        print(f"  Ensemble weights: " +
              ", ".join(f"{k}={v:.3f}" for k,v in weights.items()))

    # ── 8. Retrain supervised on full train set for test-set inference ────────
    if verbose: print("  Retraining RF + LSTM on full training set...")
    rf_full   = _train_rf(X_tr_s, y_train, seed=GLOBAL_SEED)
    lstm_full = _train_lstm(X_tr_s, y_train, X_val_s, y_val_a, seed=GLOBAL_SEED)

    # ── 9. Test-set probabilities ─────────────────────────────────────────────
    if verbose: print("  Computing test-set ensemble probabilities...")
    gp_te   = gmm.predict_proba(X_test)[:, gmm_ord]
    hp_te   = forward_filter_proba(hmm, X_test)[:, hmm_ord]
    rf_p_te = rf_full.predict_proba(X_te_s)
    lp_te   = _lstm_predict_proba(lstm_full, X_te_s)

    ens_proba = _ensemble_probas(gp_te, hp_te, rf_p_te, lp_te, weights)
    conf      = ens_proba.max(axis=1)
    conflict  = _conflict_score(gp_te, hp_te, rf_p_te, lp_te)

    # Confidence fallback
    low_conf = conf < CONFIDENCE_THRESHOLD
    if low_conf.any():
        equal_ens = 0.25*(gp_te + hp_te + rf_p_te + lp_te)
        ens_proba[low_conf] = equal_ens[low_conf]

    # ── 10. HJB baseline ──────────────────────────────────────────────────────
    sigma_test = sigma_ann.reindex(X_test_raw.index).values
    r_d_test   = r_daily.reindex(X_test_raw.index).values
    ret_test   = ret.reindex(X_test_raw.index).fillna(0).values

    u_star     = build_hjb_signal(
        logret.reindex(X_test_raw.index),
        sigma_ann.reindex(X_test_raw.index),
        r_daily.reindex(X_test_raw.index),
    )
    w_hjb_test = u_star.shift(1).fillna(0).values

    sigma_med  = float(sigma_ann.reindex(X_train_raw.index).median())
    base_band  = build_no_trade_band(
        sigma_ann, TC, BAND_MULT, BAND_ROLL, WARMUP
    ).reindex(X_test_raw.index).values

    # Baseline backtest (no regime signal)
    b_eq, b_wt, b_tc = _run_backtest(
        np.column_stack([np.zeros(len(X_test_raw)),
                         np.ones(len(X_test_raw)),
                         np.zeros(len(X_test_raw))]),  # all neutral
        np.ones(len(X_test_raw)),
        np.zeros(len(X_test_raw)),
        w_hjb_test, base_band, ret_test, r_d_test,
        sigma_test, sigma_med,
    )

    # Ensemble backtest
    e_eq, e_wt, e_tc = _run_backtest(
        ens_proba, conf, conflict,
        w_hjb_test, base_band, ret_test, r_d_test,
        sigma_test, sigma_med,
        muann_arr=X_test_raw["momentum"].values if "momentum" in X_test_raw.columns else None,
    )

    # Apply optional momentum gate post-backtest
    if gate_fn is not None:
        muann_test = X_test_raw["momentum"].values if "momentum" in X_test_raw.columns else None
        if muann_test is not None:
            p_bear_test = ens_proba[:, 0]
            if verbose:
                print("  Applying momentum persistence gate...")
            e_wt_gated = gate_fn(e_wt, p_bear_test, muann_test)
            # Recompute equity curve from gated weights
            e_eq_gated = np.ones(len(e_wt_gated))
            for t in range(len(e_wt_gated)):
                port_r = (e_wt_gated[t] * ret_test[t]
                          + (1 - e_wt_gated[t]) * r_d_test[t])
                e_eq_gated[t] = (e_eq_gated[t-1] if t > 0 else 1.0) * (1 + port_r)
            e_wt = e_wt_gated
            e_eq = e_eq_gated

    test_idx = X_test_raw.index
    e_equity = pd.Series(e_eq, index=test_idx)
    b_equity = pd.Series(b_eq, index=test_idx)

    # ── 11. Classification metrics ────────────────────────────────────────────
    ens_pred = ens_proba.argmax(axis=1)
    from sklearn.metrics import recall_score
    bear_recall = float(recall_score(y_test, ens_pred,
                                      labels=[0], average="macro",
                                      zero_division=0))

    # ── 12. Per-regime weight diagnostics ─────────────────────────────────────
    e_wt_s = pd.Series(e_wt, index=test_idx)
    b_wt_s = pd.Series(b_wt, index=test_idx)
    y_test_s = pd.Series(y_test, index=test_idx)

    def mean_wt_regime(wt, regime_code):
        mask = y_test_s == regime_code
        return float(wt[mask].mean()) if mask.any() else np.nan

    # ── 13. Assemble result ───────────────────────────────────────────────────
    em = portfolio_metrics(e_equity)
    bm = portfolio_metrics(b_equity)

    result = ExperimentResult(
        ticker       = ticker,
        window_name  = wname,
        train_end    = train_end,
        test_start   = test_start,
        test_end     = test_end,
        vol_spec     = vol_spec,
        label_scheme = label_scheme,

        sharpe   = em["sharpe"],
        cagr     = em["cagr"],
        max_dd   = em["max_dd"],
        tc_bps   = e_tc,

        baseline_sharpe = bm["sharpe"],
        baseline_cagr   = bm["cagr"],
        baseline_max_dd = bm["max_dd"],

        ensemble_macro_f1    = macro_f1(y_test, ens_pred),
        ensemble_bear_recall = bear_recall,
        gmm_macro_f1  = val_f1s["gmm"],
        hmm_macro_f1  = val_f1s["hmm"],
        rf_macro_f1   = val_f1s["rf"],
        lstm_macro_f1 = val_f1s["lstm"],

        ensemble_weights = weights,

        bear_mean_weight     = mean_wt_regime(e_wt_s, 0),
        neutral_mean_weight  = mean_wt_regime(e_wt_s, 1),
        bull_mean_weight     = mean_wt_regime(e_wt_s, 2),
        baseline_bear_weight = mean_wt_regime(b_wt_s, 0),

        n_test_days  = len(X_test_raw),
        n_bear_days  = int((y_test_s == 0).sum()),
        runtime_s    = time.time() - t_start,

        # Daily series for quiet bear analysis
        daily_weights          = e_wt_s,
        daily_baseline_weights = b_wt_s,
        daily_labels           = y_test_s,
        daily_sigma            = pd.Series(sigma_test, index=test_idx),
        daily_returns          = ret.reindex(test_idx).fillna(0),
        daily_ensemble_proba   = ens_proba,  # (N,3) bear/neutral/bull
        daily_equity_curve     = e_equity,
        daily_baseline_curve   = b_equity,
        # Per-model probabilities for ablation study
        gmm_proba              = gp_te,
        hmm_proba              = hp_te,
        rf_proba               = rf_p_te,
        lstm_proba             = lp_te,
        ensemble_model_weights = weights,
    )

    if verbose:
        print(f"\n  RESULT: Sharpe={result.sharpe:.3f} (vs baseline {result.baseline_sharpe:.3f})"
              f"  MaxDD={result.max_dd:.2%} (vs {result.baseline_max_dd:.2%})"
              f"  CAGR={result.cagr:.2%}")
        print(f"  Bear weight: {result.bear_mean_weight:.3f} "
              f"(vs baseline {result.baseline_bear_weight:.3f}, "
              f"{(1-result.bear_mean_weight/result.baseline_bear_weight)*100:.0f}% reduction)"
              if result.baseline_bear_weight and not np.isnan(result.baseline_bear_weight)
              else f"  Bear weight: {result.bear_mean_weight} "
              f"(vs baseline {result.baseline_bear_weight}, no bear days)")
        print(f"  Runtime: {result.runtime_s:.0f}s")

    # ── 14. Save ──────────────────────────────────────────────────────────────
    if save_result:
        fname = (f"{ticker.replace('^','')}_"
                 f"{wname}_{vol_spec}_{label_scheme}.pkl")
        fpath = os.path.join(RESULTS_DIR, fname)
        with open(fpath, "wb") as f:
            pickle.dump(result, f)
        if verbose:
            print(f"  Saved: {fpath}")

    return result
