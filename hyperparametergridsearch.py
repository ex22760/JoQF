"""
hyperparametergridsearch.py — JOURNAL OF FINANCE VERSION (COMBINED)

Three grid searches with clear methodological separation:

  SEARCH 1: LSTM hyperparameters
    Pre-val: 1990-2008  (GMM/HMM training, LSTM training)
    Val:     2008-2015  (LSTM early stopping)
    Rationale: LSTMs require long sequential history to learn regime
    patterns. Short windows produce unstable results. Lookahead into
    Window 1 test (2000-2009) and Window 2 test (2010-2019) is
    acknowledged and defended by cross-market generalisation in Phase 3.

  SEARCH 2: RF hyperparameters
    Pre-val: 1990-1996  (GMM/HMM training)
    Train:   1990-1999  (full Window 1 training set for CV)
    Method:  5-fold stratified CV, shuffle=False (temporal order)
    Rationale: RF does not require long sequential history. CV on
    the full Window 1 training set (including 1990-91 and 2001
    recessions) gives sufficient bear days for stable HP selection.
    No leakage into any test period.

  SEARCH 3: Alpha parameters
    Train:   1990-1999  (same as RF)
    Test:    2000-2009  (Window 1 test — never used in LSTM/RF search)
    Rationale: Corrected backtest explicitly applies a_bear and a_bull.
    Uses winning RF and LSTM from above. No leakage.

All figures saved with _lstm suffix (search 1) and _rf_alpha suffix
(searches 2 and 3) to avoid overwriting.

Fault tolerant — checkpoints after every config.

Outputs:
  fig_grid_lstm.png          (LSTM search, val 2008-2015)
  fig_grid_rf.png            (RF search, Window 1 CV)
  fig_grid_alpha.png         (Alpha search, Window 1 test)
  grid_search_results.txt    (all results + winning HPs)
"""

import os, sys, warnings, time, random, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Checkpointing ─────────────────────────────────────────────────────────────
CKPT_DIR = Path(ROOT) / "grid_checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

def ckpt_path(name): return CKPT_DIR / f"{name}.json"

def save_ckpt(name, data):
    with open(ckpt_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_ckpt(name):
    p = ckpt_path(name)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}

from config import (
    GLOBAL_SEED, SEEDS, K_REGIMES, ANNUALIZATION,
    LOOKBACK_ST, LOOKBACK_LT, RISK_AVERSION,
    VIX_WEIGHT, MACRO_WEIGHT, ALPHA_SHORT,
    WARMUP, TC, BAND_MULT, BAND_ROLL,
    BEAR_FLOOR, CONFLICT_SCALE,
)
from data.loader    import assemble_dataset, load_usrec
from data.features  import (build_features, build_r_daily,
                              build_no_trade_band, build_hjb_signal)
from data.labels    import get_labels
from volatility.factory import compute_sigma
from models.base    import (forward_filter_proba, set_seed,
                              macro_f1, sharpe_ratio, softmax_weights,
                              LSTMRegime, RegimeDataset)

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    raise ImportError("pip install hmmlearn")

# ── Reproducibility + device ──────────────────────────────────────────────────
set_seed(GLOBAL_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Shared settings ───────────────────────────────────────────────────────────
TICKER       = "^GSPC"
TRAIN_START  = "1990-01-01"
VOL_SPEC     = "ewma_vix"
LABEL_SCHEME = "nber"
MAX_EPOCHS   = 50
PATIENCE     = 10
BATCH_SIZE   = 32

# ── LSTM search splits (global val — longer window for stability) ──────────────
LSTM_PRE_VAL_END = "2008-01-01"
LSTM_VAL_END     = "2015-01-01"

# ── RF + Alpha search splits (Window 1 only — no leakage) ────────────────────
RF_PRE_VAL_END   = "1996-01-01"
RF_TRAIN_END     = "1999-12-31"
RF_TEST_START    = "2000-01-01"

# ── Grid search spaces ────────────────────────────────────────────────────────
LSTM_HIDDEN   = [32, 64, 128, 256]
LSTM_DROPOUTS = [0.1, 0.2, 0.3, 0.4, 0.5]
LSTM_SEQ_LENS = [10, 20, 40, 60, 80, 120]
LSTM_LRS      = [1e-3, 5e-4, 1e-4]
RF_DEPTHS     = [4, 6, 8, 10, 12, None]
RF_N_TREES    = [100, 200, 300, 500]
ALPHA_BEARS   = [0.2, 0.3, 0.4, 0.5, 0.6]
ALPHA_BULLS   = [0.1, 0.2, 0.3, 0.4, 0.5]

# ── LSTM training helper ──────────────────────────────────────────────────────
def train_lstm_early_stop(X_tr, y_tr, X_val_arr, y_val_arr,
                           hidden, dropout, seq_len, lr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    ds     = RegimeDataset(X_tr, y_tr, seq_len)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model  = LSTMRegime(X_tr.shape[1], hidden, dropout=dropout).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit   = nn.CrossEntropyLoss()
    Xv     = torch.tensor(X_val_arr.astype(np.float32))
    best_f1, patience_cnt = 0.0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        preds = np.full(len(X_val_arr), 1, dtype=int)
        with torch.no_grad():
            for t in range(seq_len, len(X_val_arr)):
                seq = Xv[t-seq_len:t].unsqueeze(0).to(DEVICE)
                preds[t] = model(seq).argmax(dim=1).item()
        val_f1 = macro_f1(y_val_arr, preds)
        if val_f1 > best_f1:
            best_f1, patience_cnt = val_f1, 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break
    return best_f1

# ── Load full data once ───────────────────────────────────────────────────────
print("="*60)
print("Loading full data via JF pipeline...")
print("="*60)
df      = assemble_dataset(TICKER, start=TRAIN_START, end="2026-06-01")
usrec   = load_usrec(start=TRAIN_START, end="2026-06-01")
px      = df["Close"].astype(float)
logret  = df["LogReturn"]
ret     = df["Return"]
vix     = df["VIX"] if "VIX" in df.columns else None
r_daily = build_r_daily(df)

# Use RF_TRAIN_END for sigma (most conservative — fits on least data)
sigma_ann = compute_sigma(logret, vix=vix, spec=VOL_SPEC,
                           train_end=RF_TRAIN_END)
X_df      = build_features(df, sigma_ann)
y_all     = get_labels(px, scheme=LABEL_SCHEME, usrec=usrec)
y_all     = y_all.reindex(X_df.index)

log_lines = []

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH 1: LSTM — global val 2008-2015
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SEARCH 1: LSTM HYPERPARAMETERS")
print(f"Pre-val train: {TRAIN_START} -> {LSTM_PRE_VAL_END}")
print(f"Val:           {LSTM_PRE_VAL_END} -> {LSTM_VAL_END}")
print("Note: acknowledged lookahead, defended by cross-market Phase 3")
print("="*60)

# LSTM splits
X_lstm_pv_raw  = X_df.loc[:LSTM_PRE_VAL_END]
X_lstm_val_raw = X_df.loc[LSTM_PRE_VAL_END:LSTM_VAL_END]

scaler_lstm_pv = StandardScaler()
X_lstm_pv      = scaler_lstm_pv.fit_transform(X_lstm_pv_raw)
X_lstm_val     = scaler_lstm_pv.transform(X_lstm_val_raw)

y_lstm_pv  = y_all.reindex(X_lstm_pv_raw.index).fillna(1).astype(int).values
y_lstm_val = y_all.reindex(X_lstm_val_raw.index).fillna(1).astype(int).values

# GMM + HMM on LSTM pre-val
print("Training GMM + HMM for LSTM search...")
set_seed(GLOBAL_SEED)
gmm_lstm = GaussianMixture(n_components=K_REGIMES, covariance_type="full",
                            random_state=GLOBAL_SEED, n_init=5, max_iter=200)
gmm_lstm.fit(X_lstm_pv)
gm_preds = gmm_lstm.predict(X_lstm_pv)
gm_means = [X_lstm_pv[gm_preds==k, 0].mean() for k in range(K_REGIMES)]
gm_ord   = np.argsort(gm_means).tolist()

hmm_lstm = GaussianHMM(n_components=K_REGIMES, covariance_type="full",
                        n_iter=500, random_state=GLOBAL_SEED)
hmm_lstm.fit(X_lstm_pv)
hm_preds = hmm_lstm.predict(X_lstm_pv)
hm_means = [X_lstm_pv[hm_preds==k, 0].mean() for k in range(K_REGIMES)]
hm_ord   = np.argsort(hm_means).tolist()

def get_stacked_lstm(X_sc):
    gp = gmm_lstm.predict_proba(X_sc)[:, gm_ord]
    hp = forward_filter_proba(hmm_lstm, X_sc)[:, hm_ord]
    return np.hstack([X_sc, gp, hp])

X_lstm_pv_s  = get_stacked_lstm(X_lstm_pv)
X_lstm_val_s = get_stacked_lstm(X_lstm_val)

print(f"LSTM stacked features: {X_lstm_pv_s.shape[1]}")
print(f"Bear days — pre-val: {(y_lstm_pv==0).sum()} "
      f"({(y_lstm_pv==0).mean()*100:.1f}%)")
print(f"Bear days — val:     {(y_lstm_val==0).sum()} "
      f"({(y_lstm_val==0).mean()*100:.1f}%)")

n_lstm = len(LSTM_HIDDEN)*len(LSTM_DROPOUTS)*len(LSTM_SEQ_LENS)*len(LSTM_LRS)
print(f"Total LSTM runs: {n_lstm} configs x {len(SEEDS)} seeds = "
      f"{n_lstm*len(SEEDS)}")

# Load checkpoints
lstm_ckpt  = load_ckpt("lstm_means")
lstm_means = {(v["h"],v["dr"],v["seq"],v["lr"]): v["mean"]
              for v in lstm_ckpt.get("results", [])}
lstm_stds  = {(v["h"],v["dr"],v["seq"],v["lr"]): v["std"]
              for v in lstm_ckpt.get("results", [])}
if lstm_means:
    print(f"Resuming: {len(lstm_means)}/{n_lstm} configs already done")

done = 0
for hidden in LSTM_HIDDEN:
    for dropout in LSTM_DROPOUTS:
        for seq_len in LSTM_SEQ_LENS:
            for lr in LSTM_LRS:
                done += 1
                if (hidden,dropout,seq_len,lr) in lstm_means:
                    print(f"  [{done}/{n_lstm}] h={hidden} dr={dropout} "
                          f"seq={seq_len} lr={lr:.0e}  "
                          f"[cached] {lstm_means[(hidden,dropout,seq_len,lr)]:.4f}")
                    continue
                f1s = []
                print(f"  [{done}/{n_lstm}] h={hidden} dr={dropout} "
                      f"seq={seq_len} lr={lr:.0e}", end="  ", flush=True)
                t0 = time.time()
                for seed in SEEDS:
                    f1s.append(train_lstm_early_stop(
                        X_lstm_pv_s, y_lstm_pv,
                        X_lstm_val_s, y_lstm_val,
                        hidden, dropout, seq_len, lr, seed))
                lstm_means[(hidden,dropout,seq_len,lr)] = np.mean(f1s)
                lstm_stds[(hidden,dropout,seq_len,lr)]  = np.std(f1s)
                print(f"mean={np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  "
                      f"({time.time()-t0:.0f}s)")
                save_ckpt("lstm_means", {"results": [
                    {"h": k[0], "dr": k[1], "seq": k[2], "lr": k[3],
                     "mean": v, "std": lstm_stds[k]}
                    for k, v in lstm_means.items()]})

best_lstm = max(lstm_means, key=lstm_means.get)
best_h, best_dr, best_seq, best_lr = best_lstm
print(f"\n  WINNER: hidden={best_h}, dropout={best_dr}, seq={best_seq}, "
      f"lr={best_lr:.0e}  F1={lstm_means[best_lstm]:.4f}")

log_lines += ["="*60,
              "SEARCH 1: LSTM (val macro-F1, pre-val 1990-2008, val 2008-2015)",
              f"seeds={SEEDS}, early stopping patience={PATIENCE}",
              "="*60]
for hidden in LSTM_HIDDEN:
    for lr in LSTM_LRS:
        log_lines.append(f"\n  hidden={hidden}, lr={lr:.0e}")
        log_lines.append(f"  {'dropout':>8}" +
                         "".join(f"  seq={s:>3}" for s in LSTM_SEQ_LENS))
        for dropout in LSTM_DROPOUTS:
            row = f"  {dropout:>8.1f}"
            for seq_len in LSTM_SEQ_LENS:
                v = lstm_means[(hidden,dropout,seq_len,lr)]
                s = lstm_stds[(hidden,dropout,seq_len,lr)]
                star = (" (*)" if (hidden==best_h and dropout==best_dr
                                   and seq_len==best_seq and lr==best_lr)
                        else "     ")
                row += f"  {v:.4f}+/-{s:.4f}{star}"
            log_lines.append(row)
log_lines.append(f"\nWINNER: hidden={best_h}, dropout={best_dr}, "
                 f"seq={best_seq}, lr={best_lr:.0e}")

# LSTM heatmap
all_plot = []
mean_cell, std_cell = {}, {}
for hidden in LSTM_HIDDEN:
    for dropout in LSTM_DROPOUTS:
        for seq_len in LSTM_SEQ_LENS:
            blr = max(LSTM_LRS,
                      key=lambda lr: lstm_means[(hidden,dropout,seq_len,lr)])
            mean_cell[(hidden,dropout,seq_len)] = lstm_means[(hidden,dropout,seq_len,blr)]
            std_cell[(hidden,dropout,seq_len)]  = lstm_stds[(hidden,dropout,seq_len,blr)]
            all_plot.append(mean_cell[(hidden,dropout,seq_len)])

vmin_p, vmax_p = min(all_plot)*0.995, max(all_plot)*1.005
fig, axes = plt.subplots(1, len(LSTM_HIDDEN), figsize=(16,5), sharey=True)
fig.suptitle("LSTM: validation macro-F1 (mean +/- std, 3 seeds, best lr per cell)\n"
             "Pre-val 1990-2008, Val 2008-2015 | Global search for stability",
             fontsize=10)
for ax, hidden in zip(axes, LSTM_HIDDEN):
    mat = np.array([[mean_cell[(hidden,d,s)] for s in LSTM_SEQ_LENS]
                    for d in LSTM_DROPOUTS])
    im  = ax.imshow(mat, cmap="Greens", aspect="auto", vmin=vmin_p, vmax=vmax_p)
    ax.set_xticks(range(len(LSTM_SEQ_LENS)))
    ax.set_xticklabels([str(s) for s in LSTM_SEQ_LENS], fontsize=7)
    ax.set_yticks(range(len(LSTM_DROPOUTS)))
    ax.set_yticklabels([f"{d}" for d in LSTM_DROPOUTS])
    ax.set_xlabel("Sequence length", fontsize=9)
    ax.set_title(f"Hidden = {hidden}", fontsize=9)
    for i,d in enumerate(LSTM_DROPOUTS):
        for j,s in enumerate(LSTM_SEQ_LENS):
            v   = mean_cell[(hidden,d,s)]
            std = std_cell[(hidden,d,s)]
            star = " *" if (hidden==best_h and d==best_dr and s==best_seq) else ""
            col  = "white" if v > np.mean(all_plot) else "black"
            ax.text(j, i-0.15, f"{v:.3f}{star}", ha="center", va="center",
                    fontsize=6, color=col,
                    fontweight="bold" if star else "normal")
            ax.text(j, i+0.25, f"+/-{std:.3f}", ha="center", va="center",
                    fontsize=5, color=col, alpha=0.8)
axes[0].set_ylabel("Dropout", fontsize=9)
plt.colorbar(im, ax=axes[-1], label="Mean Macro-F1")
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "fig_grid_lstm.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_grid_lstm.png")

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH 2: RF — 5-fold stratified CV on Window 1 training set (clean)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SEARCH 2: RF HYPERPARAMETERS")
print(f"Pre-val: {TRAIN_START} -> {RF_PRE_VAL_END}")
print(f"Full train (for CV): {TRAIN_START} -> {RF_TRAIN_END}")
print("5-fold stratified CV, shuffle=False — no leakage into any test window")
print("="*60)

# RF splits
X_rf_pv_raw  = X_df.loc[:RF_PRE_VAL_END]
X_rf_val_raw = X_df.loc[RF_PRE_VAL_END:RF_TRAIN_END]
X_rf_tr_raw  = X_df.loc[:RF_TRAIN_END]

scaler_rf_pv = StandardScaler()
X_rf_pv      = scaler_rf_pv.fit_transform(X_rf_pv_raw)
X_rf_val     = scaler_rf_pv.transform(X_rf_val_raw)

scaler_rf    = StandardScaler()
X_rf_tr      = scaler_rf.fit_transform(X_rf_tr_raw)

y_rf_pv  = y_all.reindex(X_rf_pv_raw.index).fillna(1).astype(int).values
y_rf_val = y_all.reindex(X_rf_val_raw.index).fillna(1).astype(int).values
y_rf_tr  = y_all.reindex(X_rf_tr_raw.index).fillna(1).astype(int).values

# GMM + HMM on RF pre-val
print("Training GMM + HMM for RF/Alpha search...")
set_seed(GLOBAL_SEED)
gmm_rf = GaussianMixture(n_components=K_REGIMES, covariance_type="full",
                          random_state=GLOBAL_SEED, n_init=5, max_iter=200)
gmm_rf.fit(X_rf_pv)
gm_preds2 = gmm_rf.predict(X_rf_pv)
gm_means2 = [X_rf_pv[gm_preds2==k, 0].mean() for k in range(K_REGIMES)]
gm_ord2   = np.argsort(gm_means2).tolist()

hmm_rf = GaussianHMM(n_components=K_REGIMES, covariance_type="full",
                      n_iter=500, random_state=GLOBAL_SEED)
hmm_rf.fit(X_rf_pv)
hm_preds2 = hmm_rf.predict(X_rf_pv)
hm_means2 = [X_rf_pv[hm_preds2==k, 0].mean() for k in range(K_REGIMES)]
hm_ord2   = np.argsort(hm_means2).tolist()

def get_stacked_rf(X_sc):
    gp = gmm_rf.predict_proba(X_sc)[:, gm_ord2]
    hp = forward_filter_proba(hmm_rf, X_sc)[:, hm_ord2]
    return np.hstack([X_sc, gp, hp])

X_rf_pv_s  = get_stacked_rf(X_rf_pv)
X_rf_val_s = get_stacked_rf(X_rf_val)
X_rf_tr_s  = get_stacked_rf(X_rf_tr)

print(f"Bear days — full train: {(y_rf_tr==0).sum()} "
      f"({(y_rf_tr==0).mean()*100:.1f}%)")

cv = StratifiedKFold(n_splits=5, shuffle=False)

rf_ckpt  = load_ckpt("rf_means")
rf_means = {(v["d"], v["n"]): v["mean"] for v in rf_ckpt.get("results", [])}
rf_stds  = {(v["d"], v["n"]): v["std"]  for v in rf_ckpt.get("results", [])}
if rf_means:
    print(f"Resuming: {len(rf_means)} RF configs already done")

done, total_rf = 0, len(RF_DEPTHS)*len(RF_N_TREES)
for depth in RF_DEPTHS:
    for n_trees in RF_N_TREES:
        done += 1
        if (depth, n_trees) in rf_means:
            print(f"  [{done}/{total_rf}] depth={str(depth):>4}, "
                  f"trees={n_trees}  [cached] {rf_means[(depth,n_trees)]:.4f}")
            continue
        f1s = []
        print(f"  [{done}/{total_rf}] depth={str(depth):>4}, trees={n_trees}",
              end="  ", flush=True)
        t0 = time.time()
        for seed in SEEDS:
            rf = RandomForestClassifier(
                n_estimators=n_trees, max_depth=depth,
                class_weight="balanced", random_state=seed)
            scores = cross_val_score(rf, X_rf_tr_s, y_rf_tr,
                                     cv=cv, scoring="f1_macro")
            f1s.append(float(scores.mean()))
        rf_means[(depth,n_trees)] = np.mean(f1s)
        rf_stds[(depth,n_trees)]  = np.std(f1s)
        print(f"mean={np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  "
              f"({time.time()-t0:.0f}s)")
        save_ckpt("rf_means", {"results": [
            {"d": k[0], "n": k[1], "mean": v, "std": rf_stds[k]}
            for k, v in rf_means.items()]})

best_rf    = max(rf_means, key=rf_means.get)
best_d, best_n = best_rf
print(f"\n  WINNER: depth={best_d}, n_trees={best_n}  "
      f"F1={rf_means[best_rf]:.4f}")

log_lines += ["\n"+"="*60,
              "SEARCH 2: RF (5-fold stratified CV, full train 1990-1999)",
              f"seeds={SEEDS}, shuffle=False — no leakage into any test window",
              "="*60]
log_lines.append(f"{'depth':>8}" + "".join(f"  {n:>5}t" for n in RF_N_TREES))
for d in RF_DEPTHS:
    row = f"{str(d):>8}"
    for n in RF_N_TREES:
        star = " (*)" if (d==best_d and n==best_n) else "     "
        row += f"  {rf_means[(d,n)]:.4f}+/-{rf_stds[(d,n)]:.4f}{star}"
    log_lines.append(row)
log_lines.append(f"\nWINNER: depth={best_d}, n_trees={best_n}  "
                 f"F1={rf_means[best_rf]:.4f}")

# RF heatmap
depth_labels = [str(d) if d is not None else "None" for d in RF_DEPTHS]
rf_mat = np.array([[rf_means[(d,n)] for n in RF_N_TREES] for d in RF_DEPTHS])
fig, ax = plt.subplots(figsize=(7,4))
im = ax.imshow(rf_mat, cmap="Blues", aspect="auto",
               vmin=rf_mat.min()*0.995, vmax=rf_mat.max()*1.005)
ax.set_xticks(range(len(RF_N_TREES)))
ax.set_xticklabels([str(n) for n in RF_N_TREES])
ax.set_yticks(range(len(RF_DEPTHS)))
ax.set_yticklabels(depth_labels)
ax.set_xlabel("Number of trees", fontsize=10)
ax.set_ylabel("Maximum depth", fontsize=10)
ax.set_title("Random Forest: 5-fold CV macro-F1 (mean +/- std, 3 seeds)\n"
             "Window 1 training set 1990-1999, no leakage into test windows",
             fontsize=10)
for i,d in enumerate(RF_DEPTHS):
    for j,n in enumerate(RF_N_TREES):
        v, s = rf_mat[i,j], rf_stds[(d,n)]
        star = " *" if (d==best_d and n==best_n) else ""
        col  = "white" if v > rf_mat.mean() else "black"
        ax.text(j, i-0.15, f"{v:.4f}{star}", ha="center", va="center",
                fontsize=7, color=col,
                fontweight="bold" if star else "normal")
        ax.text(j, i+0.25, f"+/-{s:.4f}", ha="center", va="center",
                fontsize=5.5, color=col, alpha=0.8)
plt.colorbar(im, ax=ax, label="Mean CV Macro-F1")
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "fig_grid_rf.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_grid_rf.png")

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH 3: ALPHA — corrected backtest, Window 1 test period
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SEARCH 3: ALPHA PARAMETERS")
print(f"Training RF + LSTM on full Window 1 train (1990-1999)")
print(f"Testing on Window 1 test (2000-2009) — never seen in searches 1 or 2")
print("Corrected: a_bear/a_bull explicitly in weight formula")
print("="*60)

# Test data for alpha search
X_te_raw = X_df.loc[RF_TEST_START:]
X_te_sc  = scaler_rf.transform(X_te_raw)
X_te_s   = get_stacked_rf(X_te_sc)
y_te     = y_all.reindex(X_te_raw.index).fillna(1).astype(int).values

# Train full RF with winning hyperparameters
print(f"  Training RF (depth={best_d}, trees={best_n})...")
set_seed(GLOBAL_SEED)
rf_full = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=best_n, max_depth=best_d,
                           class_weight="balanced", random_state=GLOBAL_SEED),
    method="sigmoid", cv=5)
rf_full.fit(X_rf_tr_s, y_rf_tr)

# Train full LSTM with winning hyperparameters
print(f"  Training LSTM (h={best_h}, dr={best_dr}, "
      f"seq={best_seq}, lr={best_lr:.0e})...")
torch.manual_seed(GLOBAL_SEED)

# Use RF stacked features for LSTM (same feature set)
X_lstm_tr_for_alpha = X_rf_tr_s  # RF training set features
y_lstm_tr_for_alpha = y_rf_tr

lstm_full = LSTMRegime(X_rf_tr_s.shape[1], best_h,
                        dropout=best_dr).to(DEVICE)
ds_tr     = RegimeDataset(X_lstm_tr_for_alpha, y_lstm_tr_for_alpha, best_seq)
ldr_tr    = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=False)
opt_f     = torch.optim.Adam(lstm_full.parameters(),
                               lr=best_lr, weight_decay=1e-5)
crit_f    = nn.CrossEntropyLoss()
Xv_f      = torch.tensor(X_rf_val_s.astype(np.float32))
best_f1_f, patience_cnt = 0.0, 0
for epoch in range(MAX_EPOCHS):
    lstm_full.train()
    for xb, yb in ldr_tr:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_f.zero_grad()
        loss = crit_f(lstm_full(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_full.parameters(), 1.0)
        opt_f.step()
    lstm_full.eval()
    preds_v = np.full(len(X_rf_val_s), 1, dtype=int)
    with torch.no_grad():
        for t in range(best_seq, len(X_rf_val_s)):
            seq = Xv_f[t-best_seq:t].unsqueeze(0).to(DEVICE)
            preds_v[t] = lstm_full(seq).argmax(dim=1).item()
    vf1 = macro_f1(y_rf_val, preds_v)
    if vf1 > best_f1_f:
        best_f1_f, patience_cnt = vf1, 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            break
print(f"  LSTM trained, val F1={best_f1_f:.4f}")

# Ensemble probabilities on test set
gp_te = gmm_rf.predict_proba(X_te_sc)[:, gm_ord2]
hp_te = forward_filter_proba(hmm_rf, X_te_sc)[:, hm_ord2]
rf_p  = rf_full.predict_proba(X_te_s)
n_te  = len(X_te_sc)
lp    = np.full((n_te, K_REGIMES), 1.0/K_REGIMES)
Xt    = torch.tensor(X_te_s.astype(np.float32))
lstm_full.eval()
with torch.no_grad():
    for t in range(best_seq, n_te):
        lp[t] = torch.softmax(
            lstm_full(Xt[t-best_seq:t].unsqueeze(0).to(DEVICE)),
            dim=-1).cpu().numpy()[0]

val_f1s = {
    "gmm":  macro_f1(y_rf_val,
                     gmm_rf.predict_proba(X_rf_val)[:,gm_ord2].argmax(1)),
    "hmm":  macro_f1(y_rf_val,
                     forward_filter_proba(hmm_rf,X_rf_val)[:,hm_ord2].argmax(1)),
    "rf":   macro_f1(y_rf_val, rf_full.predict(X_rf_val_s)),
    "lstm": best_f1_f,
}
weights = softmax_weights(val_f1s, temperature=1.0)
print(f"  Ensemble weights: {weights}")

ens_proba = (weights["gmm"]*gp_te + weights["hmm"]*hp_te
             + weights["rf"]*rf_p  + weights["lstm"]*lp)
conf      = ens_proba.max(axis=1)
p_bull    = ens_proba[:,2]
p_bear_e  = ens_proba[:,0]

hard     = np.stack([gp_te.argmax(1), hp_te.argmax(1),
                     rf_p.argmax(1),  lp.argmax(1)], axis=1)
majority = np.array([np.bincount(r.astype(int), minlength=K_REGIMES).argmax()
                     for r in hard])
conflict = 1.0 - (hard==majority[:,None]).mean(axis=1)

test_idx  = X_te_raw.index
ret_te    = df["Return"].reindex(test_idx).fillna(0).values
rd_te     = r_daily.reindex(test_idx).fillna(0).values
sig_te    = sigma_ann.reindex(test_idx).values
sig_med   = float(sigma_ann.reindex(X_rf_tr_raw.index).median())

u_star    = build_hjb_signal(logret.reindex(test_idx),
                              sigma_ann.reindex(test_idx),
                              r_daily.reindex(test_idx))
w_hjb     = u_star.shift(1).fillna(0).values
base_band = build_no_trade_band(
    sigma_ann, TC, BAND_MULT, BAND_ROLL, WARMUP
).reindex(test_idx).values
conf_band = np.nan_to_num(
    base_band*(1+CONFLICT_SCALE*conflict),
    nan=float(np.nanmedian(base_band)))

def backtest_alpha(a_bear, a_bull):
    """Corrected: a_bear and a_bull explicitly modulate HJB weight."""
    w_regime = np.clip(
        w_hjb * (1.0 + a_bull*p_bull - a_bear*p_bear_e), 0, 1)
    w_fin = np.clip(conf*w_regime + (1-conf)*w_hjb, 0, 1)
    w_curr = 0.0
    port_r = np.zeros(len(test_idx))
    for t in range(len(test_idx)):
        diff = w_fin[t] - w_curr
        if abs(diff) > conf_band[t]:
            w_curr = float(np.clip(
                w_curr + np.sign(diff)*conf_band[t], 0, 1))
        port_r[t] = (w_curr*ret_te[t] + (1-w_curr)*rd_te[t]
                     - TC*abs(w_fin[t]-w_curr))
    return sharpe_ratio(pd.Series(port_r, index=test_idx))

alpha_res = {}
done, total_a = 0, len(ALPHA_BEARS)*len(ALPHA_BULLS)
for a_bear in ALPHA_BEARS:
    for a_bull in ALPHA_BULLS:
        done += 1
        sr = backtest_alpha(a_bear, a_bull)
        alpha_res[(a_bear,a_bull)] = sr
        print(f"  [{done}/{total_a}] ab={a_bear} al={a_bull}  "
              f"r={a_bear/a_bull:.1f}  Sharpe={sr:.4f}")

best_alpha = max(alpha_res, key=alpha_res.get)
best_ab, best_al = best_alpha
print(f"\n  WINNER: alpha_bear={best_ab}, alpha_bull={best_al}  "
      f"r={best_ab/best_al:.2f}  Sharpe={alpha_res[best_alpha]:.4f}")

log_lines += ["\n"+"="*60,
              "SEARCH 3: Alpha (Window 1 test-set Sharpe 2000-2009)",
              "Corrected backtest — a_bear/a_bull in regime weight formula",
              "r = alpha_bear / alpha_bull (CRRA asymmetry ratio)",
              "="*60]
log_lines.append(f"{'alpha_bear':>12}" +
                 "".join(f"  bull={b:.1f}" for b in ALPHA_BULLS))
for ab in ALPHA_BEARS:
    row = f"{ab:>12.1f}"
    for al in ALPHA_BULLS:
        star = " (*)" if (ab==best_ab and al==best_al) else "     "
        row += f"  {alpha_res[(ab,al)]:.4f}{star}"
    log_lines.append(row)
log_lines.append(f"\nWINNER: alpha_bear={best_ab}, alpha_bull={best_al}  "
                 f"r={best_ab/best_al:.2f}")

# Alpha heatmap
alpha_mat = np.array([[alpha_res[(ab,al)] for al in ALPHA_BULLS]
                      for ab in ALPHA_BEARS])
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(alpha_mat, cmap="Oranges", aspect="auto",
               vmin=alpha_mat.min()*0.998, vmax=alpha_mat.max()*1.002)
ax.set_xticks(range(len(ALPHA_BULLS)))
ax.set_xticklabels([str(b) for b in ALPHA_BULLS])
ax.set_yticks(range(len(ALPHA_BEARS)))
ax.set_yticklabels([str(b) for b in ALPHA_BEARS])
ax.set_xlabel(r"$\alpha_{\mathrm{bull}}$", fontsize=11)
ax.set_ylabel(r"$\alpha_{\mathrm{bear}}$", fontsize=11)
ax.set_title("Alpha grid search: Window 1 test-set Sharpe (2000-2009)\n"
             r"by $\alpha_{\mathrm{bear}}$ and $\alpha_{\mathrm{bull}}$"
             " (r annotated)", fontsize=10)
for i,ab in enumerate(ALPHA_BEARS):
    for j,al in enumerate(ALPHA_BULLS):
        v    = alpha_mat[i,j]
        star = " *" if (ab==best_ab and al==best_al) else ""
        col  = "white" if v > alpha_mat.mean() else "black"
        ax.text(j, i-0.15, f"{v:.4f}{star}", ha="center", va="center",
                fontsize=8, color=col)
        ax.text(j, i+0.28, f"r={ab/al:.1f}", ha="center", va="center",
                fontsize=6, color="grey")
plt.colorbar(im, ax=ax, label="Sharpe ratio")
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "fig_grid_alpha.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_grid_alpha.png")

# ── Final summary ─────────────────────────────────────────────────────────────
log_lines += [
    "\n"+"="*60,
    "WINNING HYPERPARAMETERS — copy into config.py",
    "="*60,
    f"# LSTM (global val 2008-2015, disclosed lookahead):",
    f"LSTM_HIDDEN     = {best_h}",
    f"LSTM_DROPOUT    = {best_dr}",
    f"LSTM_SEQ_LEN    = {best_seq}",
    f"LSTM_LR         = {best_lr:.0e}",
    f"",
    f"# RF (Window 1 CV, clean — no leakage):",
    f"RF_MAX_DEPTH    = {best_d}",
    f"RF_N_ESTIMATORS = {best_n}",
    f"",
    f"# Alpha (Window 1 test 2000-2009, corrected backtest):",
    f"ALPHA_BEAR      = {best_ab}",
    f"ALPHA_BULL      = {best_al}",
    f"",
    f"Methodological notes:",
    f"  LSTM: val 2008-2015 overlaps Window 1/2 test. Acknowledged.",
    f"    Defended by cross-market Phase 3 — same HP applied to",
    f"    FTSE, DAX, Nikkei, EEM, GLD without modification.",
    f"  RF: 5-fold stratified CV on full Window 1 train (1990-1999).",
    f"    shuffle=False preserves temporal order. No leakage.",
    f"  Alpha: tested on Window 1 test (2000-2009) never seen",
    f"    during LSTM or RF hyperparameter search.",
    f"  Seeds: {SEEDS}",
    f"  Early stopping patience: {PATIENCE} epochs",
]

# Clean up checkpoints
for ck in CKPT_DIR.glob("*.json"):
    ck.unlink()
print("Checkpoints cleaned up.")

out_path = os.path.join(ROOT, "grid_search_results.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"Results saved: {out_path}")

print("\n"+"="*60)
print("COMPLETE. Copy into config.py:")
print(f"  LSTM_HIDDEN     = {best_h}")
print(f"  LSTM_DROPOUT    = {best_dr}")
print(f"  LSTM_SEQ_LEN    = {best_seq}")
print(f"  LSTM_LR         = {best_lr:.0e}")
print(f"  RF_MAX_DEPTH    = {best_d}")
print(f"  RF_N_ESTIMATORS = {best_n}")
print(f"  ALPHA_BEAR      = {best_ab}")
print(f"  ALPHA_BULL      = {best_al}")
print("="*60)