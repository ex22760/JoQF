"""
config.py
Central configuration for all experiments.
Change values here and they propagate everywhere automatically.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data", "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures", "output")

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
GLOBAL_SEED = 42
SEEDS       = [42, 123, 456]      # multi-seed averaging for grid search

# ── Assets ────────────────────────────────────────────────────────────────────
PRIMARY_ASSET = "^GSPC"           # S&P 500 — primary specification

CROSS_MARKET_ASSETS = {
    "FTSE100":  "^FTSE",
    "DAX":      "^GDAXI",
    "Nikkei":   "^N225",
    "EEM":      "EEM",
    "Gold":     "GLD",
}

MULTI_ASSET_UNIVERSE = {
    "SP500":    "^GSPC",
    "LT_Bonds": "TLT",   # 20-year US treasuries — flight-to-quality (Baele et al. 2010)
    "TIPS":     "TIP",   # inflation-protected treasuries — stagflation hedge (Fleckenstein et al. 2014)
    "ST_Bonds": "SHY",   # short-term treasuries — capital preservation floor
}

# ── Rolling window definitions ─────────────────────────────────────────────────
# (train_start, train_end, val_start, val_end, test_start, test_end)
# Validation = last 20% of training period (approximate)
ROLLING_WINDOWS = [
    {
        "name":        "Window1_2000s",
        "train_start": "1990-01-01",
        "val_start":   "1996-01-01",   # ~last 4 years of train
        "train_end":   "1999-12-31",
        "test_start":  "2000-01-01",
        "test_end":    "2009-12-31",
        "description": "Train 1990-1999 | Test 2000-2009 (dot-com + GFC)",
    },
    {
        "name":        "Window2_2010s",
        "train_start": "1990-01-01",
        "val_start":   "2006-01-01",   # includes GFC onset in val
        "train_end":   "2009-12-31",
        "test_start":  "2010-01-01",
        "test_end":    "2019-12-31",
        "description": "Train 1990-2009 | Test 2010-2019 (bull market stress test)",
    },
    {
        "name":        "Window3_2020s",
        "train_start": "1990-01-01",
        "val_start":   "2016-01-01",   # last ~4 years of training
        "train_end":   "2019-12-31",
        "test_start":  "2020-01-01",
        "test_end":    "2026-02-28",
        "description": "Train 1990-2019 | Test 2020-2026 (COVID, 2022 bear, Liberation Day)",
    },
]

# ── Volatility specifications ──────────────────────────────────────────────────
VOL_SPECS = [
    "ewma",           # SPEC 1: realised vol only
    "vix",            # SPEC 2: implied vol only
    "ewma_vix",       # SPEC 3: blend (current baseline)
    "garch",          # SPEC 4: GARCH(1,1)
    "gjr_garch",      # SPEC 5: GJR-GARCH(1,1) — leverage effect
]
BASELINE_VOL_SPEC  = "ewma_vix"   # original baseline
BEST_VOL_SPEC      = "gjr_garch"  # best from Phase 2

# ── Regime labelling schemes ───────────────────────────────────────────────────
LABEL_SCHEMES = [
    "nber",           # NBER recession dates (benchmark)
    "drawdown",       # drawdown-threshold (>15% peak-to-trough)
]
BASELINE_LABEL_SCHEME = "nber"

# ── Feature engineering ───────────────────────────────────────────────────────
LOOKBACK_ST      = 20
LOOKBACK_LT      = 126
ANNUALIZATION    = 252
VIX_WEIGHT       = 0.5
MACRO_WEIGHT     = 0.01
ALPHA_SHORT      = 0.5
MACRO_LAGS       = {"CPI": 21, "Unemployment": 7, "FedFunds": 1}
WARMUP           = LOOKBACK_LT * 2
DRAWDOWN_BEAR_THRESHOLD = 0.15   # >15% peak-to-trough = bear (drawdown scheme)
PAGAN_MIN_DAYS   = 70
PAGAN_MIN_RET    = 0.15

# ── Model hyperparameters (from grid search — see grid_search_results.txt) ───
# LSTM: global val 2008-2015, acknowledged lookahead, defended by Phase 3
# RF:   Window 1 5-fold CV 1990-1999, no leakage
# Alpha: Window 1 test 2000-2009, corrected backtest
RF_MAX_DEPTH     = 4
RF_N_ESTIMATORS  = 500
LSTM_HIDDEN      = 256
LSTM_DROPOUT     = 0.2
LSTM_SEQ_LEN     = 120
LSTM_LR          = 5e-4
LSTM_EPOCHS      = 50             # early stopping will cut this
LSTM_PATIENCE    = 10
LSTM_BATCH       = 32
K_REGIMES        = 3

# ── Ensemble ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55
CONFLICT_SCALE       = 1.5
SOFTMAX_TEMP         = 1.0

# ── Allocation ────────────────────────────────────────────────────────────────
RISK_AVERSION    = 4.0
TC_BPS           = 5
TC               = TC_BPS / 1e4
BAND_MULT        = 1.5
BAND_ROLL        = 252
BEAR_FLOOR       = 0.05
ALPHA_BEAR       = 0.6    # from Window 1 corrected alpha search
ALPHA_BULL       = 0.1    # from Window 1 corrected alpha search

# ── Momentum persistence filter ───────────────────────────────────────────────
# Prevents ensemble from increasing equity weight during confirmed bear regimes
# unless short-term momentum has recovered above the threshold.
# MOM_PERSIST_THRESHOLD: normalised momentum level required to permit bear exit
#   0.0 = always allow exit (filter off)
#   0.3 = require 30% of baseline momentum recovery before permitting re-entry
# MOM_PERSIST_WINDOW: lookback for short-term momentum (trading days)
MOM_PERSIST_THRESHOLD = 0.0   # set to 0.3 for filtered experiment
MOM_PERSIST_WINDOW    = 20    # 1-month short-term momentum

# Multi-asset allocation
MIN_VAR_LOOKBACK = 126            # rolling window for min-var covariance
MIN_VAR_FLOOR    = 0.05           # minimum weight per defensive asset
MIN_VAR_SHRINK   = 0.1           # ledoit-wolf shrinkage intensity

# ── GARCH ─────────────────────────────────────────────────────────────────────
GARCH_P          = 1
GARCH_Q          = 1
GARCH_DIST       = "normal"      # or "t" for fat tails
GARCH_REFIT_FREQ = 252           # refit GARCH parameters annually

# ── Academic plot style ───────────────────────────────────────────────────────
MPL_STYLE = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "text.usetex":       False,
}

# ── Regime colours ────────────────────────────────────────────────────────────
REGIME_COLOURS = {
    "bear":    "#D62728",
    "neutral": "#AAAAAA",
    "bull":    "#2CA02C",
}