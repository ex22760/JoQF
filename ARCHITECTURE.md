# ============================================================
# JOURNAL OF FINANCE PAPER — ARCHITECTURE AND STRUCTURE
# "Real-Time Market Regime Detection: A Cross-Market,
#  Multi-Window, Volatility-Robust Ensemble Framework"
# ============================================================
#
# PAPER STRUCTURE
# ──────────────
# 1. Introduction
#    - Core question: does probabilistic regime detection generate
#      economically significant improvements, robust to regime
#      definition, time period, market, and volatility specification?
#    - Motivate with Liberation Day (Apr 2025) and COVID (Mar 2020)
#    - Preview key results across all experimental conditions
#
# 2. Related Literature
#    - Regime switching: Hamilton (1989), Ang & Bekaert (2004)
#    - HMM in finance: Nystrup et al. (2015, 2017, 2019)
#    - Ensemble methods: Kritzman et al. (2012), Gupta et al. (2025)
#    - Volatility modelling: Engle (1982), Nelson (1991), Glosten et al. (1993)
#    - Multi-asset allocation: Asness et al. (2012), Mulvey & Liu (2016)
#
# 3. Data and Methodology
#    3.1 Data sources (S&P 500 + macro, then 5 additional assets)
#    3.2 Regime labelling (NBER + drawdown-threshold, side by side)
#    3.3 Volatility specifications (5 variants, defined here)
#    3.4 Feature engineering (shared across all specs)
#    3.5 Model architecture (GMM, HMM, RF, LSTM, ensemble)
#    3.6 Ensemble combination (confidence + conflict scoring)
#    3.7 Allocation framework (HJB baseline + multi-asset extension)
#    3.8 Evaluation design (rolling windows, no lookahead)
#
# 4. Baseline Results (S&P 500, EWMA+VIX, NBER labels)
#    4.1 Rolling window performance (3 windows)
#    4.2 Per-regime diagnostics (bear weight reduction)
#    4.3 Comparison to benchmark strategies
#    → Table 1: Rolling window results
#    → Figure 1: Equity curves across 3 windows
#
# 5. Robustness to Regime Definition
#    5.1 NBER vs drawdown-threshold labels
#    5.2 Classification performance under each scheme
#    5.3 Portfolio performance under each scheme
#    → Table 2: Results under both labelling schemes
#
# 6. Robustness to Volatility Specification (NEW)
#    6.1 Five volatility models described
#    6.2 Detection performance (bear recall, macro-F1)
#    6.3 Portfolio performance (Sharpe, MaxDD)
#    6.4 Consistency across rolling windows
#    → Table 3: Performance by volatility specification
#    → Figure 2: GARCH conditional vol vs EWMA during crises
#    → Key finding: framework is robust to volatility model choice
#      OR a particular spec dominates (either is publishable)
#
# 7. Cross-Market Generalisation
#    7.1 Five additional assets (FTSE, DAX, Nikkei, EEM, GLD)
#    7.2 Architecture fixed, GMM/HMM retrained per asset
#    7.3 Best volatility spec from Section 6 used, EWMA+VIX as robustness check
#    → Table 4: Cross-market results (Sharpe, MaxDD, CAGR)
#    → Figure 3: Regime detection across markets
#
# 8. Multi-Asset Allocation
#    8.1 Asset universe: S&P 500, TLT, GLD, SHY
#    8.2 Bear regime: dynamic minimum variance over {TLT, GLD, SHY}
#    8.3 Neutral regime: vol-targeting across equities + bonds
#    8.4 Bull regime: equity momentum overlay
#    8.5 Results vs single-asset and equal-weight benchmarks
#    → Table 5: Multi-asset allocation results
#    → Figure 4: Asset allocation through time
#
# 9. Statistical Significance
#    9.1 Panel Jobson-Korkie across 3 windows × 6 assets × 2 label schemes
#    9.2 Bootstrap confidence intervals on Sharpe differences
#    9.3 Diebold-Mariano test on forecast accuracy
#    9.4 Are differences driven by volatility spec? (new test)
#    → Table 6: Statistical significance panel
#
# 10. Conclusion
#     - Answer the core question with evidence from all conditions
#     - Structural finding: bear detection tractable, bull not
#     - Policy implication: regime frameworks provide insurance value
#       even when Sharpe improvement is modest in bull-dominated periods
#
# ============================================================
# CODEBASE ARCHITECTURE
# ============================================================
#
# regime_framework/
# │
# ├── config.py                   — All constants, asset tickers,
# │                                 window definitions, hyperparameters
# │
# ├── data/
# │   ├── loader.py               — Download/cache data for any ticker
# │   ├── features.py             — Feature engineering (causal EWMA,
# │   │                             macro signals, VIX blend)
# │   └── labels.py               — NBER labels + drawdown-threshold labels
# │
# ├── volatility/
# │   ├── ewma.py                 — Current EWMA + VIX implementation
# │   ├── garch.py                — GARCH(1,1), EGARCH/GJR-GARCH
# │   └── factory.py              — get_vol_model(spec) dispatcher
# │
# ├── models/
# │   ├── unsupervised.py         — GMM + HMM (from unsupervisedmodel.py)
# │   ├── supervised.py           — RF + LSTM (from supervisedmodel.py)
# │   ├── ensemble.py             — Combination, confidence, conflict
# │   └── base.py                 — Shared utilities (forward filter etc.)
# │
# ├── allocation/
# │   ├── hjb.py                  — HJB baseline (from baselinemodel.py)
# │   ├── single_asset.py         — Current dynamic allocation
# │   └── multi_asset.py          — NEW: TLT/GLD/SHY dynamic min-var
# │
# ├── evaluation/
# │   ├── metrics.py              — Sharpe, MaxDD, CAGR, macro-F1
# │   └── stats.py                — Jobson-Korkie, bootstrap, DM test
# │
# ├── pipeline.py                 — Master function:
# │                                 run_experiment(asset, train_end,
# │                                   test_start, test_end,
# │                                   vol_spec, label_scheme)
# │                                 Returns: ExperimentResult dataclass
# │
# ├── experiments/
# │   ├── rolling_windows.py      — Phase 1: 3-window backtest
# │   ├── label_robustness.py     — Phase 1b: NBER vs drawdown
# │   ├── vol_robustness.py       — Phase 2: 5 vol specs
# │   ├── cross_market.py         — Phase 3: 5 additional assets
# │   ├── multi_asset_alloc.py    — Phase 4: TLT/GLD/SHY
# │   └── significance.py         — Phase 5: statistical tests
# │
# ├── grid_search/
# │   └── hyperparameters.py      — Extended grid search (already built)
# │
# ├── figures/
# │   └── plotting.py             — All publication-quality figure code
# │
# └── run_all.py                  — Master script: runs all experiments,
#                                   saves results to results/
#
# ============================================================
# KEY DESIGN PRINCIPLES
# ============================================================
#
# 1. SINGLE ENTRY POINT
#    pipeline.py::run_experiment() is called by every experiment script.
#    No experiment duplicates data loading, feature engineering, or
#    model training logic. Changes propagate everywhere automatically.
#
# 2. ASSET-AGNOSTIC
#    Every function takes `ticker` as a parameter. The same code
#    runs on S&P 500, FTSE 100, Nikkei — no copy-pasting.
#
# 3. VOLATILITY FACTORY PATTERN
#    volatility/factory.py::get_vol_model(spec) returns a callable
#    that takes returns and outputs sigma_ann. Swapping EWMA for
#    GARCH requires changing one parameter string, nothing else.
#
# 4. RESULTS DATACLASS
#    Every experiment returns a typed ExperimentResult with fields:
#    sharpe, cagr, max_dd, macro_f1, bear_recall, vol_spec,
#    label_scheme, asset, train_window, test_window.
#    Results are stored as pickle + CSV for easy aggregation.
#
# 5. NO LOOKAHEAD ANYWHERE
#    All scalers, rolling statistics, and vol model parameters
#    are fit on train data only and applied to val/test.
#    GARCH parameters re-estimated on expanding window.
#
# 6. REPRODUCIBILITY
#    Global seed block in pipeline.py applied before every experiment.
#    All results deterministic given the same seed.
#
# ============================================================
# VOLATILITY SPECIFICATIONS (Phase 2)
# ============================================================
#
# SPEC 1: EWMA (realised vol only)
#   sigma_t = EWMSTD(returns, span=126) * sqrt(252)
#
# SPEC 2: VIX only (implied vol only)
#   sigma_t = VIX_t / 100
#
# SPEC 3: EWMA + VIX blend (current baseline)
#   sigma_t = 0.5 * EWMA + 0.5 * VIX
#
# SPEC 4: GARCH(1,1)
#   sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + beta*sigma_{t-1}^2
#   Parameters estimated on train set via MLE (arch package)
#   Forecast applied causally on test set
#
# SPEC 5: GJR-GARCH(1,1)  [preferred over EGARCH for equity]
#   sigma_t^2 = omega + (alpha + gamma*I_{t-1})*epsilon_{t-1}^2
#               + beta*sigma_{t-1}^2
#   where I_{t-1} = 1 if epsilon_{t-1} < 0 (leverage effect)
#   GJR preferred over EGARCH: simpler, interpretable gamma coefficient,
#   directly tests whether negative returns increase vol more than
#   positive returns — the asymmetry most relevant to bear detection
#
# ============================================================
# ROLLING WINDOW DESIGN
# ============================================================
#
# Window 1: Train 1990-01-01 -> 1999-12-31  Test 2000-01-01 -> 2009-12-31
#   Contains: dot-com crash, post-9/11, GFC (2008-2009) — 2 major bear markets
#
# Window 2: Train 1990-01-01 -> 2009-12-31  Test 2010-01-01 -> 2019-12-31
#   Contains: 2011 EU debt crisis, 2015-16 correction, 2018 Q4 selloff
#   Stress test: mostly bull market — tests false positive rate
#
# Window 3: Train 1990-01-01 -> 2019-12-31  Test 2019-01-01 -> 2026-02-28
#   Contains: COVID crash, 2022 bear, Liberation Day
#   Current dissertation results
#
# For each window, validation = last 20% of training period
#   Window 1 val: 1996-2000
#   Window 2 val: 2006-2010 (includes GFC onset)
#   Window 3 val: 2015-2019 (as currently implemented, approx)
#
# ============================================================
# BUILD ORDER
# ============================================================
#
# Week 1:  config.py, data/loader.py, data/features.py, data/labels.py
# Week 2:  volatility/ (all specs), models/base.py
# Week 3:  models/unsupervised.py, models/supervised.py, models/ensemble.py
# Week 4:  allocation/hjb.py, allocation/single_asset.py, pipeline.py
# Week 5:  experiments/rolling_windows.py + run, check results
# Week 6:  allocation/multi_asset.py
# Week 7:  experiments/vol_robustness.py + cross_market.py
# Week 8:  experiments/significance.py, figures/plotting.py
# Week 9+: write-up
