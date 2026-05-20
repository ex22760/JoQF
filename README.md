# Persistent Latent States and Asymmetric Volatility: Why Explicit Regime Detection Fails to Outperform a Momentum-Volatility Baseline

**Sujin Subanthran**  
Department of Engineering Mathematics, University of Bristol  
Quantitative Finance (Taylor & Francis), May 2026

---

## Overview

This repository contains the full codebase for the paper. The framework evaluates whether a probabilistic ensemble of Gaussian Mixture Model (GMM), Hidden Markov Model (HMM), Random Forest and LSTM generates economically significant portfolio improvements over a Merton-optimal Hamilton-Jacobi-Bellman (HJB) stochastic control baseline across 31 experimental conditions spanning three rolling windows, five volatility specifications and six international equity markets.

The central finding is **confident misclassification during quiet bear days**: during low-volatility episodes within sustained bear markets, the ensemble's volatility-dependent features produce neutral-like signals, causing high-confidence incorrect regime classifications and active equity re-entry at exactly the wrong moment. The mechanism is confirmed across four independent bear definitions (8/8 conditions), six markets (9/10 conditions), and all volatility specifications.

---

## Repository Structure

```
JF/
├── config.py                    # All hyperparameters, window definitions, constants
├── pipeline.py                  # Core run_experiment() function and daily series saving
├── run_all.py                   # Phase dispatcher (--phase 1 through 9)
├── run_phase.sh                 # SLURM batch script for BluePebble HPC
├── quiet_bear_analysis.py       # Primary quiet bear mechanism test (drawdown labels)
├── requirements.txt
│
├── data/
│   ├── loader.py                # Yahoo Finance + FRED data loading with caching
│   ├── features.py              # Causal feature engineering (EWMA, VIX, macro)
│   └── labels.py                # NBER, drawdown-threshold, Pagan-Sossounov labelling
│
├── volatility/
│   └── factory.py               # EWMA, VIX, EWMA+VIX, GARCH, GJR-GARCH specifications
│
├── models/
│   └── base.py                  # GMM, HMM, RF, LSTM model classes
│
├── allocation/
│   └── ...                      # HJB baseline, ensemble blending, multi-asset min-var
│
├── experiments/
│   ├── rolling_windows.py       # Phase 1: baseline rolling window backtest
│   ├── vol_robustness.py        # Phase 2: volatility specification robustness
│   ├── cross_market.py          # Phase 3: FTSE, DAX, Nikkei, EEM, GLD
│   ├── multi_asset_alloc.py     # Phase 4: TLT/TIP/SHY defensive basket
│   ├── significance.py          # Phase 5: panel Jobson-Korkie, ANOVA, sign test
│   ├── momentum_filter.py       # Phase 6: GJR-GARCH + momentum persistence gate
│   ├── phase7_analysis.py       # Phase 7: alternative bear definitions + confidence analysis
│   ├── phase8_analysis.py       # Phase 8: threshold robustness + TC/turnover
│   ├── phase9_ablation.py       # Phase 9: ensemble ablation study
│   └── generate_figures.py      # Figures 6-10 for paper
│
├── figures/
│   └── output/                  # Generated figures (png)
│
└── results/                     # Saved experiment results (pkl, csv)
```

---

## Experimental Design

Three non-overlapping rolling windows provide independent out-of-sample evaluation:

| Window | Training | Test | Key Episodes |
|--------|----------|------|--------------|
| Window 1 | 1990–1999 | 2000–2009 | Dot-com crash, GFC |
| Window 2 | 1990–2009 | 2010–2019 | Prolonged bull market |
| Window 3 | 1990–2019 | 2020–2026 | COVID, 2022 inflation bear, Liberation Day |

Five volatility specifications are evaluated: EWMA, VIX, EWMA+VIX (baseline), GARCH(1,1), GJR-GARCH(1,1). Six assets are tested: S&P 500 (primary), FTSE 100, DAX, Nikkei 225, EEM, GLD.

---

## Installation

```bash
git clone https://github.com/ex22760/JoQF.git
cd JoQF
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, scikit-learn, hmmlearn, arch, pandas, numpy, matplotlib, scipy, yfinance, fredapi.

---

## Data

All data is downloaded automatically from public sources:

- **Equity prices:** Yahoo Finance (daily OHLCV via `yfinance`)
- **Macro series:** FRED API (CPI, unemployment rate, Federal Funds Rate, NBER recession indicators)
- **VIX:** Yahoo Finance (`^VIX`)

No proprietary data is required. Data is cached locally in `data/cache/` after first download. A valid FRED API key is required — set it in `config.py`:

```python
FRED_API_KEY = "your_key_here"  # https://fred.stlouisfed.org/docs/api/api_key.html
```

---

## Running the Experiments

### Local

```bash
# Run a single phase
python run_all.py --phase 1

# Run all phases sequentially
for phase in 1 2 3 4 5 6 7 8 9; do
    python run_all.py --phase $phase
done

# Run quiet bear primary analysis
python quiet_bear_analysis.py

# Generate all paper figures
python experiments/generate_figures.py
```

### HPC (BluePebble / SLURM)

```bash
# Submit individual phases (GPU recommended for Phases 1, 2, 3, 6)
PHASE=1 sbatch run_phase.sh
PHASE=2 sbatch run_phase.sh

# Phases 4, 5, 7, 8, 9 can run interactively (no GPU needed)
python run_all.py --phase 4
python experiments/phase7_analysis.py
python experiments/phase8_analysis.py
python experiments/phase9_ablation.py
```

Phases are independent and can be submitted simultaneously. Phase 5 (statistical significance) should run after Phases 1–3 to ensure all pkl results are available.

---

## Key Hyperparameters

All hyperparameters are defined in `config.py`. The values below were selected on Window 1 training data only (1990–1999) and held fixed across all subsequent windows and markets.

| Parameter | Value | Selection method |
|-----------|-------|-----------------|
| RF max depth | 4 | 5-fold CV on Window 1 training set |
| RF n estimators | 500 | 5-fold CV on Window 1 training set |
| LSTM hidden size | 256 | Grid search on 2008–2015 validation period |
| LSTM dropout | 0.2 | Grid search on 2008–2015 validation period |
| LSTM sequence length | 120 | Grid search on 2008–2015 validation period |
| α_bear | 0.6 | Grid search on Window 1 test set |
| α_bull | 0.1 | Grid search on Window 1 test set |
| Risk aversion γ | 4 | Standard CRRA literature value |
| Confidence threshold | 0.55 | Implementation detail; invariant across [0.45, 0.70] |

---

## Reproducing Paper Results

Running all nine phases produces the complete set of results. Key outputs:

| Phase | Output files | Paper table/figure |
|-------|-------------|-------------------|
| 1 | `results/GSPC_Window*_ewma_vix_nber.pkl` | Table 1, Figure 6 |
| 2 | `results/GSPC_Window*_*_nber.pkl` | Table 2 |
| 3 | `results/*_Window*_ewma_vix_nber.pkl` | Table 4 |
| 4 | `results/multi_asset_allocation.csv` | Table 5 |
| 5 | `results/phase5_significance.csv` | Table 6, Figure 9 |
| 6 | `results/phase6_momentum_gate.csv` | Table 11 |
| 7 | `results/phase7_*.csv` | Tables 9, 10; Figures 7, conf histograms |
| 8 | `results/phase8_*.csv` | Tables 13, 14; Figures 9, 10 |
| 9 | `results/phase9_ablation.csv` | Table 12, Figure 10 |
| figures | `figures/output/fig*.png` | All paper figures |

Expected total runtime on a single RTX 2080 Ti: approximately 12–15 hours for all GPU phases. Analysis phases (4, 5, 7, 8, 9) complete in under 5 minutes each on CPU.

---

## Citation

```bibtex
@article{subanthran2026regime,
  title   = {Persistent Latent States and Asymmetric Volatility: Why Explicit 
             Regime Detection Fails to Outperform a Momentum-Volatility Baseline},
  author  = {Subanthran, Sujin},
  journal = {Quantitative Finance},
  year    = {2026},
  note    = {Unsubmitted}
}
```

---

## License

MIT License. Data from Yahoo Finance and FRED is subject to their respective terms of use.
Persistent Latent States and Asymmetric Volatility: Why
Explicit Regime Detection Fails to Outperform a Momentum-Volatility Baseline

---
## Acknowledgements
The author thanks Dr Yani Berdeni and Dr Nicolas Verschueren Van Rees for supervision 
and guidance throughout this project. This work was carried out using the 
computational facilities of the Advanced Computing Research Centre, 
University of Bristol - http://www.bristol.ac.uk/acrc/.
