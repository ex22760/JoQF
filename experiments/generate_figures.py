"""
experiments/generate_figures.py
Generates Figures 6-9 for the JF paper.
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import RESULTS_DIR, FIGURES_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)

C_ENS    = "#1f77b4"
C_BASE   = "#ff7f0e"
C_BNH    = "#2ca02c"
C_BEAR   = "#d62728"
C_NEUT   = "#aec7e8"
C_BULL   = "#98df8a"


def load_result(ticker, window_name, vol_spec="ewma_vix", label_scheme="nber"):
    fname = (f"{ticker.replace('^','')}_{window_name}_"
             f"{vol_spec}_{label_scheme}.pkl")
    path  = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"  WARNING: {fname} not found")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def load_all(label_scheme="nber", vol_spec="ewma_vix"):
    results = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
            try:
                r = pickle.load(f)
                if (getattr(r, "label_scheme", "") == label_scheme and
                    getattr(r, "vol_spec", "") == vol_spec):
                    results.append(r)
            except Exception:
                pass
    return results


# ── Figure 6: Equity curves ───────────────────────────────────────────────────

def fig6_equity_curves():
    print("Generating Figure 6: Equity curves...")

    windows = [
        ("Window1_2000s", "2000-2009 (dot-com crash + GFC)"),
        ("Window3_2020s", "2020-2026 (COVID + 2022 bear + Liberation Day)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (wname, label) in zip(axes, windows):
        r = load_result("^GSPC", wname)
        if r is None:
            ax.text(0.5, 0.5, f"No data for {wname}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        idx = r.daily_equity_curve.index if hasattr(r, "daily_equity_curve") and r.daily_equity_curve is not None else None
        if idx is None:
            ax.text(0.5, 0.5, "Equity curve not saved\nRerun Phase 1",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        ens_curve  = r.daily_equity_curve  / r.daily_equity_curve.iloc[0] * 100
        base_curve = r.daily_baseline_curve / r.daily_baseline_curve.iloc[0] * 100
        bnh = (1 + r.daily_returns).cumprod() * 100

        labels = r.daily_labels
        bear_days = labels[labels == 0].index
        if len(bear_days) > 0:
            in_bear = False
            bear_start = None
            for date in idx:
                is_bear = date in bear_days
                if is_bear and not in_bear:
                    bear_start = date
                    in_bear = True
                elif not is_bear and in_bear:
                    ax.axvspan(bear_start, date, alpha=0.12,
                               color=C_BEAR, zorder=0)
                    in_bear = False
            if in_bear:
                ax.axvspan(bear_start, idx[-1], alpha=0.12,
                           color=C_BEAR, zorder=0)

        ax.plot(idx, ens_curve,  color=C_ENS,  linewidth=1.8,
                label=f"Ensemble (Sharpe={r.sharpe:.2f})", zorder=3)
        ax.plot(idx, base_curve, color=C_BASE, linewidth=1.8,
                linestyle="--",
                label=f"HJB Baseline (Sharpe={r.baseline_sharpe:.2f})", zorder=3)
        ax.plot(idx, bnh,        color=C_BNH,  linewidth=1.0,
                linestyle=":", alpha=0.7, label="Buy and Hold", zorder=2)

        ax.set_ylabel("Normalised wealth (base = 100)", fontsize=10)
        ax.set_xlabel(label, fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.grid(True, alpha=0.3)

        if wname == "Window1_2000s":
            ax.axvline(pd.Timestamp("2008-09-15"), color="grey",
                       linestyle=":", linewidth=1, alpha=0.8)
            ax.text(pd.Timestamp("2008-09-15"), ax.get_ylim()[0] * 1.02,
                    "Lehman", fontsize=7, color="grey", rotation=90,
                    va="bottom", ha="right")
        elif wname == "Window3_2020s":
            ax.axvline(pd.Timestamp("2020-03-23"), color="grey",
                       linestyle=":", linewidth=1, alpha=0.8)
            ax.text(pd.Timestamp("2020-03-23"), ax.get_ylim()[0] * 1.02,
                    "COVID low", fontsize=7, color="grey", rotation=90,
                    va="bottom", ha="right")
            ax.axvline(pd.Timestamp("2025-04-02"), color="grey",
                       linestyle=":", linewidth=1, alpha=0.8)
            ax.text(pd.Timestamp("2025-04-02"), ax.get_ylim()[0] * 1.02,
                    "Liberation Day", fontsize=7, color="grey", rotation=90,
                    va="bottom", ha="right")

    bear_patch = mpatches.Patch(color=C_BEAR, alpha=0.2, label="NBER bear period")
    for ax in axes:
        handles, labels_leg = ax.get_legend_handles_labels()
        ax.legend(handles + [bear_patch], labels_leg + ["NBER bear"],
                  fontsize=8, loc="upper left")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig6_equity_curves.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 7: Regime probability time series ──────────────────────────────────

def fig7_regime_proba():
    print("Generating Figure 7: Regime probability time series...")

    r = load_result("^GSPC", "Window3_2020s")
    if r is None or r.daily_ensemble_proba is None:
        print("  Skipping — no proba data")
        return

    idx   = r.daily_returns.index
    proba = r.daily_ensemble_proba
    price = (1 + r.daily_returns).cumprod()
    price = price / price.iloc[0]

    p_bear = proba[:, 0]
    p_neut = proba[:, 1]
    p_bull = proba[:, 2]
    conf   = proba.max(axis=1)

    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(idx, price, color="black", linewidth=1.2, zorder=3)
    labels = r.daily_labels
    for regime, col, alpha in [(0, C_BEAR, 0.25), (1, C_NEUT, 0.15),
                                (2, C_BULL, 0.20)]:
        mask = np.array(labels) == regime
        ax1.fill_between(idx, price.min(), price.max(),
                         where=mask, color=col, alpha=alpha, zorder=1)
    ax1.set_ylabel("S\\&P 500 (normalised)", fontsize=9)
    ax1.set_xlim(idx[0], idx[-1])
    ax1.tick_params(labelbottom=False)
    ax1.grid(True, alpha=0.2)

    libday = pd.Timestamp("2025-04-02")
    if libday in idx:
        ax1.axvline(libday, color="darkred", linewidth=1.5, zorder=4)
        ax1.text(libday, price.max() * 0.97, "Liberation\nDay",
                 fontsize=7.5, color="darkred", ha="center", va="top")

    ax2 = fig.add_subplot(gs[1])
    ax2.stackplot(idx, p_bear, p_neut, p_bull,
                  colors=[C_BEAR, C_NEUT, C_BULL],
                  labels=["P(bear)", "P(neutral)", "P(bull)"],
                  alpha=0.85)
    ax2.set_ylabel("Regime probability", fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(idx[0], idx[-1])
    ax2.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    ax2.tick_params(labelbottom=False)
    ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(idx, conf, color="#6b4fa2", linewidth=1.0, alpha=0.8)
    ax3.axhline(0.55, color="black", linewidth=1, linestyle="--",
                label="0.55 fallback threshold")
    ax3.fill_between(idx, 0.55, conf,
                     where=conf < 0.55, color="orange", alpha=0.4,
                     label="Below threshold (fallback active)")
    ax3.set_ylabel("Confidence score", fontsize=9)
    ax3.set_ylim(0.2, 1.0)
    ax3.set_xlim(idx[0], idx[-1])
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig7_regime_proba.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 8: Quiet bear zoomed illustration ──────────────────────────────────

def fig8_quiet_bear_zoom():
    print("Generating Figure 8: Quiet bear episode illustration...")

    r = load_result("^GSPC", "Window1_2000s")
    if r is None or not hasattr(r, "daily_equity_curve") or r.daily_equity_curve is None:
        print("  Skipping — no equity curve data. Rerun Phase 1.")
        return

    zoom_start = pd.Timestamp("2001-01-01")
    zoom_end   = pd.Timestamp("2003-01-01")

    idx    = r.daily_returns.index
    mask   = (idx >= zoom_start) & (idx <= zoom_end)
    idx_z  = idx[mask]

    w_ens  = np.array(r.daily_weights)[mask]
    w_base = np.array(r.daily_baseline_weights)[mask]
    sigma  = np.array(r.daily_sigma)[mask]
    labels = np.array(r.daily_labels)[mask]
    price  = (1 + r.daily_returns.iloc[np.where(mask)[0]]).cumprod()

    bear_mask  = labels == 0
    if bear_mask.sum() > 0:
        sigma_med  = np.median(sigma[bear_mask])
        quiet_bear = bear_mask & (sigma <= sigma_med)
    else:
        quiet_bear = np.zeros(len(labels), dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    axes[0].plot(idx_z, price / price.iloc[0] * 100,
                 color="black", linewidth=1.5)
    for d, qb in zip(idx_z, quiet_bear):
        if qb:
            axes[0].axvline(d, color=C_BEAR, alpha=0.08, linewidth=1)
    axes[0].set_ylabel("S\\&P 500 (normalised)", fontsize=10)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(idx_z, w_ens,  color=C_ENS,  linewidth=1.8,
                 label="Ensemble weight", zorder=3)
    axes[1].plot(idx_z, w_base, color=C_BASE, linewidth=1.8,
                 linestyle="--", label="HJB Baseline weight", zorder=3)

    in_qb, qb_start = False, None
    for i, (d, qb) in enumerate(zip(idx_z, quiet_bear)):
        if qb and not in_qb:
            qb_start = d
            in_qb = True
        elif not qb and in_qb:
            axes[1].axvspan(qb_start, d, alpha=0.18, color="#ff9900",
                            label="Quiet bear day" if qb_start == idx_z[quiet_bear][0] else "")
            in_qb = False
    if in_qb:
        axes[1].axvspan(qb_start, idx_z[-1], alpha=0.18, color="#ff9900")

    axes[1].set_ylabel("Equity weight", fontsize=10)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(idx_z, sigma, color="#6b4fa2", linewidth=1.2,
                 label="$\\hat{\\sigma}_{ann}$")
    if bear_mask.sum() > 0:
        axes[2].axhline(sigma_med, color="orange", linewidth=1.5,
                        linestyle="--",
                        label=f"Median bear volatility ({sigma_med:.3f})")
    axes[2].fill_between(idx_z, 0, sigma,
                         where=quiet_bear, alpha=0.25, color="orange",
                         label="Quiet bear ($\\sigma$ below median)")
    axes[2].set_ylabel("Realised vol (ann.)", fontsize=10)
    axes[2].set_xlabel("Date", fontsize=10)
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].grid(True, alpha=0.25)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig8_quiet_bear_zoom.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 9: Sharpe-MaxDD scatter ───────────────────────────────────────────

def fig9_sharpe_maxdd_scatter():
    print("Generating Figure 9: Sharpe-MaxDD scatter...")

    results = load_all(label_scheme="nber", vol_spec="ewma_vix")
    if not results:
        print("  No results found")
        return

    asset_colours = {
        "^GSPC":  "#1f77b4",
        "^FTSE":  "#ff7f0e",
        "^GDAXI": "#2ca02c",
        "^N225":  "#d62728",
        "EEM":    "#9467bd",
        "GLD":    "#8c564b",
    }
    asset_labels = {
        "^GSPC":  "S\\&P 500",
        "^FTSE":  "FTSE 100",
        "^GDAXI": "DAX",
        "^N225":  "Nikkei",
        "EEM":    "EEM",
        "GLD":    "GLD",
    }
    window_markers = {
        "Window1_2000s": "o",
        "Window2_2010s": "s",
        "Window3_2020s": "^",
    }

    fig, ax = plt.subplots(figsize=(9, 7))

    plotted_assets  = set()
    plotted_windows = set()

    for r in results:
        ticker   = getattr(r, "ticker", "^GSPC")
        wname    = getattr(r, "window_name", "")
        d_sharpe = r.sharpe - r.baseline_sharpe
        d_maxdd  = r.max_dd - r.baseline_max_dd

        col = asset_colours.get(ticker, "grey")
        mk  = window_markers.get(wname, "o")

        ax.scatter(d_sharpe, d_maxdd * 100,
                   color=col, marker=mk, s=80, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=3)

        plotted_assets.add(ticker)
        plotted_windows.add(wname)

    ax.axvline(0, color="black", linewidth=1, zorder=2)
    ax.axhline(0, color="black", linewidth=1, zorder=2)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[0] + 0.02*(xlim[1]-xlim[0]),
            ylim[1] - 0.04*(ylim[1]-ylim[0]),
            "Worse Sharpe\nBetter MaxDD", ha="left", va="top",
            fontsize=8, color="#ff7f0e", style="italic", fontweight="bold")

    asset_handles = [mpatches.Patch(color=c, label=asset_labels.get(t, t))
                     for t, c in asset_colours.items()
                     if any(getattr(r, "ticker", "") == t for r in results)]
    window_handles = [
        plt.Line2D([0], [0], marker="o", color="grey", linestyle="",
                   markersize=7, label="Window 1 (2000-09)"),
        plt.Line2D([0], [0], marker="s", color="grey", linestyle="",
                   markersize=7, label="Window 2 (2010-19)"),
        plt.Line2D([0], [0], marker="^", color="grey", linestyle="",
                   markersize=7, label="Window 3 (2020-26)"),
    ]

    leg1 = ax.legend(handles=asset_handles, title="Asset",
                     fontsize=8, title_fontsize=9,
                     loc="lower right", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=window_handles, title="Window",
              fontsize=8, title_fontsize=9,
              loc="lower left", framealpha=0.9)

    ax.set_xlabel("$\\Delta$Sharpe (ensemble $-$ baseline)", fontsize=11)
    ax.set_ylabel("$\\Delta$MaxDD in percentage points\n(negative = ensemble better)",
                  fontsize=11)
    ax.grid(True, alpha=0.25)

    out = os.path.join(FIGURES_DIR, "fig9_sharpe_maxdd_scatter.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_generate_figures():
    print("\n" + "="*60)
    print("GENERATING PAPER FIGURES 6-9")
    print("="*60)
    fig6_equity_curves()
    fig7_regime_proba()
    fig8_quiet_bear_zoom()
    fig9_sharpe_maxdd_scatter()
    print("\nAll figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    run_generate_figures()