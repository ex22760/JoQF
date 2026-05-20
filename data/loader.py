"""
data/loader.py
Download and cache price + macro data for any ticker.
All data cached locally so experiments re-run without API calls.
"""

import os, pickle
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import pandas_datareader.data as web
    # Test it actually works (version conflicts can cause import-time errors)
    PDR_AVAILABLE = True
except Exception:
    PDR_AVAILABLE = False

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import DATA_DIR, MACRO_LAGS


def load_price_data(ticker: str,
                    start: str = "1988-01-01",
                    end:   str = "2026-06-01",  # wide range, cached once
                    force_refresh: bool = False) -> pd.DataFrame:
    """
    Download OHLCV data for ticker from Yahoo Finance.
    Returns DataFrame with columns: Open, High, Low, Close, Volume.
    Cached to DATA_DIR/prices/{ticker}.pkl.
    """
    cache_path = os.path.join(DATA_DIR, "prices", f"{ticker.replace('^','')}.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path) and not force_refresh:
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
        # Stale check: if cache ends before 2019, re-download
        if not df.empty and df.index[-1].year < 2019:
            print(f"  [loader] {ticker}: cache is stale (ends {df.index[-1].date()}), re-downloading...")
        else:
            print(f"  [loader] {ticker}: loaded from cache ({len(df)} rows)")
            return df

    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    print(f"  [loader] {ticker}: downloading from Yahoo Finance...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna(how="all")

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)

    print(f"  [loader] {ticker}: downloaded {len(df)} rows, cached to {cache_path}")
    return df


def load_macro_data(start: str = "1988-01-01",
                    end:   str = "2026-03-01",
                    force_refresh: bool = False) -> pd.DataFrame:
    """
    Download CPI, unemployment, Federal Funds Rate from FRED.
    Returns daily DataFrame with publication lags applied.
    Cached to DATA_DIR/macro/macro.pkl.
    """
    cache_path = os.path.join(DATA_DIR, "macro", "macro.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path) and not force_refresh:
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
        print(f"  [loader] macro: loaded from cache ({len(df)} rows)")
        return df

    print("  [loader] macro: downloading from FRED...")

    series = {
        "CPI":          "CPIAUCSL",
        "Unemployment": "UNRATE",
        "FedFunds":     "FEDFUNDS",
        "USREC":        "USREC",
    }

    frames = {}
    for name, fred_code in series.items():
        s = None
        # Try pandas_datareader first
        if PDR_AVAILABLE:
            try:
                s = web.DataReader(fred_code, "fred", start, end).iloc[:, 0]
            except Exception as e:
                print(f"  [loader] pandas_datareader failed for {fred_code}: {e}")

        # Fallback: download directly from FRED API via requests
        if s is None:
            try:
                import requests
                url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
                       f"?id={fred_code}&vintage_date={end[:10]}")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                from io import StringIO
                s = pd.read_csv(StringIO(resp.text),
                                parse_dates=[0], index_col=0).iloc[:, 0]
                s = s.replace(".", float("nan")).astype(float)
                print(f"  [loader] {fred_code}: downloaded via FRED API")
            except Exception as e:
                print(f"  [loader] WARNING: FRED API failed for {fred_code}: {e}")

        if s is not None:
            frames[name] = s

    if not frames:
        print("  [loader] WARNING: No macro data available. "
              "Continuing without macro features.")
        daily_idx = pd.date_range(start, end, freq="B")
        macro = pd.DataFrame(index=daily_idx)
        with open(cache_path, "wb") as f:
            pickle.dump(macro, f)
        return macro

    macro = pd.DataFrame(frames)
    # Reindex to daily, forward-fill
    daily_idx = pd.date_range(start, end, freq="B")
    macro = macro.reindex(daily_idx).ffill()

    with open(cache_path, "wb") as f:
        pickle.dump(macro, f)

    print(f"  [loader] macro: downloaded {len(macro)} rows, cached")
    return macro


def load_vix_data(start: str = "1988-01-01",
                  end:   str = "2026-03-01",
                  force_refresh: bool = False) -> pd.Series:
    """Download VIX from Yahoo Finance. Returns daily Series."""
    df = load_price_data("^VIX", start=start, end=end,  # cache keyed by end date
                         force_refresh=force_refresh)
    return df["Close"].rename("VIX")


def load_usrec(start: str = "1988-01-01",
               end:   str = "2026-03-01") -> pd.Series:
    """
    Load NBER recession indicator (USREC).
    Returns daily binary Series (1 = recession, 0 = expansion).
    Tries FRED first, falls back to local USREC.csv.
    """
    try:
        macro = load_macro_data(start=start, end=end)
        if "USREC" in macro.columns:
            return macro["USREC"].fillna(0).astype(int)
    except Exception:
        pass

    # Fallback: local CSV
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "USREC.csv")
    if os.path.exists(csv_path):
        s = pd.read_csv(csv_path, parse_dates=[0], index_col=0).iloc[:, 0]
        daily_idx = pd.date_range(start, end, freq="B")
        return s.reindex(daily_idx, method="ffill").fillna(0).astype(int)

    print("  [loader] WARNING: USREC not available, defaulting to zeros")
    daily_idx = pd.date_range(start, end, freq="B")
    return pd.Series(0, index=daily_idx, name="USREC")


def apply_macro_lags(macro: pd.DataFrame,
                     lags: dict = None) -> pd.DataFrame:
    """
    Apply empirical publication lags to macro series (causal constraint).
    Default lags: CPI=21 days, Unemployment=7 days, FedFunds=1 day.
    """
    if lags is None:
        lags = MACRO_LAGS
    result = macro.copy()
    for col, lag in lags.items():
        if col in result.columns:
            result[col] = result[col].shift(lag)
    return result.ffill()


def assemble_dataset(ticker:        str,
                     start:         str = "1988-01-01",
                     end:           str = "2026-06-01",  # always wide — slicing done in pipeline
                     include_macro: bool = True,
                     include_vix:   bool = True,
                     force_refresh: bool = False) -> pd.DataFrame:
    """
    Assemble a complete dataset for a given ticker.
    Returns daily DataFrame with:
      Close, LogReturn, Return, [macro cols], [VIX]
    All macro series have publication lags applied.
    """
    price = load_price_data(ticker, start=start, end=end,
                            force_refresh=force_refresh)
    df = pd.DataFrame(index=price.index)
    df["Close"]     = price["Close"].astype(float)
    df["LogReturn"] = np.log(df["Close"]).diff()
    df["Return"]    = df["Close"].pct_change()

    if include_macro:
        try:
            macro = load_macro_data(start=start, end=end,
                                    force_refresh=force_refresh)
            macro_lagged = apply_macro_lags(macro)
            for col in ["CPI", "Unemployment", "FedFunds"]:
                if col in macro_lagged.columns:
                    df[col] = macro_lagged[col].reindex(df.index).ffill()
        except Exception as e:
            print(f"  [loader] WARNING: macro data unavailable: {e}")

    if include_vix:
        try:
            vix = load_vix_data(start=start, end=end,
                                force_refresh=force_refresh)
            df["VIX"] = vix.reindex(df.index).ffill()
        except Exception as e:
            print(f"  [loader] WARNING: VIX data unavailable: {e}")

    df = df.dropna(subset=["Close", "LogReturn"])
    print(f"  [loader] {ticker}: assembled {len(df)} rows "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df