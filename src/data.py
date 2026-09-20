"""
Data acquisition layer.

Three independent sources, each with a documented reason for being used:

  1. Yahoo Finance (via yfinance) -- equity and ETF total-return prices.
     `auto_adjust=True` gives split- and dividend-adjusted closes, so the
     constituent returns and the index returns are on the same (total-return)
     footing. Mixing price-return constituents with a total-return index would
     bias the variance decomposition.

  2. CBOE's public index CDN -- daily history for VIX, VIX9D, VIX3M, VIX6M.
     Yahoo's ^VIX3M / ^VIX9D / ^VIX6M tickers return a SINGLE row (today only),
     verified empirically, so they cannot support a historical study. The CBOE
     CDN is the authoritative publisher of these indices.

  3. FRED -- VIXCLS (cross-check on CBOE's VIX) and VXVCLS. VXVCLS is the same
     index as VIX3M under its pre-2017 name (VXV) and starts 2007-12-04 versus
     the CBOE file's 2009-09-18. Splicing recovers the 2008 crisis, which is
     the single most important correlation event in the sample.

All network results are cached to CSV so the analysis is reproducible offline
and so re-runs do not hammer the providers.
"""

from __future__ import annotations

import io
import time
import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import requests

import config

_HEADERS = {"User-Agent": "Mozilla/5.0 (academic research script)"}


# --------------------------------------------------------------------------
# Generic cache helper
# --------------------------------------------------------------------------
def _cache_path(key: str):
    return config.CACHE_DIR / f"{key}.csv"


def _read_cache(key: str) -> pd.DataFrame | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df if len(df) else None


def _write_cache(key: str, df: pd.DataFrame) -> None:
    df.to_csv(_cache_path(key))


# --------------------------------------------------------------------------
# Equity / ETF prices
# --------------------------------------------------------------------------
def fetch_prices(
    tickers: Sequence[str],
    start: str = config.START,
    end: str = config.END,
    cache_key: str = "prices",
    force: bool = False,
    max_retries: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """Download adjusted closes for `tickers`.

    Returns (prices, dropped) where `dropped` lists tickers that could not be
    retrieved. Failures are reported rather than silently swallowed: a ticker
    vanishing usually means a corporate action (acquisition / delisting), and
    that is information about the panel, not a nuisance.

    Retries exist because Yahoo intermittently 404s valid symbols under load --
    observed during development for CB, which succeeded on a later attempt.
    """
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))  # de-duplicate, keep order

    cached = None if force else _read_cache(cache_key)
    if cached is not None:
        have = [t for t in tickers if t in cached.columns]
        if len(have) == len(tickers):
            return cached[tickers].loc[str(start):str(end)], []
        # Partial cache: fetch ONLY what is missing and merge. Without this, a
        # transient network failure (observed for CAT and CVX) permanently
        # amputates those names from the panel, because the truncated result
        # is what gets cached.
        missing_from_cache = [t for t in tickers if t not in cached.columns]
        print(f"    cache hit for {len(have)}/{len(tickers)}; "
              f"fetching {len(missing_from_cache)} missing: {missing_from_cache}")
        tickers = missing_from_cache

    frames = []
    dropped = []
    if cached is not None:
        frames.append(cached)
    # Batch in chunks: large single requests are more likely to time out, and
    # a chunk failure then costs us only that chunk.
    chunk = 25
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        got = None
        for attempt in range(max_retries):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = yf.download(
                        batch, start=start, end=end, auto_adjust=True,
                        progress=False, threads=True, group_by="column",
                    )
                if raw is not None and len(raw):
                    got = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
                    break
            except Exception as exc:  # noqa: BLE001 - we want to retry anything
                print(f"    [retry {attempt+1}/{max_retries}] batch failed: {type(exc).__name__}")
            time.sleep(2 * (attempt + 1))
        if got is None:
            dropped.extend(batch)
            continue
        if isinstance(got, pd.Series):
            got = got.to_frame(batch[0])
        frames.append(got)

    if not frames:
        raise RuntimeError("No price data could be downloaded at all.")

    for i, f in enumerate(frames):
        f.index = pd.to_datetime(f.index)
        if getattr(f.index, "tz", None) is not None:
            frames[i] = f.tz_localize(None)

    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]

    # A column that is entirely NaN means the symbol did not resolve.
    empty = [c for c in px.columns if px[c].notna().sum() == 0]
    dropped.extend(empty)
    px = px.drop(columns=empty)

    missing = [t for t in tickers if t not in px.columns]
    dropped = sorted(set(dropped) | set(missing))

    px = px.sort_index()
    px.index = pd.to_datetime(px.index)
    if getattr(px.index, "tz", None) is not None:
        px = px.tz_localize(None)
    _write_cache(cache_key, px)
    return px.loc[str(start):str(end)], dropped


# --------------------------------------------------------------------------
# CBOE volatility indices
# --------------------------------------------------------------------------
def fetch_cboe_index(name: str, force: bool = False) -> pd.Series:
    """Daily close of a CBOE volatility index from the public CDN."""
    key = f"cboe_{name}"
    if not force:
        cached = _read_cache(key)
        if cached is not None:
            return cached.iloc[:, 0]

    url = config.CBOE_CDN.format(name=name)
    r = requests.get(url, headers=_HEADERS, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    s = df.set_index("DATE")["CLOSE"].astype(float).sort_index()
    s.name = name
    _write_cache(key, s.to_frame())
    return s


def fetch_fred(series_id: str, force: bool = False) -> pd.Series:
    """Daily series from FRED's public CSV endpoint (no API key required)."""
    key = f"fred_{series_id}"
    if not force:
        cached = _read_cache(key)
        if cached is not None:
            return cached.iloc[:, 0]

    url = config.FRED_CSV.format(sid=series_id)
    r = requests.get(url, headers=_HEADERS, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    dcol, vcol = df.columns[0], df.columns[1]
    df[dcol] = pd.to_datetime(df[dcol])
    # FRED marks holidays with "."; coerce turns them into NaN.
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
    s = df.set_index(dcol)[vcol].dropna().sort_index()
    s.name = series_id
    _write_cache(key, s.to_frame())
    return s


def load_term_structure(force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Assemble the VIX term-structure panel.

    Returns (frame, diagnostics). The VIX3M column is spliced: CBOE's own file
    where it exists, FRED's VXVCLS before that. The overlap is checked and the
    agreement statistics are returned so the splice can be defended rather than
    assumed.
    """
    diag: dict = {}

    vix = fetch_cboe_index("VIX", force=force)
    vix9d = fetch_cboe_index("VIX9D", force=force)
    vix3m_cboe = fetch_cboe_index("VIX3M", force=force)
    vix6m = fetch_cboe_index("VIX6M", force=force)
    vxv_fred = fetch_fred("VXVCLS", force=force)

    # --- validate the splice on the overlapping sample -------------------
    ov = pd.concat([vix3m_cboe.rename("cboe"), vxv_fred.rename("fred")],
                   axis=1).dropna()
    diag["splice_overlap_days"] = int(len(ov))
    if len(ov):
        diff = ov["cboe"] - ov["fred"]
        diag["splice_mean_abs_diff"] = float(diff.abs().mean())
        diag["splice_max_abs_diff"] = float(diff.abs().max())
        diag["splice_corr"] = float(ov["cboe"].corr(ov["fred"]))

    # CBOE takes precedence where available; FRED only fills the earlier tail.
    vix3m = vix3m_cboe.combine_first(vxv_fred).sort_index()
    diag["vix3m_start_spliced"] = str(vix3m.dropna().index[0].date())
    diag["vix3m_start_cboe_only"] = str(vix3m_cboe.dropna().index[0].date())

    # Cross-check CBOE's VIX against FRED's VIXCLS as an independent read.
    vixcls = fetch_fred("VIXCLS", force=force)
    cmp = pd.concat([vix.rename("cboe"), vixcls.rename("fred")], axis=1).dropna()
    diag["vix_crosscheck_days"] = int(len(cmp))
    diag["vix_crosscheck_mean_abs_diff"] = float((cmp["cboe"] - cmp["fred"]).abs().mean())

    out = pd.DataFrame(
        {"VIX": vix, "VIX9D": vix9d, "VIX3M": vix3m, "VIX6M": vix6m}
    ).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.loc[config.START : config.END]
    return out, diag


# --------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------
def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Arithmetic (simple) returns.

    METHODOLOGICAL NOTE -- this is deliberate and matters.

    The variance decomposition rests on the portfolio identity

        R_index = sum_i w_i * R_i

    which is EXACT for simple returns and FALSE for log returns, because
    log(1 + sum_i w_i r_i) != sum_i w_i log(1 + r_i). Volatility work often
    defaults to log returns (they are additive across *time*), but here we need
    additivity across *assets*, so simple returns are the correct choice. At
    daily horizons the numerical difference is second order, but using log
    returns would make the identity we are inverting only approximately true,
    and the whole point of this study is to invert it exactly.
    """
    return prices.pct_change(fill_method=None)


def align_panel(
    prices: pd.DataFrame, min_obs: int = 252
) -> pd.DataFrame:
    """Drop columns with too little history to ever enter a rolling window."""
    keep = [c for c in prices.columns if prices[c].notna().sum() >= min_obs]
    return prices[keep]


def coverage_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker first/last observation and count -- printed in the run log so
    data problems are visible rather than buried."""
    rows = []
    for c in prices.columns:
        s = prices[c].dropna()
        rows.append(
            {
                "ticker": c,
                "start": s.index[0].date() if len(s) else None,
                "end": s.index[-1].date() if len(s) else None,
                "n": len(s),
            }
        )
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
