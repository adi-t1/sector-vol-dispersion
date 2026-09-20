"""
Central configuration for the sector volatility-dispersion study.

Every parameter that represents a *methodological judgment call* is documented
inline with the reasoning behind it, because those choices -- not the code --
are what a reviewer will interrogate.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

for _d in (CACHE_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Sample period
# --------------------------------------------------------------------------
# The nine original SPDR sector ETFs began trading 1998-12-22. We start the
# return sample at 1999-01-04 so that every series begins on a common footing;
# the first rolling estimate then appears one window later (~2000-04).
START = "1999-01-04"
END = "2026-09-16"

# --------------------------------------------------------------------------
# Universe: index level
# --------------------------------------------------------------------------
INDEX_PROXY = "SPY"  # tradable total-return proxy for the S&P 500

# SPDR sector ETFs. XLRE (2015-10) and XLC (2018-06) did not exist for most of
# the sample: real estate sat inside financials, and communication services was
# carved out of technology / consumer discretionary in the 2018 GICS revision.
# The weight estimator handles this by using only the ETFs actually trading on
# each date.
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# --------------------------------------------------------------------------
# Universe: sector level (single-name constituent panels)
# --------------------------------------------------------------------------
# Six sectors chosen to span the economically interesting range:
#   Technology  - high idiosyncratic content, product/earnings-cycle driven
#   Energy      - dominated by one common factor (crude), should show HIGH rho
#   Financials  - the sector that actually blew up in 2008; rate-sensitive
#   Health Care - binary, firm-specific events (trial results, patent cliffs)
#   Staples     - defensive, low vol, moderate correlation
#   Industrials - broad cyclical / macro beta
#
# Names were screened for (a) large-cap membership in the sector, (b) long
# price history, (c) still trading as of 2026-09. Tickers Yahoo no longer
# resolves (MRO, HES, PXD, XEC -- all acquired; BK, CTRA -- symbol resolution
# failures) were dropped. THIS INTRODUCES SURVIVORSHIP BIAS, discussed at
# length in the research note: the panel over-represents firms that made it to
# 2026, which mechanically understates realised idiosyncratic risk.
#
# Late entrants are deliberately retained. The rolling engine admits a name
# only once it has a complete trailing window, so NVDA enters in 2000, GS in
# 2000, ACN in 2002, CRM in 2005, KMI/MPC/PSX/FANG in 2012-13. An unbalanced
# panel is strictly more faithful than forcing a 1999-complete balanced one.
SECTOR_CONSTITUENTS = {
    "Technology": [
        "MSFT", "AAPL", "NVDA", "ORCL", "CSCO", "IBM", "INTC", "TXN", "ADBE",
        "QCOM", "AMAT", "MU", "ADI", "HPQ", "AMD", "ACN", "CRM", "LRCX",
        "KLAC", "NTAP",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "HAL", "VLO", "APA", "DVN",
        "BKR", "PSX", "MPC", "KMI", "WMB", "OKE", "FANG",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "SCHW",
        "TFC", "MET", "ALL", "TRV", "AIG", "AFL", "STT", "CB", "CINF",
    ],
    "Health Care": [
        "JNJ", "PFE", "MRK", "ABT", "LLY", "BMY", "AMGN", "UNH", "MDT",
        "GILD", "CVS", "BDX", "SYK", "CI", "HUM", "BSX", "ZBH", "BAX",
        "REGN", "VRTX",
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "WMT", "COST", "CL", "MO", "KMB", "GIS", "SYY",
        "HSY", "CAG", "CHD", "CLX", "STZ", "TSN", "MKC", "ADM", "EL", "KR",
    ],
    "Industrials": [
        "GE", "HON", "UNP", "CAT", "MMM", "BA", "LMT", "RTX", "DE", "EMR",
        "ITW", "NSC", "CSX", "FDX", "UPS", "NOC", "GD", "ETN", "PH", "ROK",
    ],
}

# Map each constituent panel to the ETF that tracks it.
SECTOR_PANEL_TO_ETF = {
    "Technology": "XLK",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
}

# --------------------------------------------------------------------------
# Volatility / correlation estimation
# --------------------------------------------------------------------------
# WINDOW = 63 trading days (one quarter).
#
# Why 63 and not 21 or 252:
#   * Precision. The standard error of a volatility estimate from n
#     observations is approximately sigma / sqrt(2n). At n=63 that is ~8.9% of
#     the level -- tight enough that regime moves in rho are not swamped by
#     estimation noise. At n=21 it is ~15.4%, and the implied-correlation
#     inversion (a ratio of two noisy quantities) becomes visibly unstable.
#   * Resolution. 252 days would smear March 2020 across a full year and
#     destroy exactly the event structure this study is about.
#   * The Fisher z-test needs n > 3 and its normal approximation is decent by
#     n ~ 30-60, so 63 sits comfortably inside the asymptotic regime.
#   * One quarter also aligns with the earnings cycle, the natural clock for
#     idiosyncratic dispersion.
# Robustness runs at 21 and 126 days are produced and reported.
WINDOW = 63
ROBUSTNESS_WINDOWS = (21, 126)

ANNUALIZATION = 252

# Minimum fraction of the window a name must have data for to be admitted.
# 1.0 (complete data) avoids mixing estimation samples of different length
# inside a single covariance calculation.
MIN_WINDOW_COVERAGE = 1.0

# Minimum number of admissible names before a sector-level estimate is
# reported. Below ~8 names the equal-weight idiosyncratic term (which decays
# like 1/N) is too large for the panel to behave like a diversified sector.
MIN_NAMES = 8

# Rolling weight estimation (index level): window over which SPY returns are
# regressed on sector-ETF returns.
# 252 days: S&P sector weights drift over quarters, not days, so a long window
# buys stability at little cost in timeliness.
WEIGHT_WINDOW = 252
WEIGHT_STEP = 21  # re-estimate monthly; daily re-estimation adds noise, not signal

# --------------------------------------------------------------------------
# Attribution / episode classification
# --------------------------------------------------------------------------
# Horizon over which changes in index variance are attributed to correlation
# vs single-name volatility. 21 days = one month.
ATTRIB_HORIZON = 21

# An episode requires index vol in the top decile of its own history.
VOL_SPIKE_QUANTILE = 0.90

# --------------------------------------------------------------------------
# VIX term structure
# --------------------------------------------------------------------------
# CBOE publishes daily history for its volatility indices on a public CDN.
# Yahoo ^VIX3M / ^VIX9D / ^VIX6M return only the current day, so they are
# unusable for a historical study -- verified, see the research note.
CBOE_CDN = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# FRED VXVCLS is the same underlying index as CBOE VIX3M (renamed from VXV in
# 2017) and starts 2007-12-04, ~21 months earlier than the CBOE CDN file.
# Splicing extends the term-structure sample back through the entire global
# financial crisis. The splice is validated on the overlap before use.
TERM_STRUCTURE_SERIES = ["VIX", "VIX9D", "VIX3M", "VIX6M"]

# --------------------------------------------------------------------------
# Statistical testing
# --------------------------------------------------------------------------
# Block length for the stationary bootstrap used as the robust benchmark
# against the (anti-conservative) textbook Fisher test. 63 days matches the
# estimation window, so a block preserves the autocorrelation induced by
# overlapping windows.
BOOTSTRAP_BLOCK = 63
BOOTSTRAP_DRAWS = 5000
RANDOM_SEED = 20260916

# --------------------------------------------------------------------------
# Event calendar used to annotate the time-series plots.
# --------------------------------------------------------------------------
EVENTS = [
    ("2000-03-10", "Dot-com peak"),
    ("2001-09-17", "9/11 reopen"),
    ("2002-07-23", "WorldCom / accounting lows"),
    ("2008-09-15", "Lehman"),
    ("2010-05-06", "Flash crash"),
    ("2011-08-08", "US downgrade / EU crisis"),
    ("2015-08-24", "China deval"),
    ("2018-02-05", "Volmageddon"),
    ("2018-12-24", "Q4-18 tightening scare"),
    ("2020-03-16", "COVID crash"),
    ("2020-11-09", "Vaccine / value rotation"),
    ("2022-06-13", "CPI shock / hiking cycle"),
    ("2023-03-13", "SVB / regional banks"),
    ("2024-08-05", "Yen carry unwind"),
    ("2025-04-07", "Tariff shock"),
]
