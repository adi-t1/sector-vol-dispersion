"""
VIX term structure, used as an INDEPENDENT sanity check.

The logic of the check. The variance decomposition is estimated entirely from
realised equity returns. The VIX complex is computed from S&P 500 OPTION prices
-- a different instrument, a different market, a forward-looking rather than
backward-looking measurement. So the term structure is genuinely independent
evidence, not a restatement of the same data.

The known stylised fact we are testing against: the VIX curve normally slopes
upward (VIX3M > VIX, "contango") because near-dated implied variance is
cheaper than far-dated in calm markets; it inverts into backwardation when a
systematic shock hits, because the market prices an immediate spike in
correlated risk that is expected to decay. If our correlation-driven episodes
are real systematic events, they should coincide with backwardation. If our
dispersion-driven episodes are real, they should not.

A caveat we state rather than bury: backwardation is driven by the LEVEL and
expected persistence of index volatility, not by correlation as such. The two
are tightly linked -- index vol cannot spike far without correlation rising,
which is the whole point of the decomposition -- but the check is a
consistency test, not a proof of causation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def build_term_structure(ts: pd.DataFrame) -> pd.DataFrame:
    """Add slope measures and regime flags to the raw index panel.

    `slope_3m_1m` = VIX3M / VIX is the headline measure:
        > 1  contango      (normal, upward sloping)
        < 1  backwardation (stressed, inverted)
    A ratio rather than a difference is used so the measure is comparable
    across vol levels -- a 2-point inversion at VIX 15 is a far bigger signal
    than a 2-point inversion at VIX 60.
    """
    out = ts.copy()
    out["slope_3m_1m"] = out["VIX3M"] / out["VIX"]
    out["slope_1m_9d"] = out["VIX"] / out["VIX9D"]
    out["slope_6m_3m"] = out["VIX6M"] / out["VIX3M"]
    out["backwardation"] = out["slope_3m_1m"] < 1.0
    out["deep_backwardation"] = out["slope_3m_1m"] < 0.95
    return out


def align(rolling: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the decomposition output with the term structure on dates.

    An inner join (rather than reindex-and-fill) is deliberate: filling a
    stale VIX level forward across a gap would manufacture agreement on
    precisely the illiquid days where the check is most fragile.
    """
    return rolling.join(ts, how="inner")


def regime_contingency(
    df: pd.DataFrame,
    classification_col: str = "episode_type",
    corr_label: str = "correlation-driven",
    disp_label: str = "dispersion-driven",
) -> dict:
    """2x2 table of episode type against term-structure state, plus a test.

    Fisher's exact test is used rather than chi-square because episode counts
    are small (there are only a handful of genuine crises in 27 years) and the
    chi-square approximation is unreliable in that regime. The odds ratio is
    reported because it is the effect size that actually answers the question:
    how many times more likely is backwardation during a correlation-driven
    episode?
    """
    sub = df[df[classification_col].isin([corr_label, disp_label])]
    sub = sub[sub["backwardation"].notna()]
    if sub.empty:
        return {"error": "no overlapping observations"}

    table = pd.crosstab(sub[classification_col], sub["backwardation"])

    # Use reindex, NOT df[[False, True]]. When the column labels are booleans
    # and the frame happens to have as many rows as labels, pandas interprets
    # df[[False, True]] as a boolean ROW MASK rather than a column selection
    # and silently returns the wrong rows -- which it did here, zeroing out the
    # entire correlation-driven row of a 2x2 table without raising anything.
    # reindex is unambiguous about which axis is being addressed.
    table = table.reindex(columns=[False, True], fill_value=0)
    table = table.reindex(index=[corr_label, disp_label], fill_value=0)
    table = table.fillna(0).astype(int)

    mat = table.to_numpy()
    try:
        odds, p = stats.fisher_exact(mat)
    except Exception:
        odds, p = np.nan, np.nan

    n_corr = mat[0].sum()
    n_disp = mat[1].sum()
    return {
        "table": table,
        "n_corr_driven": int(n_corr),
        "n_disp_driven": int(n_disp),
        "pct_backwardation_corr": float(mat[0, 1] / n_corr * 100) if n_corr else np.nan,
        "pct_backwardation_disp": float(mat[1, 1] / n_disp * 100) if n_disp else np.nan,
        "odds_ratio": float(odds),
        "fisher_p": float(p),
    }


def slope_by_rho_quintile(df: pd.DataFrame, rho_col: str = "rho_implied") -> pd.DataFrame:
    """Mean term-structure slope within quintiles of realised correlation.

    This is the continuous version of the check and is more informative than
    the 2x2 table because it uses every day in the sample rather than only the
    extreme episodes -- if the relationship is real it should be monotone, not
    just present at the tails.
    """
    sub = df[[rho_col, "slope_3m_1m", "VIX", "backwardation"]].dropna()
    if sub.empty:
        return pd.DataFrame()
    q = pd.qcut(sub[rho_col], 5, labels=[f"Q{i}" for i in range(1, 6)])
    g = sub.groupby(q, observed=True)
    return pd.DataFrame(
        {
            "n": g.size(),
            "mean_rho": g[rho_col].mean(),
            "mean_slope_3m_1m": g["slope_3m_1m"].mean(),
            "median_slope_3m_1m": g["slope_3m_1m"].median(),
            "pct_backwardated": g["backwardation"].mean() * 100,
            "mean_vix": g["VIX"].mean(),
        }
    )


def lead_lag_profile(
    df: pd.DataFrame,
    rho_col: str = "rho_implied",
    slope_col: str = "slope_3m_1m",
    max_lag: int = 100,
    step: int = 1,
) -> pd.DataFrame:
    """corr(slope_t, rho_{t+k}) across leads and lags k.

    This exists to resolve a real discrepancy rather than to decorate the
    study. At the level of discrete episodes our classification agrees with the
    term structure perfectly, but the DAY-level contingency table shows no
    relationship at all. The explanation is a timing mismatch between the two
    instruments, and this function measures it:

      * The VIX curve is option-implied and FORWARD-looking. It inverts on the
        day the market decides the next month is dangerous.
      * Realised correlation from a trailing 63-day window is BACKWARD-looking.
        On the first day of a crisis, 62 of the 63 returns in the window are
        still pre-crisis, so the estimate barely moves.

    If that is the right explanation, the association should be strongest when
    realised correlation is shifted FORWARD by something on the order of half
    the estimation window, and it is: the correlation strengthens from about
    -0.32 contemporaneously to about -0.41 at a lag of 20-40 days. The sign is
    negative throughout because a higher correlation goes with a flatter or
    inverted curve.

    This is a property of the measurement, not a tradable lead-lag claim: a
    trailing estimator cannot be used to anticipate anything.
    """
    sub = df[[rho_col, slope_col]].dropna()
    rows = []
    for k in range(-max_lag, max_lag + 1, step):
        rows.append({"lag_days": k,
                     "corr": float(sub[slope_col].corr(sub[rho_col].shift(-k)))})
    out = pd.DataFrame(rows)
    out.attrs["peak_lag"] = int(out.loc[out["corr"].abs().idxmax(), "lag_days"])
    out.attrs["peak_corr"] = float(out.loc[out["corr"].abs().idxmax(), "corr"])
    out.attrs["contemporaneous"] = float(
        out.loc[out["lag_days"] == 0, "corr"].iloc[0])
    return out


def partial_check(df: pd.DataFrame, rho_col: str = "rho_implied") -> dict:
    """Does correlation say anything about the curve BEYOND the vol level?

    This is the sharp version of the sanity check, and the one worth defending.
    Backwardation is mostly a function of how high VIX is. If rho only predicts
    the curve because rho and VIX move together, the check is circular. So we
    regress the slope on log(VIX) alone, then on log(VIX) plus rho, and ask
    whether rho carries incremental information.

    We report the partial correlation of slope and rho after removing log(VIX)
    from both, which is exactly that incremental content.
    """
    sub = df[[rho_col, "slope_3m_1m", "VIX"]].dropna()
    if len(sub) < 100:
        return {"error": "insufficient data"}

    x = np.log(sub["VIX"].to_numpy())
    y = sub["slope_3m_1m"].to_numpy()
    r = sub[rho_col].to_numpy()

    def resid(v):
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        return v - X @ beta

    ry, rr = resid(y), resid(r)
    partial = float(np.corrcoef(ry, rr)[0, 1])
    raw = float(np.corrcoef(y, r)[0, 1])
    # Approximate significance of the partial correlation: t with n-3 df.
    n = len(sub)
    t = partial * np.sqrt((n - 3) / max(1e-12, 1 - partial ** 2))
    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 3)))
    return {
        "n": n,
        "raw_corr_slope_rho": raw,
        "partial_corr_given_logvix": partial,
        "t_stat": float(t),
        "p_value": p,
        "corr_rho_logvix": float(np.corrcoef(r, x)[0, 1]),
    }
