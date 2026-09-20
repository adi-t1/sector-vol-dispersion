"""
Rolling application of the decomposition through history.

CAUSALITY IS THE WHOLE GAME HERE. Every estimate dated t uses only returns in
the closed window [t - W + 1, t]. Nothing at t+1 or later touches it. This is
enforced structurally by slicing positionally from the end of the window rather
than by centring a window, and it is verified by `assert_causal`, which
re-computes a handful of dates on a truncated sample and checks the answers are
bit-identical. That test is not decoration: a forward-looking "rolling"
correlation is the single most common way this kind of study goes wrong, and it
would make every episode classification worthless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.decomposition import decompose


def rolling_decomposition(
    returns: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    index_returns: pd.Series | None = None,
    window: int = 63,
    annualization: int = 252,
    min_names: int = 8,
    label: str = "",
) -> pd.DataFrame:
    """Run the decomposition on a trailing window ending at each date.

    Parameters
    ----------
    returns : (T, N) constituent simple returns.
    weights : (T, N) weights in force on each date. If None, equal weights
              across the names admissible in that window.
    index_returns : optional externally observed index return series (e.g.
              SPY). If given, the index variance is measured from it and the
              decomposition's `residual` reports how far the constituents +
              weights fall short of explaining it. If None, the index is taken
              to be the constituent portfolio itself and the identity is exact.
    min_names : windows with fewer admissible names are skipped rather than
              reported, because the 1/N own-variance term dominates in small
              panels and the "index" stops resembling a diversified portfolio.

    Returns a DataFrame indexed by date with one column per decomposition field
    plus `n_names`.
    """
    idx = returns.index
    out_rows, out_dates = [], []

    ret_vals = returns.to_numpy()
    cols = list(returns.columns)

    for i in range(window - 1, len(idx)):
        t = idx[i]
        sl = slice(i - window + 1, i + 1)
        block = ret_vals[sl]

        # Admissible names: complete data across the whole window. Requiring
        # completeness (rather than, say, 90%) means every pairwise covariance
        # in the matrix is estimated on the SAME sample, which keeps the
        # covariance matrix internally consistent and positive semi-definite.
        ok = np.isfinite(block).all(axis=0)
        if ok.sum() < min_names:
            continue

        sub = block[:, ok]
        names = [c for c, k in zip(cols, ok) if k]

        if weights is None:
            w = np.full(len(names), 1.0 / len(names))
        else:
            w_row = weights.iloc[i].reindex(names).to_numpy(dtype=float)
            w_row = np.nan_to_num(w_row, nan=0.0)
            if w_row.sum() <= 0:
                continue
            w = w_row / w_row.sum()

        var_idx = None
        if index_returns is not None:
            iw = index_returns.iloc[sl]
            if iw.notna().sum() < window:
                continue
            var_idx = float(np.var(iw.to_numpy(), ddof=1) * annualization)

        d = decompose(sub, w, var_index=var_idx, annualization=annualization)
        row = d.to_dict()
        row["n_names"] = int(ok.sum())
        out_rows.append(row)
        out_dates.append(t)

    df = pd.DataFrame(out_rows, index=pd.DatetimeIndex(out_dates))
    df.index.name = "date"
    if label:
        df.attrs["label"] = label
    return df


def flag_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostics for the implied correlation.

    rho_implied is a ratio of two estimated quantities and is NOT mechanically
    bounded to [-1, 1] when the index variance is observed externally rather
    than computed from the constituents. Values outside the bound are not
    "bad data" to be clipped away silently -- they are direct evidence that the
    weights or the constituent set do not fully span the index. We count and
    report them.
    """
    s = df["rho_implied"]
    return pd.DataFrame(
        {
            "n_obs": [int(s.notna().sum())],
            "n_above_1": [int((s > 1).sum())],
            "n_below_0": [int((s < 0).sum())],
            "n_below_neg1": [int((s < -1).sum())],
            "pct_out_of_unit": [float(((s > 1) | (s < 0)).mean() * 100)],
            "min": [float(s.min())],
            "max": [float(s.max())],
            "mean": [float(s.mean())],
            "median": [float(s.median())],
        }
    )


def assert_causal(
    returns: pd.DataFrame,
    window: int,
    annualization: int = 252,
    min_names: int = 8,
    n_checks: int = 3,
) -> bool:
    """Verify no look-ahead: recomputing on data truncated at t must reproduce
    the value at t exactly.

    If the rolling code ever accidentally used centred windows, future returns,
    or a full-sample standardisation, truncating the input would change the
    answer at t. This checks it does not.
    """
    full = rolling_decomposition(
        returns, window=window, annualization=annualization, min_names=min_names
    )
    if full.empty:
        raise RuntimeError("causality check has no rows to test")

    positions = np.linspace(0.4, 0.95, n_checks)
    for p in positions:
        t = full.index[int(p * (len(full) - 1))]
        truncated = returns.loc[:t]
        part = rolling_decomposition(
            truncated, window=window, annualization=annualization,
            min_names=min_names,
        )
        a = full.loc[t]
        b = part.loc[t]
        for col in ("rho_implied", "own_share", "vol_index", "absorption"):
            va, vb = a[col], b[col]
            if pd.isna(va) and pd.isna(vb):
                continue
            if not np.isclose(va, vb, rtol=1e-12, atol=1e-15):
                raise AssertionError(
                    f"LOOK-AHEAD DETECTED at {t.date()} in {col}: "
                    f"full={va!r} truncated={vb!r}"
                )
    return True


def attribute_variance_change(
    df: pd.DataFrame, horizon: int = 21
) -> pd.DataFrame:
    """Attribute the change in index variance to correlation vs single-name vol.

    Index variance is  V = O + rho * B,  where O = sum w_i^2 s_i^2 and
    B = sum_{i!=j} w_i w_j s_i s_j. Both O and B depend only on volatilities;
    rho enters linearly. So over a horizon h the change decomposes as

        dV = [contribution of d(rho)] + [contribution of d(vols)] + interaction

    The split is path-dependent (you get different numbers holding rho at its
    start value vs its end value), so we use the SYMMETRIC / Shapley form --
    the average of the two orderings -- which is order-invariant and leaves no
    unexplained interaction term:

        dV_rho  = 0.5 * (B_0 + B_1) * (rho_1 - rho_0)
        dV_vol  = 0.5 * (rho_0 + rho_1) * (B_1 - B_0) + (O_1 - O_0)

    and dV_rho + dV_vol = V_1 - V_0 exactly (verified in `attrib_check`).

    Choosing the symmetric decomposition over a simple "hold the other fixed"
    version matters: during March 2020 both rho and vols moved violently, so
    the interaction term is large and an asymmetric split would attribute it
    arbitrarily to whichever factor happened to be held fixed.
    """
    o = df["own_term"]
    b = df["cross_term_base"]
    r = df["rho_implied"]
    v = df["var_index"]

    o0, b0, r0, v0 = o.shift(horizon), b.shift(horizon), r.shift(horizon), v.shift(horizon)

    d_rho = 0.5 * (b0 + b) * (r - r0)
    d_vol = 0.5 * (r0 + r) * (b - b0) + (o - o0)
    d_tot = v - v0

    out = pd.DataFrame(
        {
            "d_var": d_tot,
            "d_var_from_corr": d_rho,
            "d_var_from_vol": d_vol,
            "check": d_rho + d_vol - d_tot,
        }
    )
    # Share of the move explained by correlation. Only meaningful when the
    # total move is non-trivial, so we mask small denominators rather than
    # letting the ratio explode.
    denom = d_tot.abs()
    scale = denom.rolling(252, min_periods=60).median()
    meaningful = denom > scale
    out["corr_share"] = np.where(
        meaningful & (denom > 0), d_rho / d_tot, np.nan
    )
    return out


def attrib_check(attrib: pd.DataFrame, tol: float = 1e-10) -> float:
    """Max absolute residual of the Shapley attribution -- must be ~0."""
    return float(attrib["check"].abs().max())
