"""
Index weight estimation.

THE PROBLEM. The decomposition needs w_i, the weight of each sector in the
S&P 500. Actual historical GICS sector weights are a licensed S&P product and
are not available free. We need a defensible substitute.

THE OPTIONS CONSIDERED, and why we rejected them:

  (a) Equal weights on the 11 sector ETFs. Clean and exact, but it is simply
      not the S&P 500 -- it materially understates technology and overstates
      materials/utilities, and the whole late-sample story is about technology
      concentration. We do run this as a robustness case (`equal_weights`)
      precisely because it is exact, but it cannot be the headline.

  (b) Market-cap weights reconstructed from price x shares outstanding.
      Historical share counts are not freely available either, and holding
      shares fixed at today's count back-projects buybacks and issuance into
      history, which is a worse error than the one it fixes.

  (c) Returns-based style analysis (Sharpe 1992): recover the weights by
      regressing index returns on component returns subject to w >= 0 and
      sum(w) = 1. This is the standard technique for inferring a portfolio's
      composition when only its returns are observable. It is what we use.

WHY (c) IS DEFENSIBLE HERE. The S&P 500 *is* exactly a weighted combination of
its sectors, so the regression is not a loose factor model -- it is estimating
the coefficients of an identity that genuinely holds (up to the tracking error
between a sector ETF and the corresponding S&P sector sub-index, and up to
weight drift inside the estimation window). The constraints encode real prior
information: index weights cannot be negative and must sum to one. The fit
quality is therefore itself a test, and we report it: if the R-squared were not
near 1, the approach would be invalid. It is ~0.99+, which is reported in the
run log and in the research note.

WHAT CAN STILL GO WRONG, and how we check it:
  * Collinearity among sector returns can make individual weights unstable even
    when the fit is excellent. Mitigated by (i) a 252-day window, (ii) monthly
    rather than daily re-estimation, (iii) the non-negativity constraint, which
    is a strong regulariser. We report the weight series so instability is
    visible, and we sanity-check the estimated technology weight against
    publicly known S&P technology weights.
  * Weight drift within the window biases towards the window average. With
    quarterly-scale drift and a one-year window this is small relative to the
    effects being studied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def constrained_weights(
    y: np.ndarray, X: np.ndarray, penalty_scale: float = 1e3
) -> tuple[np.ndarray, float]:
    """Solve  min ||Xw - y||^2  s.t.  w >= 0,  sum(w) = 1.

    The sum-to-one equality is imposed by augmenting the system with a row of
    constant M and a target of M; as M grows, any deviation from sum(w)=1 is
    penalised arbitrarily hard, so the solution converges to the constrained
    optimum. M is scaled to the data so the penalty is large *relative to* the
    residual scale rather than large in absolute terms, which keeps the
    augmented problem well conditioned.

    Returns (weights, r_squared).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[1]

    m = penalty_scale * max(np.abs(X).max(), 1e-12)
    X_aug = np.vstack([X, np.full((1, n), m)])
    y_aug = np.concatenate([y, [m]])

    w, _ = nnls(X_aug, y_aug)
    total = w.sum()
    if total <= 0:
        # Degenerate: fall back to equal weights rather than emit garbage.
        w = np.full(n, 1.0 / n)
    else:
        w = w / total  # remove any residual penalty slack

    resid = y - X @ w
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return w, r2


def rolling_index_weights(
    index_ret: pd.Series,
    component_ret: pd.DataFrame,
    window: int,
    step: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Estimate sector weights on a rolling, strictly backward-looking basis.

    At each estimation date t the fit uses returns from the window ENDING at t
    (inclusive) and the resulting weights are applied from t onward -- never
    before. Between estimation dates weights are held constant (forward-filled),
    which is exactly how a real portfolio behaves between rebalances and
    guarantees no forward-looking information enters.

    Returns (weights_df, r2_series), both indexed by estimation date.
    """
    idx = component_ret.index
    rows, r2s, dates = [], [], []

    for i in range(window - 1, len(idx), step):
        t = idx[i]
        sl = slice(i - window + 1, i + 1)
        Xw = component_ret.iloc[sl]
        yw = index_ret.iloc[sl]

        # Only components with complete data in this window are eligible.
        # This is how XLRE (2015) and XLC (2018) enter the universe.
        cols = [c for c in Xw.columns if Xw[c].notna().all()]
        both = yw.notna()
        if len(cols) < 2 or both.sum() < window * 0.9:
            continue
        Xv = Xw.loc[both, cols].to_numpy()
        yv = yw[both].to_numpy()
        if not np.isfinite(Xv).all() or not np.isfinite(yv).all():
            continue

        w, r2 = constrained_weights(yv, Xv)
        rows.append(pd.Series(w, index=cols))
        r2s.append(r2)
        dates.append(t)

    W = pd.DataFrame(rows, index=pd.DatetimeIndex(dates)).fillna(0.0)
    return W, pd.Series(r2s, index=pd.DatetimeIndex(dates), name="weight_fit_r2")


def expand_weights(W: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill estimated weights onto the daily grid.

    Forward-fill (never interpolate, never backfill) is what keeps the
    computation causal: the weight in force on any day is the one estimated at
    the most recent *past* estimation date.
    """
    return W.reindex(target_index.union(W.index)).ffill().reindex(target_index)


def equal_weights(component_ret: pd.DataFrame) -> pd.DataFrame:
    """Equal weight across components available on each date (robustness case)."""
    avail = component_ret.notna()
    counts = avail.sum(axis=1).replace(0, np.nan)
    return avail.div(counts, axis=0).fillna(0.0)
