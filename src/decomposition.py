r"""
The variance decomposition engine.

This module implements, and makes numerically testable, the identity

    Var(R_p) = sum_i w_i^2 sigma_i^2  +  sum_{i != j} w_i w_j rho_ij sigma_i sigma_j
               \_______ own ________/    \_________ cross-covariance _____________/

and its inversion for an average pairwise correlation.

--------------------------------------------------------------------------
A NAMING WARNING THAT MATTERS
--------------------------------------------------------------------------
The first term is routinely called the "idiosyncratic" term and the second the
"systematic" term. That is loose language and it is worth being precise about,
because it is the first thing a careful reader will push back on.

`sum_i w_i^2 sigma_i^2` is the OWN-VARIANCE term. Each sigma_i^2 is a stock's
TOTAL variance -- it already contains that stock's exposure to the market
factor. So the own-variance term is not "idiosyncratic risk" in the CAPM sense
(the variance of a residual after projecting on a factor). What this identity
actually separates is:

    * risk that diversification kills   (own-variance, decays like 1/N), versus
    * risk that diversification cannot kill (cross-covariance, does not decay).

That is a *diversification* decomposition, not a *factor-model* decomposition.
The two coincide only in the limiting case where the single-name residuals are
mutually uncorrelated and the factor loadings are the whole story.

We therefore report the average pairwise correlation rho_avg as the primary
object of interest -- it is the quantity that is invariant to this ambiguity --
and we cross-validate it against a genuine factor-style measure (the absorption
ratio, i.e. the share of total variance captured by the first principal
component of the correlation matrix) in `absorption_ratio`. Throughout the
code, "own" and "cross" are used for the mathematically exact terms, and
"idiosyncratic"/"systematic" only as shorthand in output labels.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def own_variance_term(w: np.ndarray, sigma: np.ndarray) -> float:
    """sum_i w_i^2 sigma_i^2 -- the diversifiable ("idiosyncratic") term."""
    w = np.asarray(w, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return float(np.sum((w * sigma) ** 2))


def cross_base(w: np.ndarray, sigma: np.ndarray) -> float:
    """sum_{i != j} w_i w_j sigma_i sigma_j.

    Uses the closed form

        sum_{i != j} w_i w_j s_i s_j
            = (sum_i w_i s_i)^2 - sum_i w_i^2 s_i^2

    which is both O(N) instead of O(N^2) and, more importantly, the algebraic
    step that makes the derivation in the research note transparent: the double
    sum over ordered pairs is just "square of the total, minus the diagonal".
    """
    w = np.asarray(w, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    ws = w * sigma
    return float(ws.sum() ** 2 - np.sum(ws ** 2))


def portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
    """Exact w' Sigma w. Used as ground truth in the unit tests."""
    w = np.asarray(w, dtype=float)
    return float(w @ np.asarray(cov, dtype=float) @ w)


def weighted_average_correlation(
    w: np.ndarray, sigma: np.ndarray, corr: np.ndarray
) -> float:
    """The covariance-weighted mean of the off-diagonal correlations,

        rho_bar = sum_{i!=j} w_i w_j s_i s_j rho_ij / sum_{i!=j} w_i w_j s_i s_j

    This is the quantity the inversion below recovers EXACTLY when the index
    variance is the true portfolio variance. It is *not* the simple unweighted
    mean of the correlation matrix's off-diagonal entries -- those differ
    whenever weights or vols are unequal, and conflating them is a common and
    consequential error.
    """
    w = np.asarray(w, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    corr = np.asarray(corr, dtype=float)

    ws = np.outer(w * sigma, w * sigma)
    off = ~np.eye(len(w), dtype=bool)
    denom = ws[off].sum()
    if abs(denom) < 1e-18:
        return np.nan
    return float((ws[off] * corr[off]).sum() / denom)


def simple_average_correlation(corr: np.ndarray) -> float:
    """Unweighted mean of off-diagonal correlations -- reported alongside the
    weighted version so the gap between the two is visible."""
    corr = np.asarray(corr, dtype=float)
    off = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.nanmean(corr[off]))


def implied_average_correlation(
    var_index: float, w: np.ndarray, sigma: np.ndarray
) -> float:
    """Invert the identity for rho_avg:

        rho_avg = (sigma_index^2 - sum_i w_i^2 sigma_i^2)
                  / sum_{i != j} w_i w_j sigma_i sigma_j

    This is the realised analogue of CBOE's implied correlation index
    methodology (which does the same inversion with option-implied vols).
    """
    denom = cross_base(w, sigma)
    if abs(denom) < 1e-18:
        return np.nan
    return float((var_index - own_variance_term(w, sigma)) / denom)


def absorption_ratio(corr: np.ndarray) -> float:
    """Share of total variance explained by the first principal component of
    the correlation matrix (Kritzman, Li, Page & Rigobon 2011).

    Included as an INDEPENDENT check on rho_avg. The two are conceptually
    distinct -- rho_avg is a pairwise average, the absorption ratio is a
    factor-structure measure -- so if they move together we have real evidence
    of a correlation regime rather than an artefact of one estimator.

    For an equicorrelated matrix with correlation rho and N assets, the first
    eigenvalue is 1 + (N-1)rho, giving AR = (1 + (N-1)rho)/N, which is a
    monotone map of rho. Departures from that map measure how far the real
    correlation structure is from equicorrelation.
    """
    corr = np.asarray(corr, dtype=float)
    if np.isnan(corr).any():
        return np.nan
    evals = np.linalg.eigvalsh(corr)
    return float(evals[-1] / evals.sum())


def equicorrelation_absorption(rho: float, n: int) -> float:
    """AR implied by equicorrelation rho -- the benchmark curve for the check
    described in `absorption_ratio`."""
    return (1.0 + (n - 1) * rho) / n


# --------------------------------------------------------------------------
# Full decomposition
# --------------------------------------------------------------------------
@dataclass
class Decomposition:
    n: int
    var_index: float
    vol_index: float
    own_term: float
    cross_term_base: float
    rho_implied: float
    rho_weighted_actual: float
    rho_simple_actual: float
    own_share: float
    cross_share: float
    avg_constituent_vol: float
    dispersion_ratio: float
    h_ratio: float
    comovement_C: float
    absorption: float
    residual: float

    def to_dict(self) -> dict:
        return asdict(self)


def decompose(
    returns: np.ndarray,
    w: np.ndarray,
    var_index: float | None = None,
    annualization: int = 252,
) -> Decomposition:
    """Full decomposition for one estimation window.

    Parameters
    ----------
    returns : (T, N) array of SIMPLE returns for the constituents.
    w       : (N,) weights. Not required to sum to 1 by the maths, but they are
              normalised here because every interpretation below assumes a
              fully-invested portfolio.
    var_index : optional externally-observed index variance (e.g. realised
              variance of SPY). If omitted, the exact portfolio variance
              w' Sigma w is used, in which case `rho_implied` must equal
              `rho_weighted_actual` to machine precision -- that equality is
              the engine's core unit test.

    All variances/vols returned are ANNUALISED.
    """
    returns = np.asarray(returns, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    n = returns.shape[1]

    # ddof=1: we are estimating the mean from the same sample, so the unbiased
    # divisor is (T-1). With T=63 the difference from ddof=0 is ~1.6% in
    # variance -- small, but it is a free correction and it keeps the identity
    # internally consistent, since the same ddof is used for cov and for the
    # index variance computed from the portfolio return series.
    cov = np.cov(returns, rowvar=False, ddof=1) * annualization
    sigma = np.sqrt(np.diag(cov))

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.outer(sigma, sigma)
        corr = np.where(denom > 0, cov / denom, np.nan)
    np.fill_diagonal(corr, 1.0)

    exact_var = portfolio_variance(w, cov)
    if var_index is None:
        var_index = exact_var

    own = own_variance_term(w, sigma)
    base = cross_base(w, sigma)
    rho_imp = implied_average_correlation(var_index, w, sigma)
    rho_w = weighted_average_correlation(w, sigma, corr)
    rho_s = simple_average_correlation(corr)

    cross_actual = rho_imp * base
    avg_vol = float(np.sum(w * sigma))

    # dispersion_ratio: 1 - sigma_index / sum_i w_i sigma_i.
    # Interpretation: the fraction of the weighted-average single-name vol that
    # is "destroyed" by imperfect correlation. It is 0 when everything moves
    # together (rho = 1) and approaches 1 as names decouple. This is the
    # quantity a dispersion trade is long.
    vol_index = float(np.sqrt(var_index)) if var_index > 0 else np.nan
    disp = 1.0 - vol_index / avg_vol if avg_vol > 0 else np.nan

    # Re-express the identity in a SCALE-FREE form. Writing A = sum_i w_i s_i
    # (the weighted average single-name vol) and h = O / A^2, substituting
    # B = A^2 - O into sigma^2 = O + rho*B gives the exact factorisation
    #
    #       sigma_index^2 = A^2 * [ rho + (1 - rho) h ]  ==  A^2 * C
    #
    # so that       log sigma_index = log A + 0.5 * log C.
    #
    # This matters a great deal for attribution. Attributing changes in the
    # variance LEVEL is dominated by A, because A is unbounded while rho is
    # confined to [0, 1]: in any large vol spike the volatility term must
    # mechanically dominate, and a level-based attribution therefore labels
    # every crisis "volatility-driven" regardless of what correlation did.
    # The log form separates a pure scale channel (A) from a pure co-movement
    # channel (C), is exactly additive, and is what `episodes` attributes on.
    h = own / (avg_vol ** 2) if avg_vol > 0 else np.nan
    c = var_index / (avg_vol ** 2) if avg_vol > 0 else np.nan

    return Decomposition(
        n=n,
        var_index=float(var_index),
        vol_index=vol_index,
        own_term=own,
        cross_term_base=base,
        rho_implied=rho_imp,
        rho_weighted_actual=rho_w,
        rho_simple_actual=rho_s,
        own_share=float(own / var_index) if var_index > 0 else np.nan,
        cross_share=float(cross_actual / var_index) if var_index > 0 else np.nan,
        avg_constituent_vol=avg_vol,
        dispersion_ratio=float(disp),
        h_ratio=float(h),
        comovement_C=float(c),
        absorption=absorption_ratio(corr),
        # Gap between the observed index variance and the variance implied by
        # the constituents. Non-zero only when var_index is supplied
        # externally; it measures weight misspecification / incomplete
        # constituent coverage and is reported rather than hidden.
        residual=float(var_index - exact_var),
    )


# --------------------------------------------------------------------------
# The 1/N result, made numerical
# --------------------------------------------------------------------------
def equal_weight_terms(sigma_bar: float, rho: float, n: int) -> dict:
    """Own vs cross term for an equally weighted portfolio of n assets that all
    have volatility sigma_bar and pairwise correlation rho.

        own   = (1/n) * sigma_bar^2                 -> 0 as n grows
        cross = (1 - 1/n) * rho * sigma_bar^2       -> rho * sigma_bar^2

    so total variance -> rho * sigma_bar^2, a floor that no amount of
    diversification removes. This is the exact statement of why diversification
    kills idiosyncratic but not systematic risk, and it is produced here
    numerically so the research note's claim can be checked, not just asserted.
    """
    own = sigma_bar ** 2 / n
    cross = (1.0 - 1.0 / n) * rho * sigma_bar ** 2
    total = own + cross
    return {
        "n": n,
        "own": own,
        "cross": cross,
        "total_var": total,
        "total_vol": np.sqrt(total),
        "own_share": own / total if total > 0 else np.nan,
        "floor_vol": np.sqrt(rho) * sigma_bar,
    }
