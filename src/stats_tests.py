r"""
Significance testing for correlation regime shifts.

THE FISHER z-TRANSFORM. A sample correlation is bounded in [-1, 1] and its
sampling distribution is badly skewed near the bounds: the sampling error of a
correlation of 0.95 is nothing like that of a correlation of 0.05, so raw
correlations cannot be compared, differenced, or averaged as if they were
normal variates. Fisher's transform

    z = arctanh(rho) = 0.5 * ln((1 + rho) / (1 - rho))

is variance-stabilising: z is approximately normal with variance 1/(n - 3),
essentially independent of the true rho. Differences in z are therefore
testable with ordinary normal theory.

WHY THE TEXTBOOK TEST IS NOT ENOUGH HERE, and what we do about it.

The 1/(n-3) result assumes n INDEPENDENT bivariate-normal observations, and it
prices the sampling error of ONE correlation. Applied to this study, three of
those assumptions fail -- and, importantly, they do not all fail in the same
direction:

  1. Serial dependence. Daily equity returns are not independent: volatility
     clusters hard. The effective number of independent observations in a
     63-day window is meaningfully below 63. This biases the test towards
     being ANTI-CONSERVATIVE -- too many false regime shifts.

  2. Cross-sectional averaging. We are not testing one pair; we are testing an
     average over N(N-1)/2 pairs. Averaging reduces sampling variance, so the
     single-pair variance 1/(n-3) OVERSTATES the uncertainty in rho_bar. This
     biases the test the opposite way, towards being CONSERVATIVE. The pairs
     share constituents and are far from independent, so the reduction is
     nowhere near the 1/K an independence assumption would give, and there is
     no clean closed form for what it actually is.

  3. Non-normality. Returns are fat-tailed, especially in exactly the crisis
     windows we most want to test.

WHICH EFFECT WINS IS AN EMPIRICAL QUESTION, and in this sample the answer is
"it depends on the episode". Measured against the block bootstrap, the naive
interval is about 30% too WIDE at the median -- effect 2 dominating -- but it
is 1.8x too NARROW for March 2020 and 1.5x too narrow for the 2025 tariff
shock, the two most violent episodes, where effect 1 takes over. That is
precisely why all three tests are reported rather than a preferred one: any
single number here would be hiding a real ambiguity.

So we report THREE tests side by side and let the disagreement between them be
part of the finding:

  (a) `fisher_two_sample` -- the textbook test, as specified. Reported because
      it is the standard, and because its gap from (c) quantifies how badly the
      iid assumption flatters the result.
  (b) `fisher_two_sample_adjusted` -- same test with n replaced by an effective
      sample size derived from the autocorrelation of the underlying series
      (a Newey-West style correction). Cheap, and removes most of problem 1.
  (c) `stationary_bootstrap_rho_diff` -- the honest benchmark. Resamples whole
      BLOCKS of days (Politis & Romano 1994), which preserves volatility
      clustering and the entire cross-sectional correlation structure, and
      recomputes the average correlation from scratch on each resample. It
      makes no normality, independence or equicorrelation assumption.

For the rolling confidence BAND we use the plain 1/(n-3) width, and label it as
such on the chart. By effect 2 above it is conservatively wide for rho_bar in
calm conditions; in the middle of a crisis, effect 1 means it should not be
read as a strict bound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.decomposition import weighted_average_correlation


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------
def fisher_z(rho):
    """arctanh with the bounds clipped just inside +-1 so that a correlation of
    exactly 1 (which happens with degenerate windows) yields a large finite z
    rather than an inf that silently poisons downstream arithmetic."""
    r = np.clip(np.asarray(rho, dtype=float), -0.999999, 0.999999)
    return np.arctanh(r)


def inv_fisher_z(z):
    return np.tanh(np.asarray(z, dtype=float))


def fisher_se(n: int) -> float:
    if n <= 3:
        return np.nan
    return 1.0 / np.sqrt(n - 3)


def fisher_ci(rho, n: int, alpha: float = 0.05):
    """Confidence interval for a correlation, computed in z-space and mapped
    back. Mapping back is what keeps the interval inside [-1, 1] and correctly
    asymmetric -- a symmetric interval in rho-space would be wrong."""
    z = fisher_z(rho)
    se = fisher_se(n)
    crit = stats.norm.ppf(1 - alpha / 2)
    return inv_fisher_z(z - crit * se), inv_fisher_z(z + crit * se)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def fisher_one_sample(rho: float, n: int, rho0: float) -> dict:
    """Test H0: true correlation equals rho0."""
    se = fisher_se(n)
    z = (fisher_z(rho) - fisher_z(rho0)) / se
    return {
        "rho": float(rho),
        "rho0": float(rho0),
        "n": int(n),
        "z_stat": float(z),
        "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))),
    }


def fisher_two_sample(rho1: float, n1: int, rho2: float, n2: int) -> dict:
    """Test H0: the two periods have the same true correlation.

        Z = (z1 - z2) / sqrt(1/(n1-3) + 1/(n2-3))
    """
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z1, z2 = fisher_z(rho1), fisher_z(rho2)
    z = (z1 - z2) / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "rho_1": float(rho1), "n_1": int(n1),
        "rho_2": float(rho2), "n_2": int(n2),
        "z_1": float(z1), "z_2": float(z2),
        "delta_z": float(z1 - z2),
        "se": float(se),
        "z_stat": float(z),
        "p_value": p,
    }


# --------------------------------------------------------------------------
# Effective sample size
# --------------------------------------------------------------------------
def effective_sample_size(x: np.ndarray, max_lag: int | None = None) -> float:
    """Sample size adjusted for serial dependence.

        n_eff = n / (1 + 2 * sum_k w_k * acf_k)

    with Bartlett weights w_k = 1 - k/(L+1), the standard Newey-West kernel.
    Using squared/absolute returns as the input captures volatility clustering,
    which is the dependence that actually matters for a variance-based
    statistic -- raw returns are close to serially uncorrelated and would
    wrongly suggest no adjustment is needed. That choice is deliberate.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float(n)
    if max_lag is None:
        max_lag = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 0:
        return float(n)
    total = 0.0
    for k in range(1, min(max_lag, n - 1) + 1):
        acf = float(x[k:] @ x[:-k]) / denom
        total += (1.0 - k / (max_lag + 1.0)) * acf
    factor = 1.0 + 2.0 * total
    if factor <= 0:
        return float(n)
    return float(min(n, n / factor))


def fisher_two_sample_adjusted(
    rho1: float, ret1: np.ndarray, rho2: float, ret2: np.ndarray
) -> dict:
    """Fisher two-sample test with n replaced by an effective sample size.

    The effective size is computed from the cross-sectional mean of SQUARED
    returns, i.e. from the realised-variance process, because that is the
    series whose persistence contaminates a variance/correlation estimate.
    """
    v1 = np.nanmean(np.asarray(ret1, dtype=float) ** 2, axis=1)
    v2 = np.nanmean(np.asarray(ret2, dtype=float) ** 2, axis=1)
    n1 = effective_sample_size(v1)
    n2 = effective_sample_size(v2)
    out = fisher_two_sample(rho1, max(int(round(n1)), 5), rho2, max(int(round(n2)), 5))
    out["n_1_raw"] = int(len(v1))
    out["n_2_raw"] = int(len(v2))
    out["n_1_eff"] = float(n1)
    out["n_2_eff"] = float(n2)
    out["shrink_1"] = float(n1 / len(v1))
    out["shrink_2"] = float(n2 / len(v2))
    return out


# --------------------------------------------------------------------------
# Stationary bootstrap
# --------------------------------------------------------------------------
def _stationary_bootstrap_idx(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices.

    Block lengths are Geometric(1/block), so the resampled series is stationary
    (unlike a fixed-block bootstrap, which is not) and the expected block
    length is `block`. Wrapping at the end keeps every observation equally
    likely to be sampled.
    """
    p = 1.0 / block
    idx = np.empty(n, dtype=np.int64)
    i = rng.integers(0, n)
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = rng.integers(0, n)
        else:
            i = (i + 1) % n
    return idx


def _avg_corr(ret: np.ndarray, w: np.ndarray | None = None) -> float:
    """Covariance-weighted average pairwise correlation of a return block."""
    if ret.shape[0] < 5:
        return np.nan
    cov = np.cov(ret, rowvar=False, ddof=1)
    s = np.sqrt(np.diag(cov))
    if np.any(s <= 0):
        return np.nan
    corr = cov / np.outer(s, s)
    np.fill_diagonal(corr, 1.0)
    if w is None:
        w = np.full(ret.shape[1], 1.0 / ret.shape[1])
    return weighted_average_correlation(w, s, corr)


def stationary_bootstrap_rho_diff(
    ret1: np.ndarray,
    ret2: np.ndarray,
    block: int = 63,
    draws: int = 5000,
    seed: int = 0,
    w1: np.ndarray | None = None,
    w2: np.ndarray | None = None,
) -> dict:
    """Bootstrap distribution of (rho_period1 - rho_period2).

    Each period is resampled independently in blocks, preserving within-period
    volatility clustering and the full cross-sectional dependence. The p-value
    is the two-sided percentile probability that the difference has the
    opposite sign to the one observed, which is the standard bootstrap test of
    H0: no difference.
    """
    rng = np.random.default_rng(seed)
    ret1 = np.asarray(ret1, dtype=float)
    ret2 = np.asarray(ret2, dtype=float)

    obs = _avg_corr(ret1, w1) - _avg_corr(ret2, w2)
    diffs = np.empty(draws)
    n1, n2 = len(ret1), len(ret2)
    for d in range(draws):
        a = _avg_corr(ret1[_stationary_bootstrap_idx(n1, block, rng)], w1)
        b = _avg_corr(ret2[_stationary_bootstrap_idx(n2, block, rng)], w2)
        diffs[d] = a - b

    diffs = diffs[np.isfinite(diffs)]
    centred = diffs - diffs.mean()
    # p-value from the null (centred) distribution: how often does a
    # no-difference world produce a gap at least as large as the observed one?
    p = float((np.abs(centred) >= abs(obs)).mean())
    return {
        "rho_1": float(_avg_corr(ret1, w1)),
        "rho_2": float(_avg_corr(ret2, w2)),
        "observed_diff": float(obs),
        "boot_mean": float(diffs.mean()),
        "boot_se": float(diffs.std(ddof=1)),
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "p_value": p,
        "draws": int(len(diffs)),
        "block": int(block),
    }


# --------------------------------------------------------------------------
# Convenience: compare two dated periods of a return panel
# --------------------------------------------------------------------------
def compare_periods(
    returns: pd.DataFrame,
    period_a: tuple[str, str],
    period_b: tuple[str, str],
    label_a: str = "A",
    label_b: str = "B",
    block: int = 63,
    draws: int = 2000,
    seed: int = 0,
    min_names: int = 8,
) -> dict:
    """Run all three tests on two calendar periods of the same panel.

    Names are restricted to those with complete data in BOTH periods, so the
    comparison is like-for-like: a change in panel composition must not be
    allowed to masquerade as a change in correlation.
    """
    a = returns.loc[period_a[0] : period_a[1]].dropna(how="all")
    b = returns.loc[period_b[0] : period_b[1]].dropna(how="all")

    # Eligibility on COVERAGE, not on zero-NaN. Requiring a column to be
    # entirely non-null made a single missing day -- e.g. the unavoidable NaN
    # on the first row of a pct_change series -- disqualify every name and
    # silently drop the whole comparison. That bug cost the dot-com test.
    # Names must be well covered in both periods; the rows that remain NaN are
    # then dropped jointly so both correlation matrices come from one clean
    # rectangular sample.
    min_cov = 0.95
    common = [
        c for c in returns.columns
        if a[c].notna().mean() >= min_cov and b[c].notna().mean() >= min_cov
    ]
    if len(common) < min_names or len(a) < 10 or len(b) < 10:
        return {"label_a": label_a, "label_b": label_b, "error": "insufficient data",
                "n_names": len(common), "n_days_a": len(a), "n_days_b": len(b)}

    A = a[common].dropna().to_numpy()
    B = b[common].dropna().to_numpy()
    if len(A) < 10 or len(B) < 10:
        return {"label_a": label_a, "label_b": label_b, "error": "insufficient data",
                "n_names": len(common), "n_days_a": len(A), "n_days_b": len(B)}
    rho_a, rho_b = _avg_corr(A), _avg_corr(B)

    naive = fisher_two_sample(rho_a, len(A), rho_b, len(B))
    adj = fisher_two_sample_adjusted(rho_a, A, rho_b, B)
    boot = stationary_bootstrap_rho_diff(A, B, block=block, draws=draws, seed=seed)

    return {
        "label_a": label_a,
        "label_b": label_b,
        "period_a": f"{period_a[0]}..{period_a[1]}",
        "period_b": f"{period_b[0]}..{period_b[1]}",
        "n_names": len(common),
        "n_days_a": len(A),
        "n_days_b": len(B),
        "rho_a": rho_a,
        "rho_b": rho_b,
        "delta_rho": rho_a - rho_b,
        "naive_z": naive["z_stat"],
        "naive_p": naive["p_value"],
        "adj_n_a": adj["n_1_eff"],
        "adj_n_b": adj["n_2_eff"],
        "adj_z": adj["z_stat"],
        "adj_p": adj["p_value"],
        "boot_ci_lo": boot["ci_lo"],
        "boot_ci_hi": boot["ci_hi"],
        "boot_p": boot["p_value"],
    }
