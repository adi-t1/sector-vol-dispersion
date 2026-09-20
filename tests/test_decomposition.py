"""
Unit tests for the variance decomposition engine.

The central test is `test_inversion_is_exact`: when the index variance IS the
portfolio variance of the constituents, the inversion

    rho_avg = (sigma_p^2 - sum w_i^2 sigma_i^2) / sum_{i!=j} w_i w_j s_i s_j

must reproduce the covariance-weighted average pairwise correlation to machine
precision. If that fails, every number in the study is wrong. Everything else
here guards against the specific ways this kind of code silently breaks
(ddof mismatches, unnormalised weights, diagonal handling in the double sum).

Run with:  python -m pytest tests -q      (or)  python tests/test_decomposition.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.decomposition import (  # noqa: E402
    absorption_ratio,
    cross_base,
    decompose,
    equal_weight_terms,
    equicorrelation_absorption,
    implied_average_correlation,
    own_variance_term,
    portfolio_variance,
    simple_average_correlation,
    weighted_average_correlation,
)

RNG = np.random.default_rng(12345)
ANN = 252


def _random_returns(t=500, n=8, seed=0):
    """Correlated Gaussian returns with heterogeneous vols."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    corr = a @ a.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    vols = rng.uniform(0.10, 0.60, size=n) / np.sqrt(ANN)
    cov = corr * np.outer(vols, vols)
    chol = np.linalg.cholesky(cov)
    return rng.normal(size=(t, n)) @ chol.T


def test_cross_base_matches_explicit_double_sum():
    w = RNG.uniform(0.05, 1.0, 7)
    w /= w.sum()
    s = RNG.uniform(0.1, 0.5, 7)
    explicit = sum(
        w[i] * w[j] * s[i] * s[j]
        for i in range(7)
        for j in range(7)
        if i != j
    )
    assert np.isclose(cross_base(w, s), explicit, rtol=1e-12, atol=1e-15)
    print("PASS cross_base closed form == explicit double sum")


def test_own_plus_cross_equals_portfolio_variance():
    """own + sum_{i!=j} w_i w_j rho_ij s_i s_j must equal w' Sigma w exactly."""
    r = _random_returns(seed=1)
    n = r.shape[1]
    w = RNG.uniform(0.05, 1.0, n)
    w /= w.sum()
    cov = np.cov(r, rowvar=False, ddof=1) * ANN
    s = np.sqrt(np.diag(cov))
    corr = cov / np.outer(s, s)

    off = ~np.eye(n, dtype=bool)
    cross_exact = (np.outer(w * s, w * s)[off] * corr[off]).sum()
    total = own_variance_term(w, s) + cross_exact
    assert np.isclose(total, portfolio_variance(w, cov), rtol=1e-12)
    print("PASS own + cross == w' Sigma w")


def test_inversion_is_exact():
    """THE core test: recovered rho == covariance-weighted average rho."""
    for seed in range(6):
        r = _random_returns(seed=seed)
        n = r.shape[1]
        w = RNG.uniform(0.05, 1.0, n)
        w /= w.sum()
        cov = np.cov(r, rowvar=False, ddof=1) * ANN
        s = np.sqrt(np.diag(cov))
        corr = cov / np.outer(s, s)

        rho_imp = implied_average_correlation(portfolio_variance(w, cov), w, s)
        rho_w = weighted_average_correlation(w, s, corr)
        assert np.isclose(rho_imp, rho_w, rtol=1e-11, atol=1e-13), (
            seed, rho_imp, rho_w
        )
    print("PASS inversion reproduces weighted average correlation exactly")


def test_decompose_end_to_end_consistency():
    """decompose() with var_index=None must be internally exact, and the index
    variance must equal the sample variance of the actual portfolio return
    series built with the same weights and same ddof."""
    r = _random_returns(t=63, n=12, seed=7)
    w = np.full(12, 1 / 12)
    d = decompose(r, w, annualization=ANN)

    port = r @ w
    var_port = np.var(port, ddof=1) * ANN

    assert np.isclose(d.var_index, var_port, rtol=1e-11), (d.var_index, var_port)
    assert np.isclose(d.rho_implied, d.rho_weighted_actual, rtol=1e-11)
    assert np.isclose(d.own_share + d.cross_share, 1.0, rtol=1e-11)
    assert np.isclose(d.residual, 0.0, atol=1e-15)
    print("PASS decompose() internally exact; shares sum to 1; residual == 0")


def test_equicorrelated_recovery():
    """With a truly equicorrelated population, the weighted, simple and implied
    correlations must all converge to the true rho."""
    n, rho, vol = 15, 0.42, 0.25 / np.sqrt(ANN)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    cov = corr * vol ** 2
    chol = np.linalg.cholesky(cov)
    rng = np.random.default_rng(99)
    r = rng.normal(size=(200_000, n)) @ chol.T

    w = np.full(n, 1 / n)
    d = decompose(r, w, annualization=ANN)
    assert abs(d.rho_implied - rho) < 0.01, d.rho_implied
    assert abs(d.rho_simple_actual - rho) < 0.01, d.rho_simple_actual
    # For equal weights and homogeneous vols the weighted and simple averages
    # coincide in the limit.
    assert abs(d.rho_weighted_actual - d.rho_simple_actual) < 0.01
    print(f"PASS equicorrelated recovery: rho_hat={d.rho_implied:.4f} vs true {rho}")


def test_absorption_matches_equicorrelation_formula():
    for n, rho in [(5, 0.3), (20, 0.6), (50, 0.15)]:
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        assert np.isclose(
            absorption_ratio(corr), equicorrelation_absorption(rho, n), rtol=1e-12
        )
    print("PASS absorption ratio matches (1+(n-1)rho)/n under equicorrelation")


def test_weighted_vs_simple_correlation_differ_when_heterogeneous():
    """Guard against someone 'simplifying' the weighted average into the simple
    one: they are genuinely different objects."""
    r = _random_returns(t=2000, n=10, seed=3)
    w = RNG.uniform(0.01, 1.0, 10)
    w /= w.sum()
    cov = np.cov(r, rowvar=False, ddof=1) * ANN
    s = np.sqrt(np.diag(cov))
    corr = cov / np.outer(s, s)
    rw = weighted_average_correlation(w, s, corr)
    rs = simple_average_correlation(corr)
    assert not np.isclose(rw, rs, atol=1e-6)
    print(f"PASS weighted ({rw:.4f}) != simple ({rs:.4f}) under heterogeneity")


def test_one_over_n_limit():
    """The headline portfolio-theory claim, numerically."""
    sig, rho = 0.30, 0.25
    prev_share = 1.0
    for n in (1, 2, 5, 10, 50, 500, 5000):
        t = equal_weight_terms(sig, rho, n)
        assert np.isclose(t["own"], sig ** 2 / n)
        assert t["own_share"] <= prev_share + 1e-12
        prev_share = t["own_share"]
    big = equal_weight_terms(sig, rho, 10 ** 7)
    assert np.isclose(big["total_vol"], np.sqrt(rho) * sig, rtol=1e-6)
    assert big["own_share"] < 1e-6
    print(f"PASS 1/N limit: vol -> sqrt(rho)*sigma = {np.sqrt(rho)*sig:.4f}")


def test_scale_free_factorisation():
    """sigma_index = A * sqrt(C) with C = rho + (1-rho)h must hold EXACTLY,
    including when the index variance is supplied externally -- that is what
    makes the log attribution exactly additive with no interaction term."""
    for seed in range(5):
        r = _random_returns(t=63, n=11, seed=seed + 40)
        w = RNG.uniform(0.02, 1.0, 11)
        w /= w.sum()

        # (a) internally exact case
        d = decompose(r, w, annualization=ANN)
        assert np.isclose(d.comovement_C,
                          d.rho_implied + (1 - d.rho_implied) * d.h_ratio,
                          rtol=1e-11), seed
        assert np.isclose(d.vol_index,
                          d.avg_constituent_vol * np.sqrt(d.comovement_C),
                          rtol=1e-11), seed

        # (b) externally supplied index variance (the index-level case)
        ext = float(np.var(r @ w, ddof=1) * ANN) * 1.35  # deliberate mismatch
        d2 = decompose(r, w, var_index=ext, annualization=ANN)
        assert np.isclose(d2.comovement_C,
                          d2.rho_implied + (1 - d2.rho_implied) * d2.h_ratio,
                          rtol=1e-11), seed
        assert np.isclose(d2.vol_index,
                          d2.avg_constituent_vol * np.sqrt(d2.comovement_C),
                          rtol=1e-11), seed

    # And the log decomposition is additive with zero residual.
    ra = _random_returns(t=63, n=11, seed=1)
    rb = _random_returns(t=63, n=11, seed=2)
    w = np.full(11, 1 / 11)
    da, db = decompose(ra, w, annualization=ANN), decompose(rb, w, annualization=ANN)
    d_log_vol = np.log(db.vol_index) - np.log(da.vol_index)
    d_log_A = np.log(db.avg_constituent_vol) - np.log(da.avg_constituent_vol)
    d_comov = 0.5 * (np.log(db.comovement_C) - np.log(da.comovement_C))
    assert np.isclose(d_log_A + d_comov, d_log_vol, rtol=1e-11, atol=1e-14)
    print("PASS scale-free factorisation sigma = A*sqrt(C); log split exactly additive")


def test_comovement_channel_is_bounded():
    """The headline algebraic point behind the 15% threshold: because C <= 1,
    the co-movement channel cannot exceed 0.5*log(1/C0) no matter how violent
    the episode, whereas the A channel is unbounded."""
    for c0 in (0.2, 0.4, 0.6):
        ceiling = 0.5 * np.log(1.0 / c0)
        # Even a move to perfect co-movement (C=1) cannot beat the ceiling.
        assert np.isclose(0.5 * (np.log(1.0) - np.log(c0)), ceiling)
        assert ceiling < 0.81
    print("PASS co-movement channel is bounded above by 0.5*log(1/C0)")


def test_weights_are_normalised():
    r = _random_returns(t=100, n=6, seed=11)
    w = np.full(6, 1.0)  # deliberately not summing to 1
    d = decompose(r, w, annualization=ANN)
    d2 = decompose(r, np.full(6, 1 / 6), annualization=ANN)
    assert np.isclose(d.var_index, d2.var_index, rtol=1e-12)
    print("PASS weights normalised internally")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nAll {len(fns)} decomposition tests passed.")
