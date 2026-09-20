"""
End-to-end analysis runner.

Prints a full diagnostic log as it goes -- coverage, fit quality, causality
checks, out-of-range counts -- because a study of this kind is only as
trustworthy as the checks its author actually looked at.

Usage:  python run_analysis.py [--force-download]
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd

import config
from src import data as dta
from src import episodes as ep
from src import plotting as plot
from src import rolling as roll
from src import stats_tests as st
from src import termstructure as ts

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

FORCE = "--force-download" in sys.argv
RESULTS: dict = {}


def hdr(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_table(df: pd.DataFrame, name: str, index: bool = True):
    path = config.TAB_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"    table  -> {path.name}  ({len(df)} rows)")


# ==========================================================================
# 1. DATA
# ==========================================================================
hdr("1. DATA ACQUISITION")

etf_tickers = [config.INDEX_PROXY] + list(config.SECTOR_ETFS)
etf_px, etf_dropped = dta.fetch_prices(etf_tickers, cache_key="etfs", force=FORCE)
print(f"  ETFs downloaded: {etf_px.shape[1]} / {len(etf_tickers)}")
if etf_dropped:
    print(f"  !! dropped: {etf_dropped}")

all_names = sorted({t for v in config.SECTOR_CONSTITUENTS.values() for t in v})
name_px, name_dropped = dta.fetch_prices(all_names, cache_key="constituents", force=FORCE)
print(f"  Constituents downloaded: {name_px.shape[1]} / {len(all_names)}")
if name_dropped:
    print(f"  !! dropped (delisted / unresolved): {name_dropped}")

cov = dta.coverage_report(etf_px)
print("\n  ETF coverage:")
print(cov.to_string(index=False))

term, term_diag = dta.load_term_structure(force=FORCE)
print("\n  VIX term structure:")
for k, v in term_diag.items():
    print(f"    {k}: {v}")
print(f"    panel: {term.shape[0]} rows, "
      f"{term.index[0].date()} -> {term.index[-1].date()}")
print(f"    non-null per column:\n{term.notna().sum().to_string()}")
RESULTS["term_structure_diagnostics"] = term_diag
RESULTS["dropped_constituents"] = name_dropped

# Returns. SIMPLE returns throughout -- see src/data.simple_returns for why.
etf_ret = dta.simple_returns(etf_px).loc[config.START : config.END]
name_ret = dta.simple_returns(name_px).loc[config.START : config.END]
spy_ret = etf_ret[config.INDEX_PROXY]
sector_etf_ret = etf_ret[[c for c in config.SECTOR_ETFS if c in etf_ret.columns]]

print(f"\n  Return sample: {etf_ret.index[0].date()} -> {etf_ret.index[-1].date()}"
      f"  ({len(etf_ret)} days)")
RESULTS["sample"] = {
    "start": str(etf_ret.index[0].date()),
    "end": str(etf_ret.index[-1].date()),
    "n_days": int(len(etf_ret)),
    "n_constituents": int(name_px.shape[1]),
}

# ==========================================================================
# 2. INDEX-LEVEL WEIGHTS
# ==========================================================================
hdr("2. INDEX WEIGHTS (returns-based style analysis)")

from src.weights import expand_weights, rolling_index_weights  # noqa: E402

W, r2 = rolling_index_weights(
    spy_ret, sector_etf_ret, window=config.WEIGHT_WINDOW, step=config.WEIGHT_STEP
)
print(f"  estimation dates: {len(W)}  ({W.index[0].date()} -> {W.index[-1].date()})")
print(f"  replication R^2: median={r2.median():.5f}  min={r2.min():.5f}  "
      f"p05={r2.quantile(0.05):.5f}")

latest = W.iloc[-1].sort_values(ascending=False)
print("\n  Latest estimated sector weights (sanity check against known S&P weights):")
for k, v in latest.items():
    if v > 0.001:
        print(f"    {k:5s} {config.SECTOR_ETFS[k]:24s} {v*100:6.2f}%")

# Sanity check on a publicly known fact: technology is by far the largest
# S&P 500 sector in the mid-2020s at roughly 30-35%.
tech_now = float(W.iloc[-1].get("XLK", np.nan))
print(f"\n  XLK weight now = {tech_now*100:.1f}%  "
      f"(external sanity check: S&P tech weight is ~30-35% in this era)")
RESULTS["weights"] = {
    "median_r2": float(r2.median()),
    "min_r2": float(r2.min()),
    "latest": {k: float(v) for k, v in latest.items() if v > 0.001},
}

Wd = expand_weights(W, sector_etf_ret.index)
save_table(W, "index_sector_weights")

# ==========================================================================
# 3. INDEX-LEVEL ROLLING DECOMPOSITION
# ==========================================================================
hdr("3. INDEX-LEVEL ROLLING DECOMPOSITION")

idx_dec = roll.rolling_decomposition(
    sector_etf_ret, weights=Wd, index_returns=spy_ret,
    window=config.WINDOW, annualization=config.ANNUALIZATION,
    min_names=5, label="index",
)
print(f"  rows: {len(idx_dec)}  {idx_dec.index[0].date()} -> {idx_dec.index[-1].date()}")

print("\n  rho_implied diagnostics (external SPY variance -> identity is approximate):")
print(roll.flag_out_of_range(idx_dec).to_string(index=False))

resid_rel = (idx_dec["residual"] / idx_dec["var_index"]).abs()
print(f"\n  |residual| / var_index:  median={resid_rel.median():.4f}  "
      f"p95={resid_rel.quantile(0.95):.4f}  max={resid_rel.max():.4f}")
print("    (this is the part of SPY variance the 11 sector ETFs + estimated")
print("     weights fail to reproduce -- tracking error, not a modelling choice)")

# Exact variant: index = the replicating sector portfolio itself.
idx_exact = roll.rolling_decomposition(
    sector_etf_ret, weights=Wd, index_returns=None,
    window=config.WINDOW, annualization=config.ANNUALIZATION,
    min_names=5, label="index_exact",
)
gap = (idx_dec["rho_implied"] - idx_exact["rho_implied"]).abs()
print(f"\n  |rho(external SPY var) - rho(exact replicating portfolio)|: "
      f"median={gap.median():.4f}  p95={gap.quantile(0.95):.4f}")

print("\n  causality check (recompute on truncated samples)...")
roll.assert_causal(sector_etf_ret, window=config.WINDOW, min_names=5, n_checks=3)
print("    PASS - no look-ahead detected")

attrib = roll.attribute_variance_change(idx_dec, horizon=config.ATTRIB_HORIZON)
print(f"  Shapley attribution residual (max |check|): "
      f"{roll.attrib_check(attrib):.3e}  (must be ~0)")

print("\n  Summary statistics, index level:")
summ = idx_dec[["vol_index", "rho_implied", "own_share", "dispersion_ratio",
                "absorption", "n_names"]].describe().T
print(summ.to_string())
save_table(idx_dec, "index_decomposition")
save_table(summ, "index_summary_stats")

RESULTS["index"] = {
    "rho_mean": float(idx_dec["rho_implied"].mean()),
    "rho_median": float(idx_dec["rho_implied"].median()),
    "rho_min": float(idx_dec["rho_implied"].min()),
    "rho_max": float(idx_dec["rho_implied"].max()),
    "own_share_mean": float(idx_dec["own_share"].mean()),
    "own_share_median": float(idx_dec["own_share"].median()),
    "own_share_max": float(idx_dec["own_share"].max()),
    "vol_mean": float(idx_dec["vol_index"].mean()),
    "residual_rel_median": float(resid_rel.median()),
    "rho_vs_exact_median_gap": float(gap.median()),
    "rho_absorption_corr": float(idx_dec["rho_implied"].corr(idx_dec["absorption"])),
}

# ==========================================================================
# 4. EPISODES
# ==========================================================================
hdr("4. EPISODE IDENTIFICATION AND CLASSIFICATION")

idx_lab = ep.classify_days(idx_dec)
print("  day-level labels:")
print(idx_lab["episode_type"].value_counts(dropna=False).to_string())

episodes = ep.identify_episodes(idx_dec, vol_quantile=config.VOL_SPIKE_QUANTILE)
print(f"\n  {len(episodes)} volatility episodes identified "
      f"(index vol in top {int((1-config.VOL_SPIKE_QUANTILE)*100)}% of history to date)")
if not episodes.empty:
    show = episodes[["peak", "days", "vol_baseline", "vol_peak",
                     "rho_baseline", "rho_peak", "rho_pct_peak",
                     "comovement_share", "corr_uplift_frac_of_max",
                     "classification", "event"]].copy()
    show["peak"] = show["peak"].dt.date
    for c in ("vol_baseline", "vol_peak", "rho_baseline", "rho_peak",
              "rho_pct_peak", "comovement_share", "corr_uplift_frac_of_max"):
        show[c] = show[c].round(3)
    print(show.to_string(index=False))

    print("\n  Counterfactual: what the volatility peak would have been if")
    print("  correlation had stayed at its pre-episode level (vols as realised):")
    cf = episodes[["peak", "event", "vol_peak", "vol_if_corr_frozen",
                   "vol_if_vols_frozen", "corr_uplift_pct",
                   "corr_uplift_ceiling_pct", "corr_uplift_frac_of_max"]].copy()
    cf["peak"] = cf["peak"].dt.date
    for c in ("vol_peak", "vol_if_corr_frozen", "vol_if_vols_frozen"):
        cf[c] = (cf[c] * 100).round(1)
    for c in ("corr_uplift_pct", "corr_uplift_ceiling_pct"):
        cf[c] = cf[c].round(1)
    cf["corr_uplift_frac_of_max"] = cf["corr_uplift_frac_of_max"].round(3)
    cf.columns = ["peak", "event", "actual_vol%", "vol_if_corr_frozen%",
                  "vol_if_vols_frozen%", "corr_uplift%", "max_possible_uplift%",
                  "frac_of_max"]
    print(cf.to_string(index=False))

    print("\n  classification counts:")
    print(episodes["classification"].value_counts().to_string())
    print(f"  max |variance Shapley residual|: {episodes['resid'].abs().max():.3e}")
    print(f"  max |log-channel residual|:      {episodes['log_resid'].abs().max():.3e}")
    save_table(episodes, "volatility_episodes", index=False)

disp_eps = ep.dispersion_episodes(idx_dec)
print(f"\n  {len(disp_eps)} high-dispersion episodes (independent of vol level)")
if not disp_eps.empty:
    d2 = disp_eps.copy()
    for c in ("start", "end"):
        d2[c] = d2[c].dt.date
    for c in ("mean_dispersion", "mean_rho", "mean_vol", "mean_avg_name_vol"):
        d2[c] = d2[c].round(3)
    print(d2.to_string(index=False))
    save_table(disp_eps, "dispersion_episodes", index=False)

RESULTS["episodes"] = {
    "n_vol_episodes": int(len(episodes)),
    "classification_counts": (episodes["classification"].value_counts().to_dict()
                              if not episodes.empty else {}),
    "n_dispersion_episodes": int(len(disp_eps)),
    "day_labels": idx_lab["episode_type"].value_counts(dropna=False).to_dict(),
}

# ==========================================================================
# 5. SECTOR-LEVEL DECOMPOSITION
# ==========================================================================
hdr("5. SECTOR-LEVEL DECOMPOSITION (single-name constituents, equal weight)")

sector_results: dict[str, pd.DataFrame] = {}
sector_rows = []
for sector, tickers in config.SECTOR_CONSTITUENTS.items():
    have = [t for t in tickers if t in name_ret.columns]
    r = name_ret[have]
    dec = roll.rolling_decomposition(
        r, weights=None, index_returns=None,
        window=config.WINDOW, annualization=config.ANNUALIZATION,
        min_names=config.MIN_NAMES, label=sector,
    )
    if dec.empty:
        print(f"  {sector}: SKIPPED (no admissible windows)")
        continue
    sector_results[sector] = dec

    # The identity must be exact here -- equal weights, portfolio built from
    # the same names. This is the strongest internal check in the study.
    ident = (dec["rho_implied"] - dec["rho_weighted_actual"]).abs().max()
    shares = (dec["own_share"] + dec["cross_share"] - 1.0).abs().max()

    print(f"  {sector:18s} n={len(dec):5d}  names {dec['n_names'].min()}-{dec['n_names'].max()}"
          f"  rho mean={dec['rho_implied'].mean():.3f}"
          f"  own_share mean={dec['own_share'].mean()*100:5.2f}%"
          f"  |identity err|={ident:.2e}  |shares-1|={shares:.2e}")

    sector_rows.append({
        "sector": sector,
        "n_obs": len(dec),
        "n_names_min": int(dec["n_names"].min()),
        "n_names_max": int(dec["n_names"].max()),
        "rho_mean": dec["rho_implied"].mean(),
        "rho_median": dec["rho_implied"].median(),
        "rho_std": dec["rho_implied"].std(),
        "rho_min": dec["rho_implied"].min(),
        "rho_max": dec["rho_implied"].max(),
        "own_share_mean": dec["own_share"].mean(),
        "dispersion_mean": dec["dispersion_ratio"].mean(),
        "avg_name_vol_mean": dec["avg_constituent_vol"].mean(),
        "portfolio_vol_mean": dec["vol_index"].mean(),
        "absorption_mean": dec["absorption"].mean(),
        "identity_max_err": ident,
    })
    save_table(dec, f"sector_decomposition_{sector.replace(' ', '_')}")

sector_summary = pd.DataFrame(sector_rows).sort_values("rho_mean", ascending=False)
print("\n  Sector summary (sorted by mean realised internal correlation):")
print(sector_summary.round(4).to_string(index=False))
save_table(sector_summary, "sector_summary", index=False)
RESULTS["sectors"] = sector_summary.round(5).to_dict(orient="records")

# Heatmap panel: monthly percentile of each sector's own rho history.
panel_rows = {}
for sector, dec in sector_results.items():
    pct = ep.expanding_pct_rank(dec["rho_implied"], min_periods=252)
    panel_rows[sector] = pct.resample("ME").mean()
panel = pd.DataFrame(panel_rows).T.dropna(axis=1, how="all")
panel = panel.loc[sector_summary["sector"]]
save_table(panel, "sector_regime_panel")

# ==========================================================================
# 6. FISHER z SIGNIFICANCE TESTS
# ==========================================================================
hdr("6. FISHER z-TRANSFORM SIGNIFICANCE TESTS")

COMPARISONS = [
    (("2000-03-01", "2001-03-30"), ("1999-01-04", "2000-02-29"),
     "Dot-com unwind 2000-01", "Late bull 1999"),
    (("2008-09-01", "2009-03-31"), ("2007-01-03", "2008-06-30"),
     "GFC peak 2008-09..2009-03", "Pre-GFC 2007..2008H1"),
    (("2011-08-01", "2011-12-30"), ("2010-11-01", "2011-07-29"),
     "EU crisis / downgrade 2011H2", "Preceding 2010-11..2011-07"),
    (("2020-02-20", "2020-04-30"), ("2019-06-03", "2020-02-19"),
     "COVID crash 2020-02..04", "Pre-COVID 2019-06..2020-02"),
    (("2020-11-02", "2021-06-30"), ("2020-02-20", "2020-04-30"),
     "Vaccine rotation 2020-11..2021-06", "COVID crash 2020-02..04"),
    (("2022-01-03", "2022-10-31"), ("2021-01-04", "2021-12-31"),
     "Hiking cycle 2022", "Calm 2021"),
    (("2023-03-01", "2023-05-31"), ("2022-11-01", "2023-02-28"),
     "SVB 2023-03..05", "Preceding 2022-11..2023-02"),
    (("2024-07-15", "2024-09-13"), ("2024-04-01", "2024-07-12"),
     "Yen carry unwind 2024", "Preceding 2024-04..07"),
    (("2025-04-01", "2025-06-30"), ("2025-01-02", "2025-03-31"),
     "Tariff shock 2025-Q2", "Preceding 2025-Q1"),
]

test_rows = []
print("  Index level (11 sector ETFs, equal weight within test for like-for-like):")
for pa, pb, la, lb in COMPARISONS:
    res = st.compare_periods(
        sector_etf_ret, pa, pb, la, lb,
        block=config.BOOTSTRAP_BLOCK, draws=2000,
        seed=config.RANDOM_SEED, min_names=5,
    )
    if "error" in res:
        print(f"    {la:38s} SKIP ({res['error']})")
        continue
    res["level"] = "index (sector ETFs)"
    test_rows.append(res)
    print(f"    {la:38s} rho {res['rho_b']:.3f} -> {res['rho_a']:.3f}  "
          f"d={res['delta_rho']:+.3f}  naive p={res['naive_p']:.2e}  "
          f"adj p={res['adj_p']:.2e}  boot p={res['boot_p']:.4f}")

print("\n  Sector level (Technology and Energy constituents):")
for sector in ("Technology", "Energy"):
    cols = [t for t in config.SECTOR_CONSTITUENTS[sector] if t in name_ret.columns]
    for pa, pb, la, lb in COMPARISONS:
        res = st.compare_periods(
            name_ret[cols], pa, pb, la, lb,
            block=config.BOOTSTRAP_BLOCK, draws=1000,
            seed=config.RANDOM_SEED, min_names=config.MIN_NAMES,
        )
        if "error" in res:
            continue
        res["level"] = sector
        test_rows.append(res)
        print(f"    {sector:12s} {la:34s} d={res['delta_rho']:+.3f}  "
              f"naive p={res['naive_p']:.2e}  boot p={res['boot_p']:.4f}")

tests = pd.DataFrame(test_rows)
save_table(tests, "significance_tests", index=False)

if not tests.empty:
    idx_tests = tests[tests["level"] == "index (sector ETFs)"]
    n_naive_sig = int((idx_tests["naive_p"] < 0.05).sum())
    n_boot_sig = int((idx_tests["boot_p"] < 0.05).sum())
    n_adj_sig = int((idx_tests["adj_p"] < 0.05).sum())
    print(f"\n  Of {len(idx_tests)} index-level comparisons significant at 5%: "
          f"naive Fisher {n_naive_sig}, effective-n adjusted {n_adj_sig}, "
          f"block bootstrap {n_boot_sig}.")

    # How the textbook interval compares with the honest one, episode by
    # episode. The direction of the error is not constant, which is the point.
    wn, wb = [], []
    for r in idx_tests.itertuples():
        se = np.sqrt(1.0 / (r.n_days_a - 3) + 1.0 / (r.n_days_b - 3))
        dz = st.fisher_z(r.rho_a) - st.fisher_z(r.rho_b)
        lo = st.inv_fisher_z(st.fisher_z(r.rho_b) + dz - 1.959964 * se)
        hi = st.inv_fisher_z(st.fisher_z(r.rho_b) + dz + 1.959964 * se)
        wn.append(float(hi - lo))
        wb.append(float(r.boot_ci_hi - r.boot_ci_lo))
    ratio = np.array(wb) / np.array(wn)
    worst = idx_tests.iloc[int(np.argmax(ratio))]["label_a"]
    print(f"  Bootstrap CI width / naive CI width: median={np.median(ratio):.2f}  "
          f"min={ratio.min():.2f}  max={ratio.max():.2f} ({worst})")
    print("    < 1: the textbook interval is too WIDE -- it prices the sampling error")
    print("         of a SINGLE pair, but we are testing an AVERAGE over many pairs.")
    print("    > 1: it is too NARROW -- volatility clustering means the effective")
    print("         number of independent observations is far below the day count.")
    print("    Both happen in this sample, so no single interval is 'the' right one.")

    med_shrink = float(np.nanmedian(
        idx_tests["adj_n_a"] / idx_tests["n_days_a"]))
    print(f"  Median effective-sample-size shrinkage: n_eff / n = {med_shrink:.2f}")
    RESULTS["significance"] = {
        "n_tests_index": int(len(idx_tests)),
        "n_sig_naive": n_naive_sig,
        "n_sig_adjusted": n_adj_sig,
        "n_sig_bootstrap": n_boot_sig,
        "median_neff_shrinkage": med_shrink,
        "ci_width_ratio_median": float(np.median(ratio)),
        "ci_width_ratio_min": float(ratio.min()),
        "ci_width_ratio_max": float(ratio.max()),
    }

# ==========================================================================
# 7. VIX TERM-STRUCTURE CROSS-CHECK
# ==========================================================================
hdr("7. VIX TERM-STRUCTURE CROSS-REFERENCE")

tsx = ts.build_term_structure(term)
merged = ts.align(idx_lab, tsx)
print(f"  overlapping observations: {len(merged)}  "
      f"{merged.index[0].date()} -> {merged.index[-1].date()}")
print(f"  days backwardated: {int(merged['backwardation'].sum())} "
      f"({merged['backwardation'].mean()*100:.1f}%)")

cont = ts.regime_contingency(merged)
if "error" not in cont:
    print("\n  Contingency: episode type x term-structure state")
    print(cont["table"].to_string())
    print(f"\n    backwardation during correlation-driven days: "
          f"{cont['pct_backwardation_corr']:.1f}%  (n={cont['n_corr_driven']})")
    print(f"    backwardation during dispersion-driven days:  "
          f"{cont['pct_backwardation_disp']:.1f}%  (n={cont['n_disp_driven']})")
    print(f"    odds ratio = {cont['odds_ratio']:.2f}   Fisher exact p = {cont['fisher_p']:.3e}")
    RESULTS["term_structure_check"] = {
        k: v for k, v in cont.items() if k != "table"
    }
    save_table(cont["table"], "termstructure_contingency")

quint = ts.slope_by_rho_quintile(merged)
print("\n  Term-structure slope by correlation quintile:")
print(quint.round(4).to_string())
save_table(quint, "termstructure_by_rho_quintile")

ll = ts.lead_lag_profile(merged)
print(f"\n  Lead-lag: corr(curve slope at t, realised rho at t+k)")
print(f"    contemporaneous (k=0): {ll.attrs['contemporaneous']:+.4f}")
print(f"    strongest at k = {ll.attrs['peak_lag']:+d} days: {ll.attrs['peak_corr']:+.4f}")
print(f"    (half the {config.WINDOW}-day estimation window is {config.WINDOW//2} days --")
print("     a trailing estimator necessarily lags a forward-looking one)")
save_table(ll, "termstructure_lead_lag", index=False)
RESULTS["lead_lag"] = {
    "contemporaneous": ll.attrs["contemporaneous"],
    "peak_lag_days": ll.attrs["peak_lag"],
    "peak_corr": ll.attrs["peak_corr"],
}

partial = ts.partial_check(merged)
print("\n  Is the link circular? (does rho add anything beyond the VIX level?)")
for k, v in partial.items():
    print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
RESULTS["term_structure_partial"] = partial

# Episode-by-episode agreement with the term structure.
if not episodes.empty:
    rows = []
    for r in episodes.itertuples():
        win = tsx.loc[r.start : r.end]
        if win.empty or win["slope_3m_1m"].notna().sum() == 0:
            continue
        rows.append({
            "peak": r.peak.date(),
            "event": r.event,
            "classification": r.classification,
            "frac_of_max": round(r.corr_uplift_frac_of_max, 3),
            "min_slope_3m_1m": round(float(win["slope_3m_1m"].min()), 3),
            "pct_days_backwardated": round(float(win["backwardation"].mean() * 100), 1),
            "peak_vix": round(float(win["VIX"].max()), 1),
            "agrees": None,
        })
    agree = pd.DataFrame(rows)
    if not agree.empty:
        # Agreement criterion: did the VIX curve INVERT at any point during
        # the episode? A binary "did it invert" test is used rather than a
        # threshold on the share of backwardated days because inversion is
        # itself the qualitative regime signal, and picking a percentage
        # threshold would be a free parameter we would then have to defend.
        inverted = agree["min_slope_3m_1m"] < 1.0
        agree["agrees"] = np.where(
            agree["classification"] == "correlation-driven", inverted,
            np.where(agree["classification"] == "dispersion-driven", ~inverted, None),
        )
        print("\n  Episode-level agreement with the term structure:")
        print(agree.to_string(index=False))
        save_table(agree, "episode_termstructure_agreement", index=False)
        checked = agree[agree["classification"].isin(
            ["correlation-driven", "dispersion-driven"])]
        if len(checked):
            rate = float(checked["agrees"].astype(bool).mean() * 100)
            print(f"\n    agreement rate: {rate:.0f}% of {len(checked)} classifiable episodes")
            RESULTS["episode_agreement_rate"] = rate

# ==========================================================================
# 8. ROBUSTNESS ACROSS WINDOWS
# ==========================================================================
hdr("8. ROBUSTNESS: ESTIMATION WINDOW")

by_window = {config.WINDOW: idx_dec}
for w in config.ROBUSTNESS_WINDOWS:
    by_window[w] = roll.rolling_decomposition(
        sector_etf_ret, weights=Wd, index_returns=spy_ret,
        window=w, annualization=config.ANNUALIZATION, min_names=5,
    )
comp = pd.DataFrame({f"rho_{w}": d["rho_implied"] for w, d in by_window.items()})
print("  correlation between rho series estimated at different windows:")
print(comp.corr().round(4).to_string())
print("\n  means:")
print(comp.mean().round(4).to_string())
save_table(comp, "window_robustness")
RESULTS["window_robustness"] = {
    "corr_matrix": comp.corr().round(4).to_dict(),
    "means": comp.mean().round(4).to_dict(),
}

# ==========================================================================
# 9. FIGURES
# ==========================================================================
hdr("9. FIGURES")
plot.set_style()

plot.fig_index_decomposition(idx_dec, config.WINDOW)
if not episodes.empty:
    plot.fig_episode_attribution(episodes)
plot.fig_sector_heatmap(panel)
plot.fig_sector_small_multiples(sector_results)
plot.fig_term_structure(merged, quint)
plot.fig_diversification_limit(
    sigma_bar=float(idx_dec["avg_constituent_vol"].mean()),
    rho=float(idx_dec["rho_implied"].median()),
)
if not tests.empty:
    plot.fig_significance(tests[tests["level"] == "index (sector ETFs)"])
plot.fig_weight_diagnostics(W, r2)
plot.fig_window_robustness(by_window)
plot.fig_absorption_check(idx_dec)
plot.fig_lead_lag(ll, config.WINDOW)

# ==========================================================================
# 10. KEY NUMBERS FOR THE WRITE-UP
# ==========================================================================
hdr("10. KEY NUMBERS")

covid = idx_dec.loc["2020-03-01":"2020-04-15"]
calm21 = idx_dec.loc["2021-01-01":"2021-12-31"]
gfc = idx_dec.loc["2008-09-15":"2009-03-31"]
RESULTS["key_periods"] = {}
for nm, seg in [("COVID 2020-03..04", covid), ("Calm 2021", calm21),
                ("GFC 2008-09..2009-03", gfc)]:
    if seg.empty:
        continue
    d = {
        "vol_mean": float(seg["vol_index"].mean()),
        "rho_mean": float(seg["rho_implied"].mean()),
        "own_share_mean": float(seg["own_share"].mean()),
        "dispersion_mean": float(seg["dispersion_ratio"].mean()),
    }
    RESULTS["key_periods"][nm] = d
    print(f"  {nm:24s} vol={d['vol_mean']*100:5.1f}%  rho={d['rho_mean']:.3f}  "
          f"own_share={d['own_share_mean']*100:5.2f}%  disp={d['dispersion_mean']:.3f}")

with open(config.OUT_DIR / "results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\n  key results -> {(config.OUT_DIR / 'results.json').name}")

hdr("DONE")
