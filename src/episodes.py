r"""
Identification and classification of volatility episodes.

The research question -- is a given vol spike systematic or idiosyncratic? --
demands a classification rule that is mechanical rather than narrative. It is
easy to look at March 2020 and pronounce it "correlation-driven"; the point of
this module is to have the data say so without being told.

TWO LEVELS OF CLASSIFICATION, because they answer different questions:

  * `classify_days` labels every day. This is what feeds the contingency table
    against the VIX term structure, where we need many observations.

  * `identify_episodes` finds discrete, contiguous stress events and attributes
    each one's variance increase to correlation vs volatility using the exact
    Shapley decomposition. This is what gets named, dated and annotated on the
    charts.

CAUSALITY. All thresholds are EXPANDING-WINDOW quantiles, i.e. the percentile
of today's value within history up to today only. Using full-sample quantiles
would leak information: whether March 2020 counts as "top decile vol" would
depend on what happened in 2022. The cost is that the earliest years are ranked
against a short history; the benefit is that the classification is one a
researcher could actually have made in real time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------
# Causal percentile ranks
# --------------------------------------------------------------------------
def expanding_pct_rank(s: pd.Series, min_periods: int = 252) -> pd.Series:
    """Percentile of each value within the history up to and including itself.

    Implemented as a rolling rank over an expanding window. O(n^2) in principle
    but n is ~6500 here, which is trivially fast, and the explicit form is far
    easier to verify as causal than a clever incremental one.
    """
    clean = s.dropna()
    v = clean.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(min_periods - 1, len(v)):
        out[i] = np.count_nonzero(v[: i + 1] <= v[i]) / (i + 1)
    return (
        pd.Series(out, index=clean.index)
        .reindex(s.index)
        .rename(f"{s.name}_pct")
    )


# --------------------------------------------------------------------------
# Day-level regime labels
# --------------------------------------------------------------------------
def classify_days(
    df: pd.DataFrame,
    vol_col: str = "vol_index",
    rho_col: str = "rho_implied",
    vol_threshold: float = 0.70,
    rho_split: float = 0.50,
    min_periods: int = 252,
) -> pd.DataFrame:
    """Label each day as correlation-driven, dispersion-driven, or calm.

    The rule: a day is an elevated-volatility day if index vol is in the top
    30% of its own history to date. Among those days, it is correlation-driven
    if realised average correlation is in the upper half of ITS history to
    date, and dispersion-driven otherwise.

    Why condition on elevated vol rather than classifying all days? Because the
    research question is about what drives vol *spikes*. On a quiet day the
    decomposition is still well defined but the question is not interesting,
    and including hundreds of calm days would swamp the contingency table.

    Why split rho at the median rather than at a fixed level like 0.3? Because
    the unconditional level of realised correlation has drifted upward over the
    sample (indexation, ETF growth), so a fixed cut would mechanically label
    the 2000s dispersion-driven and the 2020s correlation-driven.
    """
    out = df.copy()
    out["vol_pct"] = expanding_pct_rank(out[vol_col], min_periods)
    out["rho_pct"] = expanding_pct_rank(out[rho_col], min_periods)
    out["disp_pct"] = expanding_pct_rank(out["dispersion_ratio"], min_periods)

    elevated = out["vol_pct"] >= vol_threshold
    hi_rho = out["rho_pct"] >= rho_split

    label = pd.Series("calm", index=out.index, dtype=object)
    label[elevated & hi_rho] = "correlation-driven"
    label[elevated & ~hi_rho] = "dispersion-driven"
    label[out["vol_pct"].isna() | out["rho_pct"].isna()] = np.nan
    out["episode_type"] = label
    return out


# --------------------------------------------------------------------------
# Episode-level identification
# --------------------------------------------------------------------------
def _shapley_attribution(row0: pd.Series, row1: pd.Series) -> dict:
    """Exact symmetric attribution of the variance LEVEL change between dates.

    V = O + rho * B, so

        dV_rho = 0.5 (B0 + B1)(rho1 - rho0)
        dV_vol = 0.5 (rho0 + rho1)(B1 - B0) + (O1 - O0)

    and the two sum to V1 - V0 identically.

    KEPT, BUT DEMOTED TO A DIAGNOSTIC. This attribution is arithmetically
    correct and analytically misleading, and understanding why is central to
    the study's argument. B scales with the square of single-name volatility
    and is unbounded; rho is confined to [0, 1]. In a crisis A might rise
    six-fold while rho rises from 0.55 to 0.90 -- so the volatility term must
    dominate the variance level no matter how dramatic the correlation move
    was. Run as the primary classifier it labels Lehman and March 2020
    "dispersion-driven", which is plainly wrong. We report it for
    completeness and classify on the scale-free log decomposition instead.
    """
    o0, b0, r0, v0 = row0["own_term"], row0["cross_term_base"], row0["rho_implied"], row0["var_index"]
    o1, b1, r1, v1 = row1["own_term"], row1["cross_term_base"], row1["rho_implied"], row1["var_index"]
    d_rho = 0.5 * (b0 + b1) * (r1 - r0)
    d_vol = 0.5 * (r0 + r1) * (b1 - b0) + (o1 - o0)
    d_tot = v1 - v0
    return {
        "d_var": d_tot,
        "d_var_from_corr": d_rho,
        "d_var_from_vol": d_vol,
        "resid": d_rho + d_vol - d_tot,
        "var_level_corr_share": d_rho / d_tot if abs(d_tot) > 1e-12 else np.nan,
    }


def _log_channel_attribution(row0: pd.Series, row1: pd.Series) -> dict:
    r"""Scale-free attribution of a volatility move. THIS IS THE PRIMARY ONE.

    From the exact factorisation sigma_index = A * sqrt(C), where
    A = sum_i w_i sigma_i and C = rho + (1 - rho) h,

        d log sigma_index  =  d log A  +  0.5 * d log C
                              \_______/     \__________/
                             dispersion      co-movement
                              channel          channel

    The split is exact and additive with no interaction term, and both
    channels are dimensionless, so a 40% vol episode and a 15% vol episode are
    directly comparable. `comovement_share` is the fraction of the log-vol move
    delivered by the co-movement channel.

    Also computes two counterfactuals, which is how the result is best
    communicated to a non-technical reader:

      vol_if_corr_frozen -- what the volatility peak would have been had
          single-name vols moved exactly as they did but correlation stayed at
          its pre-episode level;
      vol_if_vols_frozen -- the mirror image.

    The gap between the actual peak and `vol_if_corr_frozen` is the part of the
    spike that correlation, and only correlation, is responsible for.
    """
    a0, a1 = row0["avg_constituent_vol"], row1["avg_constituent_vol"]
    c0, c1 = row0["comovement_C"], row1["comovement_C"]
    v0, v1 = row0["vol_index"], row1["vol_index"]
    r0, r1 = row0["rho_implied"], row1["rho_implied"]
    h0, h1 = row0["h_ratio"], row1["h_ratio"]

    if not all(np.isfinite([a0, a1, c0, c1, v0, v1])) or min(a0, a1, c0, c1, v0, v1) <= 0:
        return {k: np.nan for k in (
            "d_log_vol", "d_log_A", "d_comovement", "log_resid",
            "comovement_share", "vol_if_corr_frozen", "vol_if_vols_frozen",
            "corr_uplift_pct")}

    d_log_vol = np.log(v1) - np.log(v0)
    d_log_a = np.log(a1) - np.log(a0)
    d_comov = 0.5 * (np.log(c1) - np.log(c0))

    vol_cf_corr = a1 * np.sqrt(max(r0 + (1 - r0) * h1, 1e-12))
    vol_cf_vols = a0 * np.sqrt(max(r1 + (1 - r1) * h0, 1e-12))

    # How much of the volatility peak is attributable to correlation alone.
    uplift = (v1 / vol_cf_corr - 1.0) if vol_cf_corr > 0 else np.nan

    # ...and how far that is towards the maximum the algebra allows. With
    # single-name vols fixed at their realised peak values, index vol is
    # A1 * sqrt(C), and C cannot exceed 1. So the largest uplift correlation
    # could possibly have delivered from this starting point is
    #       ceiling = A1 / vol_if_corr_frozen - 1 = sqrt(1/C0') - 1.
    # Expressing the realised uplift as a fraction of its own ceiling gives a
    # measure on [0, 1] that is free of BOTH the volatility scale and the
    # starting correlation level -- so a 30%-vol episode starting from rho=0.7
    # is directly comparable with a 75%-vol episode starting from rho=0.37.
    # This is the quantity the episode classification uses.
    c0_adj = max(r0 + (1 - r0) * h1, 1e-12)
    ceiling = 1.0 / np.sqrt(c0_adj) - 1.0
    frac = uplift / ceiling if ceiling > 1e-9 else np.nan

    return {
        "d_log_vol": d_log_vol,
        "d_log_A": d_log_a,
        "d_comovement": d_comov,
        "log_resid": d_log_a + d_comov - d_log_vol,
        "comovement_share": d_comov / d_log_vol if abs(d_log_vol) > 1e-9 else np.nan,
        "vol_if_corr_frozen": vol_cf_corr,
        "vol_if_vols_frozen": vol_cf_vols,
        "corr_uplift_pct": uplift * 100,
        "corr_uplift_ceiling_pct": ceiling * 100,
        "corr_uplift_frac_of_max": frac,
    }


def identify_episodes(
    df: pd.DataFrame,
    vol_quantile: float = 0.90,
    min_periods: int = 252,
    max_gap: int = 10,
    min_length: int = 5,
    baseline_lookback: int = 21,
    comov_threshold: float = 0.15,
    rho_pct_hi: float = 0.70,
) -> pd.DataFrame:
    """Find contiguous high-volatility episodes and attribute each one.

    `max_gap` lets an episode survive a few days dipping below the threshold --
    real crises are not monotone, and splitting March 2020 into four separate
    "episodes" because vol wobbled would be an artefact of the threshold.

    `baseline_lookback` sets how far before the episode start we measure the
    pre-stress state. 21 trading days is one month: long enough to be outside
    the run-up, short enough that the comparison is to the immediately
    preceding regime rather than to some distant average.
    """
    d = df.copy()
    d["vol_pct"] = expanding_pct_rank(d["vol_index"], min_periods)
    d["rho_pct"] = expanding_pct_rank(d["rho_implied"], min_periods)
    flag = (d["vol_pct"] >= vol_quantile).fillna(False).to_numpy()

    # Group flagged days into episodes, bridging gaps of <= max_gap.
    episodes = []
    i = 0
    n = len(d)
    while i < n:
        if not flag[i]:
            i += 1
            continue
        start = i
        end = i
        j = i + 1
        gap = 0
        while j < n:
            if flag[j]:
                end = j
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            j += 1
        if end - start + 1 >= min_length:
            episodes.append((start, end))
        i = j

    rows = []
    for start, end in episodes:
        seg = d.iloc[start : end + 1]
        peak_pos = int(np.nanargmax(seg["vol_index"].to_numpy())) + start
        base_pos = max(0, start - baseline_lookback)

        row0 = d.iloc[base_pos]
        row1 = d.iloc[peak_pos]
        attrib = _shapley_attribution(row0, row1)
        logattr = _log_channel_attribution(row0, row1)

        rows.append(
            {
                "start": d.index[start],
                "peak": d.index[peak_pos],
                "end": d.index[end],
                "days": end - start + 1,
                "baseline_date": d.index[base_pos],
                "vol_baseline": row0["vol_index"],
                "vol_peak": row1["vol_index"],
                "avg_name_vol_baseline": row0["avg_constituent_vol"],
                "avg_name_vol_peak": row1["avg_constituent_vol"],
                "rho_baseline": row0["rho_implied"],
                "rho_peak": row1["rho_implied"],
                "rho_pct_peak": row1.get("rho_pct", np.nan),
                "d_rho": row1["rho_implied"] - row0["rho_implied"],
                "disp_baseline": row0["dispersion_ratio"],
                "disp_peak": row1["dispersion_ratio"],
                "absorption_baseline": row0["absorption"],
                "absorption_peak": row1["absorption"],
                **attrib,
                **logattr,
            }
        )

    ep = pd.DataFrame(rows)
    if ep.empty:
        return ep

    # ------------------------------------------------------------------
    # Classification.
    #
    # We classify on `corr_uplift_frac_of_max`: how far correlation travelled
    # towards perfect co-movement, as a fraction of the distance it could
    # possibly have travelled from where it started.
    #
    # Why not on `comovement_share` (the co-movement channel's share of the
    # log-vol move)? Because that statistic still carries the volatility scale
    # in its DENOMINATOR, and so penalises exactly the largest events. March
    # 2020 is the clean illustration: correlation rose from 0.57 to 0.87 and
    # added 19% to the volatility peak, but because single-name vol rose more
    # than six-fold at the same time, the co-movement share is only 9% and the
    # episode looks unremarkable. Normalising by the algebraic ceiling instead
    # of by the size of the vol move removes that artefact, and it is the
    # reason the two thresholds below are 0.50 and 0.20 rather than something
    # that has to be apologised for.
    #
    # The `rho_pct_hi` condition remains as a guard: an episode is only called
    # correlation-driven if realised correlation actually reached a high level
    # by its own historical standards, not merely if it moved.
    #
    # Both the share-based and variance-level statistics are still reported in
    # the output table so a reader can see all three and disagree with us.
    # ------------------------------------------------------------------
    def _label(row):
        f, rp = row["corr_uplift_frac_of_max"], row["rho_pct_peak"]
        if not np.isfinite(f):
            return "undetermined"
        if f >= 0.50 and (not np.isfinite(rp) or rp >= rho_pct_hi):
            return "correlation-driven"
        if f <= 0.20:
            return "dispersion-driven"
        return "mixed"

    ep["classification"] = ep.apply(_label, axis=1)
    ep["event"] = ep.apply(lambda r: _match_event(r["start"], r["end"]), axis=1)
    return ep.sort_values("peak").reset_index(drop=True)


def _match_event(start, end, tolerance_days: int = 45) -> str:
    """Attach a human-readable name from the event calendar if one falls in or
    near the episode window. Names are labels for the reader, not inputs to the
    classification -- the attribution is computed before any name is attached.
    """
    names = []
    for date_str, name in config.EVENTS:
        d = pd.Timestamp(date_str)
        if start - pd.Timedelta(days=tolerance_days) <= d <= end + pd.Timedelta(days=tolerance_days):
            names.append(name)
    return "; ".join(names) if names else ""


def dispersion_episodes(
    df: pd.DataFrame,
    quantile: float = 0.90,
    min_periods: int = 252,
    max_gap: int = 10,
    min_length: int = 10,
) -> pd.DataFrame:
    """Periods of unusually HIGH dispersion, identified independently of vol.

    This exists because the two phenomena are not mirror images. A
    correlation-driven crisis is a high-vol event; a dispersion regime is often
    a perfectly calm-looking market in which single names are moving a lot but
    cancelling out at the index level. Searching only within vol spikes would
    systematically miss them -- which is precisely the asymmetry the research
    question is about.
    """
    d = df.copy()
    d["disp_pct"] = expanding_pct_rank(d["dispersion_ratio"], min_periods)
    flag = (d["disp_pct"] >= quantile).fillna(False).to_numpy()

    out, i, n = [], 0, len(d)
    while i < n:
        if not flag[i]:
            i += 1
            continue
        start, end, j, gap = i, i, i + 1, 0
        while j < n:
            if flag[j]:
                end, gap = j, 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            j += 1
        if end - start + 1 >= min_length:
            seg = d.iloc[start : end + 1]
            out.append(
                {
                    "start": d.index[start],
                    "end": d.index[end],
                    "days": end - start + 1,
                    "mean_dispersion": seg["dispersion_ratio"].mean(),
                    "mean_rho": seg["rho_implied"].mean(),
                    "mean_vol": seg["vol_index"].mean(),
                    "mean_avg_name_vol": seg["avg_constituent_vol"].mean(),
                    "event": _match_event(d.index[start], d.index[end]),
                }
            )
        i = j
    return pd.DataFrame(out)
