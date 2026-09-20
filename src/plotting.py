"""
Figures.

Design rules followed here (and the reasons, since they are not arbitrary):

  * NO DUAL-AXIS CHARTS. Plotting correlation and volatility against two
    different y-scales on one frame lets the author manufacture any apparent
    lead-lag relationship by choosing the scales. Wherever two quantities must
    be compared through time they are drawn as stacked panels sharing an
    x-axis, so the reader compares positions in time, not slopes against
    incommensurable units.
  * Categorical colours are assigned in a fixed validated order, never cycled.
    The palette was checked for colour-vision-deficiency separation rather than
    chosen by eye.
  * Sequential encoding uses one hue light-to-dark; the regime heatmap uses a
    diverging blue-gray-red ramp because it has a genuine neutral midpoint
    (the median regime), and red is aligned with high correlation so that
    "red = stress" matches the reader's prior.
  * Every figure has a companion CSV in outputs/tables, which is both the
    accessibility fallback and the reproducibility artefact.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm

import config

# --------------------------------------------------------------------------
# Palette (validated: worst adjacent CVD dE 9.1, normal-vision dE 22.9)
# --------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

_BLUE_ARM = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
_RED_ARM = ["#fadada", "#f3b3b3", "#ea8686", "#e34948", "#c22f2f", "#942020", "#611212"]
DIVERGING = LinearSegmentedColormap.from_list(
    "corr_regime", _BLUE_ARM[::-1] + ["#f0efec"] + _RED_ARM, N=256
)
SEQUENTIAL = LinearSegmentedColormap.from_list("mag", _BLUE_ARM[::-1], N=256)


def set_style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK2,
            "axes.labelsize": 10,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _grid(ax, axis="y"):
    ax.grid(True, axis=axis, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, name: str):
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"    figure -> {path.name}")
    return path


def _annotate_events(ax, events, y_frac=0.97, rotation=90, fontsize=7.5, subset=None):
    """Vertical event markers. Labels are staggered in two tiers so that
    clustered events (2018, 2020) do not overprint each other."""
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    tier = 0
    for date_str, name in events:
        if subset is not None and name not in subset:
            continue
        d = pd.Timestamp(date_str)
        if not (ax.get_xlim()[0] <= mdates.date2num(d) <= ax.get_xlim()[1]):
            continue
        ax.axvline(d, color=MUTED, linewidth=0.7, linestyle=(0, (3, 3)), alpha=0.75, zorder=1)
        y = ymin + span * (y_frac - 0.14 * (tier % 2))
        ax.text(
            d, y, f" {name}", rotation=rotation, fontsize=fontsize,
            color=INK2, va="top", ha="left", zorder=5,
        )
        tier += 1


# --------------------------------------------------------------------------
# Figure 1 -- index-level decomposition through time
# --------------------------------------------------------------------------
def fig_index_decomposition(df: pd.DataFrame, window: int, name="fig1_index_decomposition"):
    """Three stacked panels sharing the time axis: index volatility, realised
    average correlation, and the own-variance (diversifiable) share."""
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.85], "hspace": 0.13})

    ax = axes[0]
    ax.plot(df.index, df["vol_index"] * 100, color=SERIES[0], linewidth=1.4)
    ax.set_ylabel("Annualised vol (%)")
    ax.set_title(f"S&P 500 realised volatility ({window}-day trailing window)", loc="left")
    _grid(ax)
    _annotate_events(ax, config.EVENTS)

    ax = axes[1]
    from src.stats_tests import fisher_ci
    lo, hi = fisher_ci(df["rho_implied"].clip(-0.99, 0.99), window)
    ax.fill_between(df.index, lo, hi, color=SERIES[0], alpha=0.16, linewidth=0,
                    label="95% Fisher band (single-pair, conservative)")
    ax.plot(df.index, df["rho_implied"], color=SERIES[0], linewidth=1.4,
            label="Realised average correlation")
    ax.axhline(df["rho_implied"].median(), color=MUTED, linewidth=0.9,
               linestyle="--", label=f"Sample median ({df['rho_implied'].median():.2f})")
    ax.set_ylabel(r"$\bar{\rho}$")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("Realised average pairwise correlation, recovered by inverting the variance identity", loc="left")
    ax.legend(loc="upper left", ncol=3)
    _grid(ax)

    ax = axes[2]
    ax.fill_between(df.index, 0, df["own_share"] * 100, color=SERIES[1],
                    alpha=0.35, linewidth=0)
    ax.plot(df.index, df["own_share"] * 100, color=SERIES[1], linewidth=1.3)
    ax.set_ylabel("Own-variance share (%)")
    ax.set_xlabel("")
    ax.set_title("Diversifiable (own-variance) share of index variance -- collapses in crises, expands in dispersion regimes", loc="left")
    _grid(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.align_ylabels(axes)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 2 -- episode attribution
# --------------------------------------------------------------------------
def fig_episode_attribution(ep: pd.DataFrame, name="fig2_episode_attribution"):
    """Horizontal bars: how far correlation moved towards perfect co-movement
    during each episode, as a fraction of the maximum move available from that
    episode's starting level.

    Sorted chronologically, not by size, so the reader can see regime drift.
    The bar is the statistic the classification actually uses, and the colour
    is the classification, so the two cannot disagree; the direct label adds
    the concrete counterfactual (realised vol peak vs the peak that would have
    occurred had correlation stayed put), which is what makes the size of the
    effect legible in units a reader already understands."""
    d = ep.dropna(subset=["corr_uplift_frac_of_max"]).copy()
    if d.empty:
        return None
    d = d.sort_values("peak")
    labels = [
        f"{r.peak:%Y-%m}  {r.event if r.event else '(unnamed)'}"
        for r in d.itertuples()
    ]
    share = d["corr_uplift_frac_of_max"].clip(0, 1.05) * 100
    colors = [
        CRITICAL if c == "correlation-driven"
        else (SERIES[0] if c == "dispersion-driven" else MUTED)
        for c in d["classification"]
    ]

    fig, ax = plt.subplots(figsize=(13, max(5, 0.48 * len(d) + 2.4)))
    y = np.arange(len(d))
    ax.barh(y, share, color=colors, height=0.66, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK2)
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, linewidth=1.0)
    for t, lab in ((20, "dispersion\nthreshold"), (50, "correlation\nthreshold")):
        ax.axvline(t, color=MUTED, linewidth=1.0, linestyle="--", zorder=2)
        ax.text(t, -0.72, lab, fontsize=7.5, color=MUTED, ha="center", va="bottom")

    ax.set_xlabel("How far correlation travelled towards perfect co-movement,\n"
                  "as a % of the maximum possible from its pre-episode level")
    ax.set_title(
        "Volatility episodes: correlation-driven (red), mixed (grey), dispersion-driven (blue)",
        loc="left",
    )
    # Direct labels carry the concrete counterfactual, which is the point of
    # the exercise: what the vol peak was, versus what it would have been had
    # correlation not moved.
    for yi, r in zip(y, d.itertuples()):
        ax.text(r.corr_uplift_frac_of_max * 100 + 1.5, yi,
                f"  {r.corr_uplift_frac_of_max*100:.0f}%"
                f"   ({r.vol_peak*100:.0f}% vs {r.vol_if_corr_frozen*100:.0f}% counterfactual)",
                va="center", ha="left", fontsize=8, color=INK2, zorder=4)
    _grid(ax, axis="x")
    ax.set_xlim(0, 100)
    ax.set_ylim(len(d) - 0.4, -1.1)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 3 -- sector regime heatmap
# --------------------------------------------------------------------------
def fig_sector_heatmap(panel: pd.DataFrame, name="fig3_sector_heatmap",
                       title_extra=""):
    """Rows = sectors, columns = months, colour = causal percentile rank of
    realised average correlation within that sector's own history.

    A percentile rank rather than the raw level, because sectors have very
    different unconditional correlations (energy ~0.55, health care ~0.25) and
    a raw-level heatmap would show only that difference, washing out the
    regime dynamics that are the actual subject.
    """
    # Aggregate to QUARTERS and bin into five discrete regimes.
    #
    # Both choices are about legibility, and both were made after looking at
    # the monthly/continuous version, which was unreadable. A percentile rank
    # is uniform by construction, so a continuous ramp gives every cell equal
    # visual weight and the result is noise: the eye cannot find a regime in
    # it. Quarterly columns average out single-month flicker, and five bins
    # turn a continuous field into the categorical statement the figure is
    # actually making -- "this sector was in an unusually correlated state in
    # this quarter". Nothing is hidden: the underlying monthly series is in
    # outputs/tables/sector_regime_panel.csv.
    q = panel.T.resample("QE").mean().T
    data = q.to_numpy(dtype=float)

    bounds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cmap = LinearSegmentedColormap.from_list(
        "regime5", ["#2a78d6", "#9ec5f4", "#f0efec", "#ea8686", "#c22f2f"], N=5
    )
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(14, 0.66 * len(q) + 3.2))
    x = mdates.date2num(q.columns.to_pydatetime())
    xedges = np.append(x, x[-1] + 91)
    im = ax.pcolormesh(xedges, np.arange(len(q) + 1), data,
                       cmap=cmap, norm=norm, shading="flat")

    ax.set_yticks(np.arange(len(q)) + 0.5)
    ax.set_yticklabels(q.index, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(xedges[0], xedges[-1])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        "Sector correlation regimes through time" + title_extra +
        "\nQuarterly mean percentile of each sector's own realised average pairwise correlation"
        " (expanding window, causal)",
        loc="left",
    )

    # Event markers, drawn above the cells with a light halo so they read
    # against both the dark red and dark blue ends of the ramp.
    for d, nm in config.EVENTS:
        tstamp = pd.Timestamp(d)
        xv = mdates.date2num(tstamp)
        if xedges[0] <= xv <= xedges[-1]:
            ax.axvline(tstamp, color="#fcfcfb", linewidth=1.8, alpha=0.85, zorder=3)
            ax.axvline(tstamp, color=INK, linewidth=0.8, alpha=0.7, zorder=4,
                       linestyle=(0, (2, 2)))

    cb = fig.colorbar(im, ax=ax, pad=0.012, aspect=24, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    cb.ax.set_yticklabels(["0-20\ndispersion", "20-40", "40-60\nnormal",
                           "60-80", "80-100\ncorrelation"], fontsize=8)
    cb.set_label("Percentile of own correlation history", color=INK2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=INK2, length=0)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 4 -- per-sector small multiples
# --------------------------------------------------------------------------
def fig_sector_small_multiples(results: dict, name="fig4_sector_rho_panels"):
    """One panel per sector, single series each.

    Small multiples rather than six overlaid lines: six series on one frame is
    unreadable regardless of palette, and identity would be carried by colour
    alone. Here the panel title carries identity and colour carries nothing,
    which is strictly more accessible.
    """
    secs = list(results.keys())
    ncol = 2
    nrow = int(np.ceil(len(secs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.5 * nrow + 1.2),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    # Cross-sector median as a common reference in every panel.
    common = pd.concat([r["rho_implied"].rename(s) for s, r in results.items()], axis=1)
    med = common.median(axis=1)

    for ax, sec in zip(axes, secs):
        r = results[sec]
        ax.plot(med.index, med, color=MUTED, linewidth=1.0, alpha=0.85,
                label="Cross-sector median")
        # Label generically: the blue line is a DIFFERENT sector in each
        # panel, so naming one sector in the shared legend would be wrong.
        # Identity is carried by the panel title.
        ax.plot(r.index, r["rho_implied"], color=SERIES[0], linewidth=1.3,
                label="This panel's sector")
        ax.set_title(sec, loc="left", fontsize=11)
        ax.set_ylim(-0.05, 1.0)
        _grid(ax)
        mean_rho = r["rho_implied"].mean()
        ax.text(0.985, 0.06, f"mean $\\bar{{\\rho}}$ = {mean_rho:.2f}",
                transform=ax.transAxes, ha="right", fontsize=9, color=INK2)

    for ax in axes[len(secs):]:
        ax.set_visible(False)
    axes[0].legend(loc="upper left", ncol=2, fontsize=8.5)
    for ax in axes[-ncol:]:
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        "Realised average pairwise correlation within each sector's constituents",
        x=0.09, ha="left", fontsize=13, fontweight="semibold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 5 -- VIX term-structure cross-check
# --------------------------------------------------------------------------
def fig_term_structure(df: pd.DataFrame, quint: pd.DataFrame,
                       name="fig5_term_structure_check"):
    fig = plt.figure(figsize=(12.5, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.34, wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    sub = df.dropna(subset=["slope_3m_1m", "rho_implied"])
    back = sub["slope_3m_1m"] < 1.0
    ax.fill_between(sub.index, 0, 1, where=back.to_numpy(), transform=ax.get_xaxis_transform(),
                    color=CRITICAL, alpha=0.16, linewidth=0,
                    label="VIX curve in backwardation (VIX3M < VIX)")
    ax.plot(sub.index, sub["rho_implied"], color=SERIES[0], linewidth=1.3,
            label="Realised average correlation")
    ax.set_ylabel(r"$\bar{\rho}$")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("Realised correlation against VIX term-structure state", loc="left")
    ax.legend(loc="upper left", ncol=2)
    _grid(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = fig.add_subplot(gs[1, 0])
    if not quint.empty:
        xs = np.arange(len(quint))
        ax.bar(xs, quint["pct_backwardated"], color=SERIES[0], width=0.64, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{q}\n{r:.2f}" for q, r in
                            zip(quint.index, quint["mean_rho"])], fontsize=9)
        ax.set_xlabel(r"Quintile of realised $\bar{\rho}$ (mean below)")
        ax.set_ylabel("Days backwardated (%)")
        # Deliberately NOT claiming monotonicity here: Q4 sits below Q3. Only
        # the mean-slope panel on the right is monotone across all five
        # quintiles. Saying "rises monotonically" would be false.
        ax.set_title("Backwardation is concentrated in the top correlation quintile\n"
                     "(not monotone: Q4 sits below Q3)", loc="left", fontsize=10.5)
        ax.set_ylim(0, float(quint["pct_backwardated"].max()) * 1.22)
        for xi, v in zip(xs, quint["pct_backwardated"]):
            ax.text(xi, v + float(quint["pct_backwardated"].max()) * 0.035,
                    f"{v:.0f}%", ha="center", fontsize=8.5, color=INK2)
        _grid(ax)

    ax = fig.add_subplot(gs[1, 1])
    if not quint.empty:
        xs = np.arange(len(quint))
        ax.bar(xs, quint["mean_slope_3m_1m"], color=SERIES[1], width=0.64, zorder=3)
        ax.axhline(1.0, color=CRITICAL, linewidth=1.2, linestyle="--",
                   label="Flat curve (contango / backwardation boundary)")
        ax.set_xticks(xs)
        ax.set_xticklabels(quint.index, fontsize=9)
        ax.set_xlabel(r"Quintile of realised $\bar{\rho}$")
        ax.set_ylabel("Mean VIX3M / VIX")
        ax.set_ylim(min(0.95, quint["mean_slope_3m_1m"].min() - 0.03), None)
        ax.set_title("Mean curve slope by correlation quintile", loc="left", fontsize=11)
        ax.legend(loc="lower left", fontsize=8.5)
        _grid(ax)
    return _save(fig, name)


def fig_lead_lag(ll: pd.DataFrame, window: int, name="fig11_lead_lag"):
    """Why the day-level term-structure test comes out flat.

    The single most likely misreading of this chart is as a trading signal, so
    the annotation says explicitly that it is a measurement artefact.
    """
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(ll["lag_days"], ll["corr"], color=SERIES[0])
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.axvline(0, color=AXIS, linewidth=1.0, linestyle="--")
    peak_lag, peak_corr = ll.attrs["peak_lag"], ll.attrs["peak_corr"]
    ax.plot([peak_lag], [peak_corr], marker="o", markersize=8, color=CRITICAL, zorder=4)
    span = ll["corr"].max() - ll["corr"].min()
    ax.annotate(
        f"strongest at k = {peak_lag:+d} days  (corr = {peak_corr:.2f})",
        xy=(peak_lag, peak_corr), xytext=(peak_lag + 26, peak_corr + 0.22 * span),
        fontsize=9, color=INK2, ha="left",
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9),
    )
    ax.axvspan(window // 2 - 6, window // 2 + 6, color=MUTED, alpha=0.14, linewidth=0)
    ax.text(window // 2, ll["corr"].min() - 0.06 * span,
            f"half the {window}-day\nestimation window",
            fontsize=8, color=INK2, ha="center", va="top")
    ax.set_ylim(ll["corr"].min() - 0.26 * span, ll["corr"].max() + 0.10 * span)
    ax.set_xlabel(r"Lag $k$ (days): correlation measured at $t+k$ vs curve slope at $t$")
    ax.set_ylabel("Correlation")
    ax.set_title(
        "The realised correlation estimate lags the option-implied curve\n"
        "A measurement artefact of a trailing window, not a tradable lead-lag relationship",
        loc="left")
    _grid(ax)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 6 -- the 1/N result
# --------------------------------------------------------------------------
def fig_diversification_limit(sigma_bar: float, rho: float, rho_low: float = 0.20,
                              name="fig6_diversification_limit"):
    """The 1/N result, with a high- and a low-correlation regime side by side.

    Showing only the sample-median correlation makes diversification look
    almost useless (vol falls from 20% to 16%, and the y-axis has to be
    truncated to show anything at all). That is true AT rho = 0.64, but it is
    only half the story: the floor is a function of rho, so the same 500
    holdings buy a completely different amount of risk reduction in a
    dispersion regime. Plotting both regimes on one unbroken axis starting at
    zero is what makes the dependence on rho -- rather than on N -- the visible
    message, which is the actual content of the theorem.
    """
    from src.decomposition import equal_weight_terms
    ns = np.unique(np.round(np.logspace(0, 3, 90)).astype(int))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True,
                             gridspec_kw={"hspace": 0.17})
    ax = axes[0]
    for i, r in enumerate([rho, rho_low]):
        t = pd.DataFrame([equal_weight_terms(sigma_bar, r, int(n)) for n in ns])
        floor = np.sqrt(r) * sigma_bar * 100
        ax.plot(t["n"], t["total_vol"] * 100, color=SERIES[i],
                label=fr"$\bar\rho$ = {r:.2f}" +
                      (" (sample median)" if i == 0 else " (dispersion regime)"))
        ax.axhline(floor, color=SERIES[i], linestyle="--", linewidth=1.3, alpha=0.8)
        ax.text(ns[-1], floor + sigma_bar * 100 * 0.022, f"floor {floor:.1f}%",
                color=SERIES[i], fontsize=9, va="bottom", ha="right",
                fontweight="semibold")
    ax.set_xscale("log")
    ax.set_ylim(0, sigma_bar * 100 * 1.12)
    ax.set_ylabel("Annualised vol (%)")
    ax.set_title(
        fr"Diversification hits a floor set by correlation, not by N "
        fr"($\bar\sigma$ = {sigma_bar*100:.0f}% throughout)",
        loc="left")
    ax.legend(loc="upper right", ncol=2)
    _grid(ax)

    ax = axes[1]
    t = pd.DataFrame([equal_weight_terms(sigma_bar, rho, int(n)) for n in ns])
    ax.plot(t["n"], t["own_share"] * 100, color=SERIES[1],
            label=r"Own-variance (diversifiable) share $\propto 1/N$")
    for n_mark in (10, 50, 500):
        row = t[t["n"] == min(t["n"], key=lambda v: abs(v - n_mark))].iloc[0]
        ax.plot([row["n"]], [row["own_share"] * 100], marker="o", markersize=7,
                color=SERIES[1], zorder=4)
        ax.text(row["n"], row["own_share"] * 100 + 3.5,
                f"N={int(row['n'])}: {row['own_share']*100:.1f}%",
                fontsize=8.5, color=INK2, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("Number of equally weighted holdings, N (log scale)")
    ax.set_ylabel("Share of variance (%)")
    ax.set_title("The diversifiable term decays like 1/N; the correlation term does not decay at all",
                 loc="left")
    ax.legend(loc="upper right")
    _grid(ax)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 7 -- Fisher z / bootstrap comparison
# --------------------------------------------------------------------------
def fig_significance(tests: pd.DataFrame, name="fig7_significance"):
    """Forest plot of the tested correlation shifts.

    Both the naive Fisher interval and the block-bootstrap interval are drawn
    for each comparison, because the gap between them IS the result: it shows
    how much the iid assumption overstates confidence.
    """
    d = tests.dropna(subset=["delta_rho"]).copy()
    if d.empty:
        return None
    d = d.reset_index(drop=True)
    y = np.arange(len(d))

    from src.stats_tests import fisher_z, inv_fisher_z
    from scipy import stats as st
    crit = st.norm.ppf(0.975)
    naive_lo, naive_hi = [], []
    for r in d.itertuples():
        se = np.sqrt(1 / (r.n_days_a - 3) + 1 / (r.n_days_b - 3))
        dz = fisher_z(r.rho_a) - fisher_z(r.rho_b)
        naive_lo.append(inv_fisher_z(fisher_z(r.rho_b) + dz - crit * se) - r.rho_b)
        naive_hi.append(inv_fisher_z(fisher_z(r.rho_b) + dz + crit * se) - r.rho_b)

    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.62 * len(d) + 2.2)))
    ax.axvline(0, color=AXIS, linewidth=1.1)
    for i, r in enumerate(d.itertuples()):
        ax.plot([naive_lo[i], naive_hi[i]], [i - 0.13, i - 0.13], color=SERIES[1],
                linewidth=2.4, solid_capstyle="round",
                label="Naive Fisher 95% CI" if i == 0 else None, zorder=3)
        ax.plot([r.boot_ci_lo, r.boot_ci_hi], [i + 0.13, i + 0.13], color=SERIES[0],
                linewidth=2.4, solid_capstyle="round",
                label="Block-bootstrap 95% CI" if i == 0 else None, zorder=3)
        ax.plot([r.delta_rho], [i], marker="o", markersize=7, color=INK, zorder=4,
                label="Observed change" if i == 0 else None)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.label_a}\n  vs {r.label_b}" for r in d.itertuples()],
                       fontsize=8.5, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel(r"Change in realised average correlation, $\Delta\bar{\rho}$")
    ax.set_title(
        "Correlation regime shifts: naive Fisher vs block-bootstrap intervals\n"
        "The two disagree in BOTH directions -- the textbook interval is usually too wide "
        "(it prices a single pair, not an average of many),\nbut too narrow in the most "
        "violent episodes, where volatility clustering dominates",
        loc="left", fontsize=11)
    ax.legend(loc="lower right", ncol=3)
    _grid(ax, axis="x")
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 8 -- weight-estimation diagnostics
# --------------------------------------------------------------------------
def fig_weight_diagnostics(W: pd.DataFrame, r2: pd.Series,
                           name="fig8_weight_diagnostics"):
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 0.8], "hspace": 0.16})

    show = [c for c in ["XLK", "XLF", "XLV", "XLE"] if c in W.columns]
    ax = axes[0]
    for i, c in enumerate(show):
        ax.plot(W.index, W[c] * 100, color=SERIES[i], label=f"{c} ({config.SECTOR_ETFS[c]})")
        # Direct label at the right edge -- required relief for the lower-contrast slots.
        s = W[c].dropna()
        if len(s):
            ax.text(s.index[-1], s.iloc[-1] * 100, f"  {c}", color=SERIES[i],
                    fontsize=9, va="center", fontweight="semibold")
    ax.set_ylabel("Estimated index weight (%)")
    ax.set_title("Sector weights recovered from returns-based style analysis (NNLS, sum-to-one)", loc="left")
    ax.legend(loc="upper left", ncol=4)
    _grid(ax)

    ax = axes[1]
    ax.plot(r2.index, r2, color=SERIES[2], linewidth=1.4)
    ax.set_ylim(min(0.9, float(r2.min()) - 0.01), 1.001)
    ax.set_ylabel(r"Replication $R^2$")
    ax.set_title("Quality of the SPY replication -- the validity test for the weight estimator", loc="left")
    ax.text(0.995, 0.08, f"median $R^2$ = {r2.median():.4f}", transform=ax.transAxes,
            ha="right", fontsize=9.5, color=INK2)
    _grid(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 9 -- robustness across estimation windows
# --------------------------------------------------------------------------
def fig_window_robustness(by_window: dict, name="fig9_window_robustness"):
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for i, (w, df) in enumerate(sorted(by_window.items())):
        ax.plot(df.index, df["rho_implied"], color=SERIES[i], linewidth=1.15,
                alpha=0.95 if w == config.WINDOW else 0.75,
                label=f"{w}-day window" + (" (baseline)" if w == config.WINDOW else ""))
    ax.set_ylabel(r"$\bar{\rho}$")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title(
        "Robustness: the correlation series is a regime signal, not a window artefact\n"
        "Shorter windows are noisier but the regimes coincide",
        loc="left")
    ax.legend(loc="upper left", ncol=3)
    _grid(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 10 -- rho vs absorption ratio triangulation
# --------------------------------------------------------------------------
def fig_absorption_check(df: pd.DataFrame, name="fig10_absorption_check"):
    """Two independent measures of 'how systematic is the market right now'.

    Drawn as two panels, not two y-axes: they are on genuinely different
    scales and overlaying them with separate axes would invite the reader to
    read a relationship off the wiggles rather than off the statistics quoted
    in the title.
    """
    d = df.dropna(subset=["rho_implied", "absorption"])
    r = float(d["rho_implied"].corr(d["absorption"]))

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.0), sharex=True,
                             gridspec_kw={"hspace": 0.16})
    axes[0].plot(d.index, d["rho_implied"], color=SERIES[0], linewidth=1.3)
    axes[0].set_ylabel(r"$\bar{\rho}$ (inversion)")
    axes[0].set_title(
        f"Triangulation: average pairwise correlation vs first-eigenvalue share (corr = {r:.3f})",
        loc="left")
    _grid(axes[0])

    axes[1].plot(d.index, d["absorption"] * 100, color=SERIES[1], linewidth=1.3)
    axes[1].set_ylabel("Absorption ratio (%)")
    axes[1].set_title("Share of variance in the first principal component of the correlation matrix", loc="left")
    _grid(axes[1])
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, name)
