"""Plotting utilities for the particle-filter / PCE pipelines in this package
(particle_filtering_pipeline.py, offline_parameter_transform_and_pce_learning.py,
designed_sample_pce.py, pf_vs_pce_comparison.py, fuq_vs_pce_sobol_comparison.py).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

from uqef_dynamic.utils import utility
from uqef_dynamic.models.hbv_sask import hbvsask_utility as hbv
from uqef_dynamic.models.hbv_sask import HBVSASKModel as hbvmodel

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#637939', '#393b79', '#8c6d31', '#843c39', '#7b4173',
    '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d',
    '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476',
    '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc'
    ]
PARAMETERS = ["TT", "C0", "ETF", "FC", "beta", "FRAC", "K2", "LP", "K1", "alpha", "PM"]
COLORS_DICT = {PARAMETERS[idx]: COLORS[idx] for idx in range(len(PARAMETERS))}
COLORS_QOI = ['#0072B2', '#E69F00', '#CC79A7', '#009E73']


__all__ = ["plot_sensitivity_vs_identifiability", "plot_streamflow_bands",
          "plot_pooled_chains", "plot_pce_after_particle_filter",
          "plot_pce_after_particle_filter_heatmap_style",
          "plot_overlay", "plot_residuals", "plot_sobol_overlay",
          "COLORS", "PARAMETERS", "COLORS_DICT", "COLORS_QOI"]


def _savefig(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=150)
    plt.close(fig)


def _load_forcing_df(dates, configuration_file, inputModelDir, basin, out_dir, caller_name):
    """Read temperature/precipitation for `dates` via a throwaway HBVSASKModel
    (writing_results_to_a_file=False, plotting=False, so it has no side effects
    beyond reading the forcing data it already loads at construction) — shared
    by every caller that offers a "wall" forcing style, so configuration_file/
    inputModelDir/basin handling lives in one place. Notably: basin is only
    included in the constructor call when given, since HBVSASKModel infers it
    from the configuration file's own model_settings.basin when its kwarg is
    omitted entirely, but treats an explicitly passed basin=None as "use None".

    Returns None (with a printed note) if configuration_file/inputModelDir are
    not given, or `dates` did not parse as real timestamps.
    """
    if configuration_file is None or inputModelDir is None:
        print(f"{caller_name}: plot_forcing_data=True but configuration_file/"
              "inputModelDir not given — skipping the forcing panels.")
        return None
    x = pd.to_datetime(dates, errors="coerce")
    if pd.isna(x).any():
        print(f"{caller_name}: plot_forcing_data=True but dates did not parse "
              "as real timestamps — skipping the forcing panels.")
        return None
    model_kwargs = dict(configurationObject=configuration_file, inputModelDir=inputModelDir,
                        workingDir=out_dir, writing_results_to_a_file=False, plotting=False)
    if basin is not None:
        model_kwargs["basin"] = basin
    temp_model = hbvmodel.HBVSASKModel(**model_kwargs)
    forcing_full = temp_model.time_series_measured_data_df
    if forcing_full.index.name == utility.TIME_COLUMN_NAME:
        forcing_full = forcing_full.reset_index()
    return forcing_full[forcing_full[utility.TIME_COLUMN_NAME].isin(x)]


def plot_sensitivity_vs_identifiability(samples_npz, sobol_s1, out_dir=None,
                                        skip_fraction=0.1, width_threshold=0.5,
                                        s1_threshold=None, filename="sensitivity_vs_identifiability.png"):
    """Scatter forward-GSA sensitivity against posterior identifiability.

    Two orthogonal quantities that are easy to conflate:

      x  Sobol S1 from a FORWARD (prior-based) GSA — does the parameter move the
         output across its plausible range? Passed in; not computed here.
      y  posterior width / prior width — did the filter learn anything about it?
         1.0 means the posterior is as wide as the prior (learned nothing).

    A parameter can be strongly influential yet unidentifiable: a precipitation
    multiplier moves streamflow a lot, but is confounded with everything else
    that scales flow, so its posterior stays wide. That is the top-right
    quadrant, and it is a result rather than a failure.

    NOTE this deliberately avoids posterior-conditional sensitivity indices.
    Those are computed over the posterior, so a converged parameter shows a low
    index purely because its range has shrunk — confounding sensitivity with
    identifiability, the two things this plot separates.

    Args:
        samples_npz:     posterior_parameter_samples.npz, or its directory.
        sobol_s1:        dict {param_name: S1} from the forward GSA. Parameters
                         missing from it are skipped.
        out_dir:         output directory; defaults to the npz's directory.
        skip_fraction:   drop this leading fraction of dates as filter warm-up,
                         when the posterior is still collapsing from the prior.
        width_threshold: horizontal quadrant line (relative width).
        s1_threshold:    vertical quadrant line; defaults to the median S1.

    Returns:
        dict {param_name: (s1, relative_width)}.
    """
    if os.path.isdir(str(samples_npz)):
        samples_npz = os.path.join(str(samples_npz), "posterior_parameter_samples.npz")
    d = np.load(samples_npz, allow_pickle=True)
    th = np.asarray(d["theta"], dtype=np.float64)
    names = [str(x) for x in d["param_names"]]
    lo, hi = np.asarray(d["param_lower"], float), np.asarray(d["param_upper"], float)

    start = int(skip_fraction * th.shape[0])
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    rel_w = np.median(
        (np.percentile(th[start:], 90, axis=1) - np.percentile(th[start:], 10, axis=1)) / span,
        axis=0)

    pts = {n: (float(sobol_s1[n]), float(rel_w[j]))
           for j, n in enumerate(names) if n in sobol_s1}
    missing = [n for n in names if n not in sobol_s1]
    if missing:
        print(f"plot_sensitivity_vs_identifiability: no S1 given for {missing}; skipped.")
    if not pts:
        raise ValueError("sobol_s1 matched none of the parameter names in the npz.")

    xs = np.array([v[0] for v in pts.values()])
    ys = np.array([v[1] for v in pts.values()])
    xt = float(np.median(xs)) if s1_threshold is None else float(s1_threshold)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.axhline(width_threshold, color='grey', lw=0.8, ls='--')
    ax.axvline(xt, color='grey', lw=0.8, ls='--')
    ax.scatter(xs, ys, s=90, color='steelblue', zorder=3, edgecolor='white')
    for n, (x, y) in pts.items():
        ax.annotate(n, (x, y), xytext=(6, 5), textcoords='offset points', fontsize=10)

    # Corner captions in axes coordinates, inset so they cannot collide with points.
    for xa, ya, ha, va, txt in [
            (0.985, 0.985, 'right', 'top',    'sensitive,\nNOT identifiable'),
            (0.985, 0.015, 'right', 'bottom', 'sensitive,\nidentifiable'),
            (0.015, 0.985, 'left',  'top',    'insensitive,\nunconstrained'),
            (0.015, 0.015, 'left',  'bottom', 'insensitive,\nyet narrowed')]:
        ax.text(xa, ya, txt, transform=ax.transAxes, ha=ha, va=va,
                fontsize=8, color='grey', alpha=0.75,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.65))
    ax.margins(0.12)

    ax.set_xlabel('Forward-GSA Sobol $S_1$  (sensitivity, prior-based)')
    ax.set_ylabel('posterior width / prior width  (1 = nothing learned)')
    ax.set_title('Sensitivity vs identifiability')
    ax.grid(alpha=0.3)
    _savefig(fig, out_dir or os.path.dirname(os.path.abspath(samples_npz)), filename)
    return pts


def plot_streamflow_bands(dates, bands, lines, observed=None, forcing_df=None,
                          forcing_style="secondary_axis",
                          title="", out_dir=".", filename="streamflow",
                          output_formats=("pdf", "html"), warmup_steps=30,
                          y_max=None, x_range=None, show=False):
    """Shared plotly streamflow figure: shaded percentile band(s) + line(s) +
    observed + optional forcing (precipitation, optionally temperature) +
    warm-up shading + the standard title/legend layout.

    Used by both main_routine's single-chain particle_filter_streamflow.pdf
    and plot_pooled_chains' pooled_streamflow.png, so the two stay visually
    identical wherever their content actually is the same — refactored out of
    what used to be two independently-maintained implementations (one plotly,
    one matplotlib) that had drifted apart in layout and content.

    Args:
        dates: length n_dates, chronologically ordered (list or DatetimeIndex).
        bands: list of dicts, each {"upper", "lower", "name", "fillcolor"} —
              (n_dates,) arrays; drawn in list order, first = furthest back
              (so the widest band should come first).
        lines: list of dicts, each {"y", "name"} plus optional "color"
              (default "blue"), "width" (default 2), "dash" (default None),
              "visible" (True | 'legendonly', default True), "showlegend"
              (default True — set False on repeats of an already-labelled
              line, e.g. per-chain mean lines that should share one legend
              entry).
        observed: optional (n_dates,) array, drawn as the standard orange line.
        forcing_df: optional DataFrame with a "precipitation" column, drawn
              per `forcing_style`. None skips the forcing panel(s) entirely
              regardless of `forcing_style` (e.g. when forcing data isn't
              available in the caller's context).
        forcing_style: "secondary_axis" (default) draws precipitation as bars
              on a secondary y-axis overlaid on the SAME panel as the
              streamflow bands/lines (matches particle_filter_streamflow.pdf's
              original layout) — forcing_df just needs a "precipitation"
              column, any index (used directly as x).
              "wall" instead builds a 3-row subplot: temperature (row 1),
              precipitation with its y-axis reversed so it visually "hangs"
              from the top like a wall (row 2), then the streamflow
              bands/lines/observed in the last row — reusing
              hbvsask_utility._add_forcing_data directly for rows 1-2, so it
              is pixel-identical to the layout already used in the FUQ/SA
              notebook's own plotting pipeline. Needs forcing_df to have a
              utility.TIME_COLUMN_NAME column (not just an index) plus
              "temperature" and "precipitation" — exactly what
              _add_forcing_data itself expects.
        title:  full figure title (build the "N particles | P-factor=... |
              RMSE=..." string in the caller, since what varies — pooled vs.
              single-chain — differs enough that a shared template would
              need as many parameters as just building the string directly).
        out_dir, filename: written to <out_dir>/<filename>.<ext> for each
              ext in output_formats. "html" uses plotly's own interactive
              writer; anything else goes through fig.write_image (needs
              kaleido — a failure there is caught and printed, not raised,
              matching every other plot in this module).
        warmup_steps: dates[0:warmup_steps] shaded grey and annotated
              "Warm-up" — the first steps carry the prior's spread, which is
              orders of magnitude wider than anything afterward. Spans every
              row when forcing_style="wall", so the warm-up period reads
              consistently across the temperature/precipitation panels too.
        y_max: fixed axis max; default 1.4x max(observed), falling back to
              1.4x max(lines[0]["y"]) when there is no observed series.
        x_range: optional explicit [start, end] for the x-axis (e.g. the
              model's own start_date_predictions/end_date), overriding
              plotly's auto-range from `dates`.
        show: call fig.show() before saving (matches main_routine's own
              single-chain call, which always showed the figure; default
              False so batch/pooling code doesn't pop up a browser tab).

    Returns:
        The plotly Figure.
    """
    dates = list(dates)
    has_forcing = forcing_df is not None and "precipitation" in forcing_df.columns
    use_wall = has_forcing and forcing_style == "wall"

    if use_wall:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.15, 0.15, 0.7],
            subplot_titles=("Temperature [°C]", "Precipitation [mm/day]", None))
        fig = hbv._add_forcing_data(fig, forcing_df)
        main_row = 3

        def add(trace):
            fig.add_trace(trace, row=main_row, col=1)
    else:
        fig = go.Figure()

        def add(trace):
            fig.add_trace(trace)

    for b in bands:
        add(go.Scatter(
            x=dates + dates[::-1], y=list(b["upper"]) + list(b["lower"])[::-1],
            fill="toself", fillcolor=b["fillcolor"], line=dict(color="rgba(0,0,0,0)"),
            name=b["name"], hoverinfo="skip"))

    if has_forcing and not use_wall:
        N_max = forcing_df["precipitation"].max()
        fig.add_trace(go.Bar(
            x=forcing_df.index, y=forcing_df["precipitation"],
            name="Precipitation", yaxis="y2", marker_color="rgba(31,119,180,0.5)"))

    if observed is not None:
        add(go.Scatter(
            x=dates, y=list(observed), name="Observed",
            line=dict(color="orange", width=2.5)))

    for l in lines:
        add(go.Scatter(
            x=dates, y=list(l["y"]), name=l["name"], mode="lines",
            line=dict(color=l.get("color", "blue"), width=l.get("width", 2),
                      dash=l.get("dash")),
            visible=l.get("visible", True), showlegend=l.get("showlegend", True)))

    if y_max is None:
        if observed is not None and np.any(np.isfinite(np.asarray(observed, dtype=float))):
            y_max = float(np.nanmax(observed))
        elif lines:
            y_max = float(np.nanmax(lines[0]["y"]))

    xaxis_kwargs = dict(title_text="Date", type="date")
    if x_range is not None:
        xaxis_kwargs["range"] = x_range
    yaxis_kwargs = dict(title_text="Q [m³/s]", mirror=True)
    if has_forcing and not use_wall:
        yaxis_kwargs.update(side="left", domain=[0, 0.7],
                            tickfont={"color": "#d62728"},
                            title=dict(font={"color": "#d62728"}))
    if y_max is not None and np.isfinite(y_max) and y_max > 0:
        yaxis_kwargs["range"] = [0, y_max * 1.4]
    if use_wall:
        fig.update_xaxes(**xaxis_kwargs, row=main_row, col=1)
        fig.update_yaxes(**yaxis_kwargs, row=main_row, col=1)
    else:
        fig.update_xaxes(**xaxis_kwargs)
        fig.update_yaxes(**yaxis_kwargs)

    if has_forcing and not use_wall:
        fig.update_layout(yaxis2=dict(
            anchor="x", domain=[0.7, 1], mirror=True,
            range=[N_max, 0], side="right",
            tickfont={"color": "#1f77b4"}, nticks=3,
            title=dict(text="N [mm/h]", font={"color": "#1f77b4"}),
            type="linear"))

    if len(dates) > warmup_steps:
        vrect_kwargs = dict(
            x0=dates[0], x1=dates[warmup_steps - 1],
            fillcolor="grey", opacity=0.12, layer="below", line_width=0,
            annotation_text="Warm-up", annotation_position="top left",
            annotation_font_size=11, annotation_font_color="grey")
        if use_wall:
            fig.add_vrect(**vrect_kwargs, row="all", col=1)
        else:
            fig.add_vrect(**vrect_kwargs)

    # In "wall" mode, row 1's own subplot title ("Temperature [°C]") sits
    # right at the top of the plotting grid, immediately below whatever top
    # margin holds the figure title + legend - a bigger top margin (not a
    # bigger legend.y, which only pushes the legend ABOVE the figure title
    # instead of making room) is what actually keeps them from touching.
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=title, showlegend=True, template="plotly_white",
        margin=dict(t=140) if use_wall else {})

    if show:
        fig.show()
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    height = 900 if use_wall else 700
    for ext in output_formats:
        path = os.path.join(out_dir, f"{filename}.{ext}")
        if ext == "html":
            pyo.plot(fig, filename=path, auto_open=False)
        else:
            try:
                fig.write_image(path, width=1400, height=height)
            except Exception as e:
                print(f"{ext.upper()} export skipped (install kaleido): {e}")
    return fig


def plot_pooled_chains(results, out_dir, observed=None, pooled_theta=None,
                       plot_forcing_data=False, configuration_file=None,
                       inputModelDir=None, basin=None):
    """Three diagnostic figures from a pool_chain_results dict.

    Args:
        plot_forcing_data: add temperature/precipitation panels to the
              pooled hydrograph, in the "wall" style (see
              plot_streamflow_bands' forcing_style docstring) — matches the
              layout already used in the FUQ/SA notebook's own plotting
              pipeline. Needs configuration_file and inputModelDir (the same
              ones that produced the pooled chains), since pooling itself
              never touches forcing data. Loaded via _load_forcing_df, which
              also documents the configuration_file/inputModelDir/basin
              handling below.
        configuration_file, inputModelDir, basin: passed straight to
              _load_forcing_df.
    """
    x = pd.to_datetime(results["dates"], errors="coerce")
    dates_are_real = not pd.isna(x).any()
    if not dates_are_real:
        x = np.arange(results["n_dates"])
    pct, cm = results["pooled_percentiles"], results["chain_means"]
    lo, hi = min(pct), max(pct)

    # 1 — pooled hydrograph with per-chain means overlaid. Built via the same
    # plot_streamflow_bands used for the single-chain particle_filter_streamflow
    # plot, so the two share layout/content wherever the pooled context actually
    # has the same information (bands, median-less mean line, observed, warm-up
    # shading, title/legend style, and now the forcing panels when requested).
    # One thing this still cannot match: the raw-Q-only reference band (
    # pool_chain_results only pools the reported Q, not a separate raw/
    # AR-uncorrected series).
    bands = [{"upper": pct[hi], "lower": pct[lo],
             "name": f"{lo}–{hi}% pooled band", "fillcolor": "rgba(173,216,230,0.35)"}]
    if 25 in pct and 75 in pct:
        bands.append({"upper": pct[75], "lower": pct[25],
                      "name": "25–75% pooled band", "fillcolor": "rgba(70,130,180,0.35)"})

    lines = [{"y": c, "name": "Individual chain means", "color": "grey", "width": 0.7,
             "showlegend": (i == 0)} for i, c in enumerate(cm)]
    lines.append({"y": results["pooled_mean"], "name": "Pooled mean",
                  "color": "blue", "width": 2})

    title = f'Pooled Particle Filter — {results["n_chains"]}×{results["n_particles_per_chain"]} particles'
    if "p_factor_pooled" in results and "rmse_pooled_mean" in results:
        title += (f'  |  P-factor={results["p_factor_pooled"]:.2f}'
                 f'  |  RMSE={results["rmse_pooled_mean"]:.2f} m³/s')

    forcing_df = None
    if plot_forcing_data:
        # results["dates"], not x: once dates_are_real is False, x has already
        # been replaced by np.arange(n_dates) above, and pd.to_datetime on
        # plain integers silently succeeds (epoch nanoseconds) instead of
        # raising/NaT-ing, which would defeat _load_forcing_df's own check.
        forcing_df = _load_forcing_df(results["dates"], configuration_file, inputModelDir,
                                      basin, out_dir, "plot_pooled_chains")

    plot_streamflow_bands(
        dates=x, bands=bands, lines=lines, observed=observed,
        forcing_df=forcing_df, forcing_style="wall",
        title=title, out_dir=out_dir, filename="pooled_streamflow",
        output_formats=("png",), warmup_steps=30)

    # 2 — between- vs within-chain variance (has the initial sample stopped mattering?)
    B, W = results["between_chain_var"], results["within_chain_var"]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(W > 0, B / W, np.nan)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    a1.plot(x, B, color='tomato', lw=1, label='between-chain variance')
    a1.plot(x, W, color='steelblue', lw=1, label='within-chain variance')
    a1.set_yscale('log'); a1.set_ylabel('variance')
    _floor = 1.0 / results["n_particles_per_chain"]
    a1.set_title(f'Chain agreement — median B/W = {results["between_over_within_median"]:.5f}, '
                 f'Monte Carlo floor 1/N = {_floor:.5f}')
    a2.plot(x, ratio, color='purple', lw=1, label='B / W')
    a2.axhline(_floor, color='red', ls='--', lw=0.9,
               label=f'MC floor 1/N = {_floor:.1e}  (chains identical up to sampling error)')
    a2.set_yscale('log'); a2.set_ylabel('B / W'); a2.set_xlabel('Date')
    for a in (a1, a2):
        a.legend(fontsize=8); a.grid(alpha=0.3)
    _savefig(fig, out_dir, "chain_agreement.png")

    # 3 — pooled vs per-chain parameter posteriors at the final timestep
    if pooled_theta is not None:
        names, per = results["param_names"], results["n_particles_per_chain"]
        fig, axs = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2))
        axs = np.atleast_1d(axs)
        for j, (ax, nm) in enumerate(zip(axs, names)):
            v = pooled_theta[-1, :, j]
            ax.hist(v, bins=40, density=True, alpha=0.45, color='steelblue', label='pooled')
            for c in range(results["n_chains"]):
                ax.hist(v[c * per:(c + 1) * per], bins=40, density=True,
                        histtype='step', lw=0.8, alpha=0.7)
            ax.set_xlabel(nm); ax.grid(alpha=0.3)
        axs[0].set_ylabel('density'); axs[0].legend(fontsize=8)
        fig.suptitle('Parameter posteriors, final timestep — filled = pooled, outlines = chains')
        _savefig(fig, out_dir, "pooled_parameter_posteriors.png")


def plot_pce_after_particle_filter(pce_result, working_dir, out_dir=None,
                                   light_output=False, plot_generalized_sobol=False,
                                   filename="pce_after_particle_filter_streamflow"):
    """Streamflow from a fitted PCE against observed, in the style of
    particle_filtering_pipeline.py's particle_filter_streamflow.pdf (band +
    mean + observed), plus a panel of each parameter's total-order Sobol index
    over the same dates, and optionally a third panel of the generalized
    (time-averaged, variance-weighted) total-Sobol index — see
    offline_parameter_transform_and_pce_learning.compute_generalized_sobol_indices
    for what "generalized" means here.

    Args:
        pce_result:  dict returned by run_pce_learning (or run_offline's
                    "pce" entry, or designed_sample_pce.run_designed_sample_pce),
                    or load_pce_output(...) on its saved .npz.
        working_dir: run folder holding averaged_and_simulated.pkl, read here
                    only for the observed-streamflow overlay. This is the
                    ORIGINAL particle_filtering_pipeline.main_routine output
                    folder — neither PCE-building path writes this file
                    itself, so working_dir must point at (or inherit from,
                    like a pooled run's directory does) that original run.
        out_dir:     where to write the .pdf/.html; default working_dir.
        plot_generalized_sobol: add a third panel for Sobol_t_generalized,
                    below the time-frozen one. Silently falls back to the
                    2-panel layout (with a printed note) if pce_result has no
                    "Sobol_t_generalized" key or it is entirely NaN — e.g. a
                    pce_result fit with compute_generalized_sobol=False, or
                    saved before that field existed.

    Returns:
        The plotly Figure.
    """
    dates = [pd.Timestamp(str(d)) for d in pce_result["dates"]]
    E = np.asarray(pce_result["E"])
    P10 = np.asarray(pce_result["P10"])
    P90 = np.asarray(pce_result["P90"])
    Sobol_t = np.asarray(pce_result["Sobol_t"])
    names = [str(x) for x in pce_result["param_names"]]

    sobol_t_generalized = None
    if plot_generalized_sobol:
        sobol_t_generalized = pce_result.get("Sobol_t_generalized")
        if sobol_t_generalized is None or np.all(np.isnan(np.asarray(sobol_t_generalized))):
            print("plot_generalized_sobol=True but pce_result has no usable "
                  "'Sobol_t_generalized' (missing, or all-NaN — fit with "
                  "compute_generalized_sobol=True to get it); falling back to "
                  "the 2-panel layout.")
            sobol_t_generalized = None
        else:
            sobol_t_generalized = np.asarray(sobol_t_generalized)

    obs_df = pd.read_pickle(
        os.path.join(str(working_dir), "averaged_and_simulated.pkl"), compression="gzip")
    observed = obs_df["observed_streamflow"].reindex(dates).to_numpy()

    # One stable color per parameter, reused across BOTH Sobol panels (falls
    # back to the plain COLORS cycle for a name outside COLORS_DICT) so the
    # same parameter reads as the same color whether the record is time-frozen
    # or generalized — plotly's own auto-cycling would otherwise assign each
    # panel's traces independently and silently mismatch the two.
    param_colors = [COLORS_DICT.get(name, COLORS[j % len(COLORS)])
                    for j, name in enumerate(names)]

    if sobol_t_generalized is not None:
        n_rows, row_heights = 3, [0.46, 0.27, 0.27]
        subplot_titles = ("Streamflow: PCE vs observed",
                          "Total-order Sobol index over time",
                          "Generalized (time-averaged) total-order Sobol index")
    else:
        n_rows, row_heights = 2, [0.62, 0.38]
        subplot_titles = ("Streamflow: PCE vs observed",
                          "Total-order Sobol index over time")

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=row_heights, subplot_titles=subplot_titles)

    fig.add_trace(go.Scatter(
        x=dates + dates[::-1], y=list(P90) + list(P10[::-1]),
        fill="toself", fillcolor="rgba(173,216,230,0.35)",
        line=dict(color="rgba(0,0,0,0)"),
        name="10-90% band (PCE)", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=observed, name="Observed",
        line=dict(color="orange", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=E, name="PCE mean", line=dict(color="blue", width=2)), row=1, col=1)

    for j, name in enumerate(names):
        fig.add_trace(go.Scatter(
            x=dates, y=Sobol_t[:, j], name=name, mode="lines",
            line=dict(color=param_colors[j], width=1.5)), row=2, col=1)

    if sobol_t_generalized is not None:
        for j, name in enumerate(names):
            fig.add_trace(go.Scatter(
                x=dates, y=sobol_t_generalized[:, j], name=name, mode="lines",
                line=dict(color=param_colors[j], width=1.5, dash="dash"),
                showlegend=False), row=3, col=1)  # legend entry already added by the time-frozen panel

    fig.update_yaxes(title_text="Q [m³/s]", row=1, col=1)
    fig.update_yaxes(title_text="Sobol_t", row=2, col=1)
    if sobol_t_generalized is not None:
        fig.update_yaxes(title_text="Generalized Sobol_t", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=n_rows, col=1)
    fig.update_layout(
        template="plotly_white", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        title="PCE surrogate after particle filtering",
        margin=dict(t=140))

    out_dir = str(out_dir or working_dir)
    if not light_output:
        pyo.plot(fig, filename=os.path.join(out_dir, filename + ".html"), auto_open=False)
    try:
        fig.write_image(os.path.join(out_dir, filename + ".pdf"), width=1400,
                        height=(900 if n_rows == 2 else 1150))
    except Exception as e:
        print(f"PDF export skipped (install kaleido): {e}")
    return fig


def plot_pce_after_particle_filter_heatmap_style(
        pce_result, working_dir, out_dir=None,
        plot_forcing_data=True, configuration_file=None, inputModelDir=None, basin=None,
        plot_heatmap=True, colorscale="Viridis",
        plot_generalized_sobol=False, add_generalized_as_heatmap=None,
        light_output=False, filename="pce_after_particle_filter_streamflow_heatmap",
        height=1400, width=1100, white_template=False,
        dtick="M2", tickformat="%b %y",
        legend_orientation="h", legend_yanchor="bottom", legend_xanchor="right",
        legend_y=1.02, legend_x=1.0,
        top_margin=20, bottom_margin=10, left_margin=20, right_margin=20):
    """Like plot_pce_after_particle_filter, but in the visual language of the
    forward-UQ/SA notebook's own plotting cell (HBV_SASK_UQ_and_SA_Plotting.ipynb,
    last code cell) rather than this module's own: forcing data as a
    temperature/precipitation "wall" (hbvsask_utility._add_forcing_data), Sobol
    indices optionally drawn as a heatmap instead of one line per parameter
    (utility._update_fig_with_si_data_for_single_df), and the figure finished
    off via utility._update_fig_layout_and_save rather than this module's own
    fig.update_layout block — so a plot produced here and one produced by the
    notebook are visually consistent. plot_pce_after_particle_filter itself is
    untouched; this is an alternative, not a replacement.

    Args:
        pce_result:  dict returned by run_pce_learning / run_designed_sample_pce*
                    (or load_pce_output(...) on its saved .npz).
        working_dir: run folder holding averaged_and_simulated.pkl, read here
                    only for the observed-streamflow overlay.
        out_dir:     where to write the .pdf/.html; default working_dir.
        plot_forcing_data, configuration_file, inputModelDir, basin: add the
                    temperature/precipitation "wall" rows; see _load_forcing_df.
                    Silently skipped (with a printed note) if
                    configuration_file/inputModelDir are not given.
        plot_heatmap: draw the Sobol_t panel as a go.Heatmap (one row per
                    parameter, color = index value) instead of one line per
                    parameter — matches the notebook's own plot_heatmap flag.
        colorscale:  plotly colorscale for the heatmap panel(s), e.g.
                    "Viridis", "Plasma".
        plot_generalized_sobol: add a further panel for Sobol_t_generalized.
                    Falls back silently (with a printed note) if pce_result has
                    no usable "Sobol_t_generalized".
        add_generalized_as_heatmap: whether the generalized panel is drawn as
                    a heatmap too; defaults to plot_heatmap's own value.
        dtick, tickformat, legend_*, *_margin, height, width, white_template:
                    forwarded to utility._update_fig_layout_and_save.

    Returns:
        The plotly Figure.
    """
    dates = [pd.Timestamp(str(d)) for d in pce_result["dates"]]
    E = np.asarray(pce_result["E"])
    P10 = np.asarray(pce_result["P10"])
    P90 = np.asarray(pce_result["P90"])
    Sobol_t = np.asarray(pce_result["Sobol_t"])
    names = [str(x) for x in pce_result["param_names"]]

    sobol_t_generalized = None
    if plot_generalized_sobol:
        sobol_t_generalized = pce_result.get("Sobol_t_generalized")
        if sobol_t_generalized is None or np.all(np.isnan(np.asarray(sobol_t_generalized))):
            print("plot_generalized_sobol=True but pce_result has no usable "
                  "'Sobol_t_generalized' (missing, or all-NaN — fit with "
                  "compute_generalized_sobol=True to get it); dropping that panel.")
            sobol_t_generalized = None
        else:
            sobol_t_generalized = np.asarray(sobol_t_generalized)

    obs_df = pd.read_pickle(
        os.path.join(str(working_dir), "averaged_and_simulated.pkl"), compression="gzip")
    observed = obs_df["observed_streamflow"].reindex(dates).to_numpy()

    out_dir = str(out_dir or working_dir)
    os.makedirs(out_dir, exist_ok=True)

    forcing_df = None
    if plot_forcing_data:
        forcing_df = _load_forcing_df(pce_result["dates"], configuration_file, inputModelDir,
                                      basin, out_dir, "plot_pce_after_particle_filter_heatmap_style")

    # ── Row layout, mirroring the notebook cell's own n_rows/subplot_titles
    # bookkeeping (forcing rows, then one row per content panel) ──
    n_rows, subplot_titles = 0, []
    if forcing_df is not None:
        n_rows += 2
        subplot_titles += ["Temperature [°C]", "Precipitation [mm/day]"]
    qoi_row = n_rows + 1
    n_rows += 1
    subplot_titles.append("PCE: QoI - Streamflow [m³/s]")
    sobol_row = n_rows + 1
    n_rows += 1
    subplot_titles.append("Time-frozen Total-order Sobol S.I.")
    if sobol_t_generalized is not None:
        gen_row = n_rows + 1
        n_rows += 1
        subplot_titles.append("Generalized Total-order Sobol S.I.")

    n_forcing_rows = 2 if forcing_df is not None else 0
    n_content_rows = n_rows - n_forcing_rows
    row_heights = ([0.12] * n_forcing_rows
                   + [(1.0 - 0.12 * n_forcing_rows) / n_content_rows] * n_content_rows)

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False,
                        vertical_spacing=0.05, subplot_titles=subplot_titles,
                        row_heights=row_heights)

    if forcing_df is not None:
        fig = hbv._add_forcing_data(fig, forcing_df)

    df_qoi = pd.DataFrame({utility.TIME_COLUMN_NAME: dates, utility.MEAN_ENTRY: E,
                           "P10": P10, "P90": P90, utility.MEASURED_ENTRY: observed})
    fig.add_trace(go.Scatter(
        x=df_qoi[utility.TIME_COLUMN_NAME], y=df_qoi[utility.MEASURED_ENTRY],
        name="Observed Streamflow [m³/s]", mode="lines",
        line=dict(color="green")), row=qoi_row, col=1)
    fig.add_trace(go.Scatter(
        x=df_qoi[utility.TIME_COLUMN_NAME], y=df_qoi[utility.MEAN_ENTRY],
        name="Mean predicted (PCE)", mode="lines",
        line=dict(color=COLORS_QOI[0])), row=qoi_row, col=1)
    fig = utility._add_10_90_percentiles(fig, df_qoi, row=qoi_row, col=1, showlegend=True)

    df_sobol = pd.DataFrame({utility.TIME_COLUMN_NAME: dates,
                             **{name: Sobol_t[:, j] for j, name in enumerate(names)}})
    fig = utility._update_fig_with_si_data_for_single_df(
        fig, current_df=df_sobol, plot_heatmap=plot_heatmap,
        si_columns_to_plot=names, si_columns_to_label=names,
        current_row=sobol_row, color_dict=COLORS_DICT,
        showscale=plot_heatmap, showlegend=not plot_heatmap, colorscale=colorscale)

    if sobol_t_generalized is not None:
        gen_as_heatmap = plot_heatmap if add_generalized_as_heatmap is None else add_generalized_as_heatmap
        # A colorbar's position is hardcoded inside _add_sensitivity_indices_as_heatmap
        # (x=1.0, y=0.5), so a second one from this panel would stack on top of
        # the Sobol_t panel's own colorbar rather than moving aside — suppressed
        # here whenever that panel already shows one, matching the notebook cell
        # this mirrors (`if add_generalized_as_heatmap: if plot_heatmap: showscale = False`).
        gen_showscale = gen_as_heatmap and not plot_heatmap
        df_sobol_gen = pd.DataFrame({utility.TIME_COLUMN_NAME: dates,
                                     **{name: sobol_t_generalized[:, j] for j, name in enumerate(names)}})
        fig = utility._update_fig_with_si_data_for_single_df(
            fig, current_df=df_sobol_gen, plot_heatmap=gen_as_heatmap,
            si_columns_to_plot=names, si_columns_to_label=names,
            current_row=gen_row, color_dict=COLORS_DICT,
            showscale=gen_showscale, showlegend=not gen_as_heatmap and not plot_heatmap,
            colorscale=colorscale)

    fig = utility._update_fig_layout_and_save(
        fig, out_dir, filename + ".pdf", dates[0], dates[-1],
        title=None, plotting_generalized_indices=False,
        height=height, width=width, white_template=white_template,
        save_fig=False,  # this module handles html/pdf export itself below, like every other plot here
        dtick=dtick, tickformat=tickformat,
        legend_orientation=legend_orientation, legend_yanchor=legend_yanchor,
        legend_xanchor=legend_xanchor, legend_y=legend_y, legend_x=legend_x, showlegend=True,
        top_margin=top_margin, bottom_margin=bottom_margin,
        left_margin=left_margin, right_margin=right_margin)

    if not light_output:
        pyo.plot(fig, filename=os.path.join(out_dir, filename + ".html"), auto_open=False)
    try:
        fig.write_image(os.path.join(out_dir, filename + ".pdf"), height=height, width=width)
    except Exception as e:
        print(f"PDF export skipped (install kaleido): {e}")
    return fig


def plot_overlay(df, out_dir, pf_band=None, filename="pf_vs_pce_streamflow_overlay",
                 light_output=False, plot_forcing_data=False,
                 configuration_file=None, inputModelDir=None, basin=None,
                 warmup_steps=30, y_max=None):
    """Observed + PF mean + PCE mean, each with its own uncertainty band, on
    one shared time axis — optionally with a forcing "wall" (temperature +
    precipitation) ahead of it, in the same layout plot_streamflow_bands uses.

    Args:
        df: as returned by load_pf_pce_observed — needs "observed", "PF_mean",
            "PF_P_lo", "PF_P_hi", "PCE_mean", "PCE_P10", "PCE_P90" columns.
        pf_band: (lo, hi) percentile pair actually backing PF_P_lo/PF_P_hi, as
            returned by load_pf_pce_observed — only used to label the PF band
            correctly (pool_chain_results may not have saved a 10/90 pair,
            see load_pf_pce_observed's own docstring). Defaults to (10, 90).
        plot_forcing_data, configuration_file, inputModelDir, basin: add the
            temperature/precipitation "wall" rows ahead of the streamflow
            panel — see _load_forcing_df. Silently skipped (with a printed
            note) if configuration_file/inputModelDir are not given.
        warmup_steps: dates[0:warmup_steps] shaded grey and annotated
            "Warm-up", and excluded from the y-axis range (see y_max) — the
            first PF particles carry the prior's full spread, orders of
            magnitude wider than anything afterward.
        y_max: fixed axis max; default 1.4x max(observed) — deliberately NOT
            derived from the PF/PCE bands, which can spike hugely during
            warm-up and would otherwise flatten the rest of the record.

    Returns:
        The plotly Figure.
    """
    dates = list(df[utility.TIME_COLUMN_NAME])
    lo, hi = pf_band if pf_band is not None else (10, 90)

    forcing_df = None
    if plot_forcing_data:
        forcing_df = _load_forcing_df(dates, configuration_file, inputModelDir,
                                      basin, out_dir, "plot_overlay")

    if forcing_df is not None:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.15, 0.15, 0.7],
            subplot_titles=("Temperature [°C]", "Precipitation [mm/day]", None))
        fig = hbv._add_forcing_data(fig, forcing_df)
        main_row = 3

        def add(trace):
            fig.add_trace(trace, row=main_row, col=1)
    else:
        fig = go.Figure()

        def add(trace):
            fig.add_trace(trace)

    # Bands first/furthest back — one per signal, each a translucent fill.
    add(go.Scatter(
        x=dates + dates[::-1], y=list(df["PF_P_hi"]) + list(df["PF_P_lo"])[::-1],
        fill="toself", fillcolor="rgba(44,160,44,0.25)", line=dict(color="rgba(0,0,0,0)"),
        name=f"{lo}–{hi}% band (PF)", hoverinfo="skip"))
    add(go.Scatter(
        x=dates + dates[::-1], y=list(df["PCE_P90"]) + list(df["PCE_P10"])[::-1],
        fill="toself", fillcolor="rgba(31,119,180,0.25)", line=dict(color="rgba(0,0,0,0)"),
        name="10–90% band (PCE)", hoverinfo="skip"))

    add(go.Scatter(x=dates, y=df["observed"], name="Observed",
                  line=dict(color="orange", width=2.5)))
    add(go.Scatter(x=dates, y=df["PF_mean"], name="PF mean (forecast)",
                  line=dict(color="green", width=2)))
    add(go.Scatter(x=dates, y=df["PCE_mean"], name="PCE mean",
                  line=dict(color="blue", width=2, dash="dash")))

    if y_max is None:
        obs = np.asarray(df["observed"], dtype=float)
        y_max = float(np.nanmax(obs)) if np.any(np.isfinite(obs)) else None

    xaxis_kwargs = dict(title_text="Date", type="date")
    yaxis_kwargs = dict(title_text="Q [m³/s]", mirror=True)
    if y_max is not None and np.isfinite(y_max) and y_max > 0:
        yaxis_kwargs["range"] = [0, y_max * 1.4]
    if forcing_df is not None:
        fig.update_xaxes(**xaxis_kwargs, row=main_row, col=1)
        fig.update_yaxes(**yaxis_kwargs, row=main_row, col=1)
    else:
        fig.update_xaxes(**xaxis_kwargs)
        fig.update_yaxes(**yaxis_kwargs)

    if len(dates) > warmup_steps:
        vrect_kwargs = dict(
            x0=dates[0], x1=dates[warmup_steps - 1],
            fillcolor="grey", opacity=0.12, layer="below", line_width=0,
            annotation_text="Warm-up", annotation_position="top left",
            annotation_font_size=11, annotation_font_color="grey")
        if forcing_df is not None:
            fig.add_vrect(**vrect_kwargs, row="all", col=1)
        else:
            fig.add_vrect(**vrect_kwargs)

    fig.update_layout(
        template="plotly_white", title="Streamflow: PF mean vs PCE mean vs observed",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=140) if forcing_df is not None else {})

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    height = 900 if forcing_df is not None else 600
    if not light_output:
        fig.write_html(os.path.join(out_dir, filename + ".html"))
    try:
        fig.write_image(os.path.join(out_dir, filename + ".pdf"), width=1400, height=height)
    except Exception as e:
        print(f"PDF export skipped (install kaleido): {e}")
    return fig


def plot_residuals(df, out_dir, filename="pf_vs_pce_residuals", light_output=False):
    """PF-observed and PCE-observed over time (top), and PF-PCE (bottom) - shows
    WHEN the two signals disagree with truth and with each other, which an
    aggregate GoF number cannot."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=("Residual vs observed", "PF mean - PCE mean"))

    fig.add_trace(go.Scatter(x=df[utility.TIME_COLUMN_NAME], y=df["PF_mean"] - df["observed"],
                             name="PF - observed", line=dict(color="green", width=1.5)),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=df[utility.TIME_COLUMN_NAME], y=df["PCE_mean"] - df["observed"],
                             name="PCE - observed", line=dict(color="blue", width=1.5)),
                 row=1, col=1)
    fig.add_hline(y=0, line=dict(color="black", width=0.7, dash="dot"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df[utility.TIME_COLUMN_NAME], y=df["PF_mean"] - df["PCE_mean"],
                             name="PF - PCE", line=dict(color="purple", width=1.5),
                             showlegend=False),
                 row=2, col=1)
    fig.add_hline(y=0, line=dict(color="black", width=0.7, dash="dot"), row=2, col=1)

    fig.update_yaxes(title_text="Q [m³/s]", row=1, col=1)
    fig.update_yaxes(title_text="Q [m³/s]", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        title="Where PF and PCE means depart from observed, and from each other",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        margin=dict(t=140))

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if not light_output:
        fig.write_html(os.path.join(out_dir, filename + ".html"))
    try:
        fig.write_image(os.path.join(out_dir, filename + ".pdf"), width=1400, height=800)
    except Exception as e:
        print(f"PDF export skipped (install kaleido): {e}")
    return fig


def plot_sobol_overlay(df, param_names, out_dir, filename="fuq_vs_pce_sobol_m_overlay",
                       light_output=False):
    """One subplot per parameter: FUQ's Sobol_m(t) vs PCE's Sobol_m(t)."""
    fig = make_subplots(
        rows=len(param_names), cols=1, shared_xaxes=True, vertical_spacing=0.02,
        subplot_titles=param_names)
    for j, p in enumerate(param_names):
        row = j + 1
        fig.add_trace(go.Scatter(
            x=df[utility.TIME_COLUMN_NAME], y=df[f"Sobol_m_{p}_FUQ"], name="FUQ (prior, MC)",
            line=dict(color="crimson", width=1.3), showlegend=(j == 0)), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df[utility.TIME_COLUMN_NAME], y=df[f"Sobol_m_{p}_PCE"], name="PCE (evolving posterior)",
            line=dict(color="teal", width=1.3, dash="dash"), showlegend=(j == 0)), row=row, col=1)
        fig.update_yaxes(title_text=p, row=row, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Date", row=len(param_names), col=1)
    fig.update_layout(
        template="plotly_white", height=180 * len(param_names) + 120,
        title="Sobol_m over time: forward-UQ prior vs PF-PCE evolving posterior",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5),
        margin=dict(t=140))

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if not light_output:
        fig.write_html(os.path.join(out_dir, filename + ".html"))
    try:
        fig.write_image(os.path.join(out_dir, filename + ".pdf"), width=1200,
                        height=180 * len(param_names) + 120)
    except Exception as e:
        print(f"PDF export skipped (install kaleido): {e}")
    return fig
