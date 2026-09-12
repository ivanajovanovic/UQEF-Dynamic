import pathlib
import pandas as pd
import sys
import os
import json
import time
import shutil
import inspect
import subprocess
from collections import defaultdict
import numpy as np
from scipy.stats import t as student_t

# importing modules/libs for plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib.pyplot as plt

# for parallel computing
import multiprocessing

import chaospy as cp

from uqef_dynamic.utils import utility
from uqef_dynamic.utils import mpart_transport      # MParT backend (optional dependency)
from uqef_dynamic.utils import gaussian_anamorphosis  # rank-based marginal transform
from uqef_dynamic.utils import transport_timeseries   # per-timestep driver + dispatcher
from uqef_dynamic.utils.transport_timeseries import gaussianize_parameter_samples
from uqef_dynamic.models.hbv_sask import hbvsask_utility as hbv
from uqef_dynamic.models.hbv_sask import HBVSASKModel as hbvmodel

PLOT_FORCING_DATA = True

#########################
# Set of utility functions 


def parameter_output_correlation(theta, qoi):
    """Median |corr(parameter, Q)| across particles, per parameter.

    At each timestep the correlation is taken ACROSS PARTICLES (not across time):
    particle i carries theta_i and produced Q_i, so this measures whether varying
    a parameter actually moves the output. If it is ~0 the likelihood has nothing
    to select on and the filter cannot learn that parameter, however sharp the
    likelihood is made.

    Args:
        theta: (n_dates, n_particles, n_params)
        qoi:   (n_dates, n_particles)

    Returns:
        (n_params,) array of median |r| over timesteps. Dates where the ensemble
        has collapsed (zero variance) are skipped rather than counted as zero.
    """
    th = np.asarray(theta, dtype=np.float64)
    q = np.asarray(qoi, dtype=np.float64)
    th = th - th.mean(axis=1, keepdims=True)
    q = q - q.mean(axis=1, keepdims=True)
    num = np.einsum('tnp,tn->tp', th, q)
    den = np.sqrt((th ** 2).sum(axis=1) * (q ** 2).sum(axis=1)[:, None])
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(den > 0, num / den, np.nan)
    return np.nanmedian(np.abs(r), axis=0)


def _savefig(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=150)
    plt.close(fig)


def _load_npz_resilient(path, retries=6, initial_delay=2.0):
    """np.load with exponential backoff, returning a plain dict.

    On macOS a project under ~/Documents with "Desktop & Documents Folders"
    sync enabled is served through fileproviderd, not the local disk directly.
    Right after a multi-chain run writes hundreds of MB of .npz, a read can
    outrun the sync daemon and fail with TimeoutError [Errno 60] before a
    single byte arrives. Retrying rides that out.

    Materialising into a dict also closes the underlying zip handle, instead of
    leaving one open per chain for the caller's whole loop.

    The durable fix is to keep regenerable run output out of the synced tree:
        xattr -w 'com.apple.fileprovider.ignore#P' 1 <run output dir>
    """
    delay = initial_delay
    for attempt in range(1, retries + 1):
        try:
            with np.load(path, allow_pickle=True) as d:
                return {k: d[k] for k in d.files}
        except (TimeoutError, OSError) as exc:
            if attempt == retries:
                raise OSError(
                    f"Could not read {path} after {retries} attempts ({exc}). "
                    "If this path is inside an iCloud-synced folder, exclude the "
                    "run output directory from sync and rerun the pooling."
                ) from exc
            print(f"  [pool] {os.path.basename(path)} unreadable ({exc}); "
                  f"retry {attempt}/{retries - 1} in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2


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


def plot_pooled_chains(results, out_dir, observed=None, pooled_theta=None):
    """Three diagnostic figures from a pool_chain_results dict."""
    x = pd.to_datetime(results["dates"], errors="coerce")
    if pd.isna(x).any():
        x = np.arange(results["n_dates"])
    pct, cm = results["pooled_percentiles"], results["chain_means"]
    lo, hi = min(pct), max(pct)

    # 1 — pooled hydrograph with per-chain means overlaid
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(x, pct[lo], pct[hi], alpha=0.25, color='steelblue', label=f'{lo}–{hi}% pooled')
    if 25 in pct and 75 in pct:
        ax.fill_between(x, pct[25], pct[75], alpha=0.35, color='steelblue', label='25–75% pooled')
    for i, c in enumerate(cm):
        ax.plot(x, c, lw=0.7, alpha=0.6, color='grey', label='individual chain means' if i == 0 else None)
    ax.plot(x, results["pooled_mean"], color='blue', lw=1.6, label='pooled mean')
    if observed is not None:
        ax.plot(x, observed, color='orange', lw=1.6, label='observed')
    ax.set_xlabel('Date'); ax.set_ylabel('Q [m³/s]')
    ax.set_title(f'Pooled {results["n_chains"]} chains × {results["n_particles_per_chain"]} particles')

    # Clip the y-axis and shade the warm-up, matching the single-chain streamflow
    # plot. The first few timesteps carry the prior's spread, which is orders of
    # magnitude wider than anything afterwards and otherwise flattens the whole
    # series. Scale to the observations when available, else to the pooled mean.
    spinup_steps = 30
    if observed is not None and np.any(np.isfinite(observed)):
        y_max = float(np.nanmax(observed))
    else:
        y_max = float(np.nanmax(results["pooled_mean"]))
    if np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(0, y_max * 1.4)
    if len(x) > spinup_steps:
        ax.axvspan(x[0], x[spinup_steps - 1], color='grey', alpha=0.12, lw=0, zorder=0)
        ax.annotate('Warm-up', xy=(x[spinup_steps // 2], ax.get_ylim()[1]),
                    xytext=(0, -10), textcoords='offset points',
                    ha='center', va='top', fontsize=8, color='grey')

    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _savefig(fig, out_dir, "pooled_streamflow.png")

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


def pool_chain_results(chain_dirs, observed=None, output_dir=None,
                       percentiles=(5, 25, 50, 75, 95), make_plots=True):
    """Pool several independent particle-filter chains into one ensemble.

    Each chain is an independent Monte Carlo estimate of the same posterior,
    differing only in its random seed (initial parameter/state draws, resampling
    offsets, perturbations). Running several and pooling them reduces the
    influence of any single unlucky initial sample.
    IMPORTANT — pool the particles, do not average the quantiles.
    The correct combination is to concatenate the per-chain particle ensembles at
    each timestep and take statistics of the pooled ensemble.

    Reads <chain_dir>/posterior_parameter_samples.npz, written by main_routine
    when save_posterior_parameter_samples=True.

    Besides pooled_chains.npz (diagnostics only - pooled_mean, chain_means,
    between/within variance, percentiles), this also writes a pooled
    posterior_parameter_samples.npz (theta, qoi, theta_used, state_used,
    state_names, dates, param_names, param_lower, param_upper - the same
    schema a single chain's own file has) so the pooled result is a drop-in
    `working_dir` for downstream PCE code
    (offline_parameter_transform_and_pce_learning.py, designed_sample_pce.py)
    without those needing to know pooling happened. The raw particle arrays
    live only there, not duplicated into pooled_chains.npz - at 10+ chains
    that would double a multi-hundred-MB file for no reason, since nothing
    else reads pooled_theta/pooled_q back off disk (both are passed to
    plot_pooled_chains in-memory below, not reloaded).

    Requires each chain's posterior_parameter_samples.npz to carry theta_used/
    state_used (written by main_routine since the theta/qoi alignment fix) -
    theta/qoi alone are NOT a valid (parameter, output) pair (theta[k] is
    resampled+jittered for use at k+1; qoi[k] came from the ensemble that
    entered step k). Re-run older chains to pool them.

    resample_indices is intentionally NOT pooled: it records within-chain
    resampling survivors and has no clean meaning once chains are concatenated
    along the particle axis. Read it from each chain's own file if needed.

    Args:
        chain_dirs:  iterable of chain output directories.
        observed:    optional (n_dates,) observed series for P-factor/RMSE. When
                     None these are skipped.
        output_dir:  where to write pooled_chains.npz / .json; defaults to the
                     parent of the first chain directory.
        percentiles: percentiles to compute on the pooled ensemble.

    Returns:
        dict with pooled statistics, per-chain means, and the between- vs
        within-chain spread diagnostic (see below).

    Raises:
        FileNotFoundError / ValueError if a chain is missing or inconsistent.
    """
    chain_dirs = [str(c) for c in chain_dirs]
    if len(chain_dirs) < 2:
        raise ValueError(f"Need at least 2 chains to pool, got {len(chain_dirs)}.")

    qois, thetas, theta_useds, state_useds = [], [], [], []
    dates_ref, names_ref, state_names_ref = None, None, None
    lower_ref, upper_ref = None, None
    for cd in chain_dirs:
        f = os.path.join(cd, "posterior_parameter_samples.npz")
        if not os.path.isfile(f):
            raise FileNotFoundError(
                f"{f} not found. Run the chain with save_posterior_parameter_samples=True.")
        # d = np.load(f, allow_pickle=True)
        d = _load_npz_resilient(f)
        if "theta_used" not in d or "state_used" not in d:
            raise ValueError(
                f"{cd}: posterior_parameter_samples.npz has no theta_used/state_used "
                f"(written by an older main_routine, before the theta/qoi alignment "
                f"fix). Re-run this chain to pool it.")
        dates = [str(x) for x in d["dates"]]
        names = [str(x) for x in d["param_names"]]
        state_names = [str(x) for x in d["state_names"]]
        lower, upper = np.asarray(d["param_lower"]), np.asarray(d["param_upper"])
        if dates_ref is None:
            dates_ref, names_ref, lower_ref, upper_ref = dates, names, lower, upper
            state_names_ref = state_names
        else:
            if dates != dates_ref:
                raise ValueError(f"{cd}: date axis differs from the first chain "
                                 f"({len(dates)} vs {len(dates_ref)} steps).")
            if names != names_ref:
                raise ValueError(f"{cd}: parameter names differ from the first chain.")
            if not (np.array_equal(lower, lower_ref) and np.array_equal(upper, upper_ref)):
                raise ValueError(f"{cd}: param_lower/param_upper differ from the first chain.")
            if state_names != state_names_ref:
                raise ValueError(f"{cd}: state_names differ from the first chain.")
        qois.append(np.asarray(d["qoi"], dtype=float))       # (n_dates, n_particles)
        # Keep theta in whatever precision it was saved with. Forcing float64
        # here would silently undo save_theta_float32=True and double the
        # pooled footprint, which matters most at the chain counts that need
        # pooling in the first place.
        thetas.append(np.asarray(d["theta"]))                # (n_dates, n_particles, n_params)
        theta_useds.append(np.asarray(d["theta_used"]))       # (n_dates, n_particles, n_params)
        state_useds.append(np.asarray(d["state_used"]))       # (n_dates, n_particles, n_states)

    n_chains = len(qois)
    # Pool along the particle axis: (n_dates, n_chains * n_particles)
    pooled_q = np.concatenate(qois, axis=1)
    pooled_theta = np.concatenate(thetas, axis=1)
    # theta_used/state_used are concatenated the same way as qoi, so per-particle
    # alignment (theta_used[k, j] produced qoi[k, j]) survives pooling.
    pooled_theta_used = np.concatenate(theta_useds, axis=1)
    pooled_state_used = np.concatenate(state_useds, axis=1)

    # Computed on the correctly-aligned (theta_used, qoi) pair — NOT (theta, qoi),
    # which are one resampling-and-jitter step apart from each other within each
    # chain (see main_routine). ~0 means the parameter does not move the output.
    param_output_corr = parameter_output_correlation(pooled_theta_used, pooled_q)
    print("median |corr(parameter, Q)| across pooled particles, on the ALIGNED "
          "(theta_used, qoi) pair (<0.1 = filter cannot learn this parameter):")
    for nm, c in zip(names_ref, param_output_corr):
        print(f"    {nm:<8}{c:6.3f}")
    print(f"    {'overall':<8}{np.nanmedian(param_output_corr):6.3f}")

    pooled_mean = pooled_q.mean(axis=1)
    pooled_pcts = {int(p): np.percentile(pooled_q, p, axis=1) for p in percentiles}
    chain_means = np.stack([q.mean(axis=1) for q in qois], axis=0)  # (n_chains, n_dates)

    # Between- vs within-chain spread — the diagnostic for "did the initial
    # sample still matter?". B is the variance of the chain means; W is the
    # average within-chain variance.
    #
    # The reference is 1/n_particles, NOT 1. Chains drawing independently from
    # the SAME posterior still have means that scatter by Monte Carlo error, with
    # variance W/N, so E[B/W] = 1/N even when they agree perfectly. Comparing
    # B/W against 1 is far too lenient: a ratio of 0.03 looks tiny next to 1 but
    # is ~60x the floor at N=2000, i.e. the chains genuinely disagree.
    #   B/W ~ 1/N        -> indistinguishable from sampling error
    #   B/W >> 1/N       -> the initial ensemble still leaves an imprint
    # A complementary, threshold-free reading is B/(W+B): the fraction of the
    # pooled variance contributed by chain-to-chain disagreement.
    B = chain_means.var(axis=0, ddof=1) if n_chains > 1 else np.zeros_like(pooled_mean)
    W = np.mean([q.var(axis=1, ddof=1) for q in qois], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(W > 0, B / W, np.nan)

    results = {
        "n_chains": n_chains,
        "n_particles_per_chain": int(qois[0].shape[1]),
        "n_particles_pooled": int(pooled_q.shape[1]),
        "n_dates": len(dates_ref),
        "dates": dates_ref,
        "param_names": names_ref,
        "pooled_mean": pooled_mean,
        "pooled_percentiles": pooled_pcts,
        "chain_means": chain_means,
        "between_chain_var": B,
        "within_chain_var": W,
        "between_over_within_median": float(np.nanmedian(ratio)),
        "max_chain_mean_spread": float(np.max(chain_means.max(axis=0) -
                                              chain_means.min(axis=0))),
        "param_output_corr": {nm: float(c) for nm, c in zip(names_ref, param_output_corr)},
    }

    if observed is not None:
        obs = np.asarray(observed, dtype=float)
        if obs.shape[0] != pooled_mean.shape[0]:
            raise ValueError(f"observed has {obs.shape[0]} steps, chains have "
                             f"{pooled_mean.shape[0]}.")
        lo, hi = pooled_pcts[min(percentiles)], pooled_pcts[max(percentiles)]
        results["p_factor_pooled"] = float(np.mean((obs >= lo) & (obs <= hi)))
        results["rmse_pooled_mean"] = float(np.sqrt(np.mean((pooled_mean - obs) ** 2)))
        results["rmse_per_chain"] = [
            float(np.sqrt(np.mean((cm - obs) ** 2))) for cm in chain_means]

    out = output_dir or os.path.dirname(os.path.abspath(chain_dirs[0]))
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(
        os.path.join(out, "pooled_chains.npz"),
        pooled_mean=pooled_mean, chain_means=chain_means,
        between_chain_var=B, within_chain_var=W,
        dates=np.array(dates_ref, dtype=object),
        param_names=np.array(names_ref, dtype=object),
        **{f"pct_{p}": v for p, v in pooled_pcts.items()})

    # Pooled posterior in the same schema a single chain's own file has, so a
    # PCE step downstream can treat `out` as an ordinary working_dir.
    np.savez_compressed(
        os.path.join(out, "posterior_parameter_samples.npz"),
        theta=pooled_theta, qoi=pooled_q,
        theta_used=pooled_theta_used, state_used=pooled_state_used,
        state_names=np.array(state_names_ref, dtype=object),
        dates=np.array(dates_ref, dtype=object),
        param_names=np.array(names_ref, dtype=object),
        param_lower=lower_ref, param_upper=upper_ref)

    # observed_streamflow is identical across chains (same forcing/measured
    # data regardless of seed); copy rather than recompute so
    # plot_pce_after_particle_filter's observed overlay works unmodified when
    # pointed at the pooled directory. predicted_streamflow in the copy is
    # that one chain's own mean, not the pooled mean - not used by that plot,
    # but worth knowing if read directly.
    _src_avg = os.path.join(chain_dirs[0], "averaged_and_simulated.pkl")
    if os.path.isfile(_src_avg):
        shutil.copy2(_src_avg, os.path.join(out, "averaged_and_simulated.pkl"))

    save_run_configuration(
        out, {k: v for k, v in results.items()
              if k not in ("pooled_mean", "pooled_percentiles", "chain_means",
                           "between_chain_var", "within_chain_var", "dates")},
        filename="pooled_chains_summary.json")

    print(f"Pooled {n_chains} chains x {results['n_particles_per_chain']} particles "
          f"= {results['n_particles_pooled']} over {results['n_dates']} timesteps")
    _bw = results["between_over_within_median"]
    _floor = 1.0 / results["n_particles_per_chain"]
    print(f"  between/within chain variance (median): {_bw:.5f}")
    print(f"    Monte Carlo floor 1/N = {_floor:.5f}  ->  observed is {_bw/_floor:.0f}x the floor")
    _verdict = ("chains differ BEYOND sampling error" if _bw > 3 * _floor
                else "consistent with sampling error")
    print(f"    {_verdict}; {_bw/(1+_bw):.1%} of the pooled variance is between-chain")
    if observed is not None:
        print(f"  pooled RMSE={results['rmse_pooled_mean']:.2f}  "
              f"per-chain RMSE={['%.2f' % r for r in results['rmse_per_chain']]}")
        print(f"  pooled P-factor={results['p_factor_pooled']:.3f}")
    if make_plots:
        plot_pooled_chains(results, out, observed=observed, pooled_theta=pooled_theta)
        print(f"  wrote pooled_streamflow.png, chain_agreement.png, "
              f"pooled_parameter_posteriors.png -> {out}")
    return results


def _current_git_commit():
    """Short git SHA of the working tree, or None outside a repo / on error.

    Recorded alongside the configuration so a result folder can be tied back to
    the exact code that produced it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True, text=True, timeout=5)
            if dirty.returncode == 0 and dirty.stdout.strip():
                sha += "-dirty"
            return sha
    except Exception:
        pass
    return None


def _json_safe(obj):
    """Best-effort conversion of a value into something json.dump can write.

    Handles the types that actually turn up in the pipeline's arguments and
    results: pathlib paths, numpy scalars/arrays, pandas timestamps, and dicts
    whose keys are ints (e.g. monthly_bias_ar keyed by calendar month). Anything
    still unknown falls back to repr(), so saving configuration can never be the
    thing that crashes a completed run.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        return str(obj)
    except Exception:  # pragma: no cover - defensive
        return repr(obj)


def save_run_configuration(target_dir, config, filename="run_configuration.json",
                           extra=None):
    """Write the run's configuration to <target_dir>/<filename> as JSON.

    Written early in the run so it survives a crash, and so any later analysis of
    the output folder can recover exactly which settings produced it (likelihood
    mode, phi_ar, beta_obs, sigma_eta, perturbation options, priors, ...).

    Args:
        target_dir: directory to write into.
        config:     dict of configuration values (non-JSON types are coerced).
        filename:   output file name.
        extra:      optional dict merged in under a separate key.

    Returns:
        The path written, or None if writing failed (never raises).
    """
    payload = {"saved_at": pd.Timestamp.now().isoformat(),
               "config": _json_safe(config)}
    if extra:
        payload["environment"] = _json_safe(extra)
    path = os.path.abspath(os.path.join(str(target_dir), filename))
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Saved run configuration -> {path}")
        return path
    except Exception as e:
        print(f"WARNING: could not write {filename}: {e}")
        return None

####################

def run_model_single_time_stamp_single_particle(hbvsaskModelObject, date_of_interest,
                                    parameter_value_dict, state_values_dict, unique_index_model_run = 0):
    """
    Runs the HBV-SASK model for a single time stamp and a single particle.

    Args:
        hbvsaskModelObject (object): An instance of the HBV-SASK model.
        date_of_interest (str): The date of interest in the format 'YYYY-MM-DD'.
        parameter_value_dict (dict): A dictionary containing the parameter values for the model.
        state_values_dict (dict): A dictionary containing the initial state values for the model.
        unique_index_model_run (int, optional): An optional unique index for the model run. Defaults to 0.

    Returns:
        tuple: A tuple containing the following elements:
            - unique_index_model_run (int): The unique index for the model run.
            - y_t_model (float): The model output for the given time stamp.
            - y_t_observed (float): The observed output for the given time stamp.
            - x_t_plus_1 (dict): The model states for the next time stamp.
            - parameter_value_dict (dict): The parameter values used for the model run.
    """
    assert date_of_interest >= hbvsaskModelObject.start_date_predictions, \
        "Sorry, the date of interest is before start_date_predictions"
    assert date_of_interest <= hbvsaskModelObject.end_date, "Sorry, the date of interest is after end_date"

    # print(f"The current day - {date_of_interest}")
    date_of_interest_plus_one = pd.to_datetime(date_of_interest) + pd.DateOffset(days=1)

    forcing = hbvsaskModelObject.time_series_measured_data_df.loc[[date_of_interest, ], :].copy()

    state_values_dict["WatershedArea_km2"] = float(
        hbvsaskModelObject.initial_condition_df["WatershedArea_km2"].values[0])
    state_values_dict[hbvsaskModelObject.time_column_name] = date_of_interest
    
    initial_condition_df = pd.DataFrame(state_values_dict, index=[0])

    results_array_changed_param = hbvsaskModelObject.run(
        i_s=[unique_index_model_run,],
        parameters=[parameter_value_dict, ],
        createNewFolder=False,
        take_direct_value=True,
        forcing=forcing,
        initial_condition_df=initial_condition_df
    )

    # extract y_t produced by the model
    model_output_dict = results_array_changed_param[0][0]['result_time_series'].loc[date_of_interest].to_dict()
    y_t_model = model_output_dict["Q_cms"]

    # extract x_(t+1) model states for the next time-stamp
    state_dict = results_array_changed_param[0][0]['state_df'].loc[date_of_interest_plus_one].to_dict()
    x_t_plus_1 = state_dict

    # extract y_t_observed
    if hbvsaskModelObject.read_measured_streamflow:
        measured_output_date = forcing["streamflow"].values[0]
    else:
        measured_output_date = None
    y_t_observed = measured_output_date

    # return model_output_dict, state_dict, measured_output_date
    return unique_index_model_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict

####################

def estimate_monthly_bias(df, simulated_column, observed_column,
                          time_column=None, calibration_end=None,
                          notebook_convention=False, min_days_per_month=5):
    """Estimate the per-calendar-month mean residual for the AR likelihood.

    Returns a dict {1..12: float} in the PIPELINE sign convention
    ε = y_observed − y_simulated, i.e. a POSITIVE value means the model
    underestimates in that month. This is the form main_routine expects for its
    monthly_bias_ar argument.

    WARNING — data leakage. The returned biases are estimated from observations.
    If they are estimated over the same period the filter is later evaluated on,
    the forecast has indirectly seen its own verification data and the skill
    scores are inflated. Pass `calibration_end` to restrict the estimate to a
    calibration window that ends before the forecast period begins.

    Args:
        df:                DataFrame holding simulated and observed series.
        simulated_column:  column with the simulated/ensemble-mean discharge.
        observed_column:   column with the observed discharge.
        time_column:       column holding timestamps; if None, the DataFrame
                           index is used (it must be a DatetimeIndex).
        calibration_end:   optional timestamp; rows at or after it are dropped
                           before estimating. Strongly recommended.
        notebook_convention: set True when `simulated_column` − `observed_column`
                           is already stored the other way round (ε = sim − obs,
                           as in HBV_SASK_Error_Analysis.ipynb). The sign is then
                           flipped so the result is always in pipeline convention.
        min_days_per_month: months with fewer valid days than this are omitted
                           (main_routine falls back to 0.0 for missing months).

    Returns:
        dict {month:int -> bias:float}
    """
    work = df.copy()
    times = pd.to_datetime(work[time_column]) if time_column is not None \
        else pd.to_datetime(work.index.to_series())

    if calibration_end is not None:
        mask = times < pd.to_datetime(calibration_end)
        work, times = work[mask], times[mask]
        if len(work) == 0:
            raise ValueError(f"No rows before calibration_end={calibration_end}.")
    else:
        print("WARNING estimate_monthly_bias: no calibration_end given — biases are "
              "estimated over the whole record. If the filter is evaluated on any of "
              "this period, forecast skill will be optimistically biased.")

    residual = work[observed_column].astype(float) - work[simulated_column].astype(float)
    if notebook_convention:
        residual = -residual

    grouped = residual.groupby(times.dt.month)
    counts = grouped.count()
    means = grouped.mean()

    bias = {int(m): float(means[m]) for m in means.index
            if counts[m] >= min_days_per_month}
    dropped = sorted(set(range(1, 13)) - set(bias))
    if dropped:
        print(f"estimate_monthly_bias: months {dropped} omitted "
              f"(fewer than {min_days_per_month} valid days); they default to 0.0.")
    return bias

####################

def build_marginal_prior(spec):
    """Build a chaospy marginal prior from one configuration entry.

    Supported "distribution" values:

    "Uniform"      — U(lower, upper). Uninformative within the bounds.

    "TruncNormal"  — Normal truncated at [lower, upper]. Follows the convention of
                     Nagel, Rieckermann & Sudret (2020, RESS 195:106737): unless the
                     entry gives explicit "mu"/"sigma", the distribution is centred on
                     the midpoint of the range and its standard deviation is set to one
                     sixth of the range, so the bounds sit at roughly ±3σ. Encodes
                     "the expert believes the midpoint; the bounds are hard limits".

    Args:
        spec: dict with at least "name", "distribution", "lower", "upper";
              optionally "mu" and "sigma" for TruncNormal.

    Returns:
        (name, marginal) where marginal is a chaospy distribution.
    """
    name = spec["name"]
    dist_type = spec["distribution"]
    lower, upper = float(spec["lower"]), float(spec["upper"])
    if upper <= lower:
        raise ValueError(f"{name}: upper ({upper}) must exceed lower ({lower}).")

    if dist_type == "Uniform":
        return name, cp.Uniform(lower, upper)

    if dist_type == "TruncNormal":
        mu = float(spec.get("mu", 0.5 * (lower + upper)))
        sigma = float(spec.get("sigma", (upper - lower) / 6.0))
        if sigma <= 0:
            raise ValueError(f"{name}: sigma must be positive, got {sigma}.")
        return name, cp.TruncNormal(lower=lower, upper=upper, mu=mu, sigma=sigma)

    raise NotImplementedError(
        f"{name}: distribution '{dist_type}' is not supported. "
        f"Use 'Uniform' or 'TruncNormal'.")


def calculate_likelihood(y_t_observed, y_t_model, error_variance):
    """
    Computing Gaussian like likelihood
    """
    if y_t_observed is not None and y_t_model is not None:
        exponent = -0.5 * ((y_t_observed - y_t_model) ** 2) / error_variance
        likelihood = np.exp(exponent) / np.sqrt(2 * np.pi * error_variance)
        return likelihood
    else:
        return 0


def calculate_likelihood_heteroscedastic(y_t_observed, y_t_model, beta_obs=0.2 / 3,
                                         sigma_eps=None):
    """Heteroscedastic Gaussian likelihood WITHOUT an AR structural-error term.

    This is the non-AR arm of the comparison (Option A). The likelihood is centred
    directly on the model output Q_i, which is also what gets reported, so the
    filter is pushed to make Q_i itself match the observation.

    Crucially the FULL error std is used here:

        σ_ε(t) = beta_obs · |y_obs(t)|

    and NOT the AR innovation std σ_η = σ_ε·√(1−φ²). That reduction is only
    justified when the predictive mean has been shifted by ε̂ = φ·ε(t−1); with no
    AR term the residual is ε(t) itself, whose spread is σ_ε. Using σ_η here would
    make the likelihood spuriously sharp and collapse the particle weights.

    Args:
        y_t_observed: scalar observed discharge (or None).
        y_t_model:    scalar model discharge for this particle (or None).
        beta_obs:     proportionality constant, σ_ε = beta_obs·|y_obs|.
                      Default 0.2/3 (20% relative error read as a 3σ bound).
        sigma_eps:    optional fixed σ_ε; overrides the heteroscedastic form
                      (homoscedastic mode, for a like-for-like comparison against
                      calculate_likelihood_ar with a fixed sigma_eta).
    """
    if y_t_observed is None or y_t_model is None:
        return 0.0
    if sigma_eps is None:
        sigma_eps = beta_obs * abs(y_t_observed)
        # sigma_eps = beta_obs * np.sqrt(np.abs(y_t_observed))
    sigma_eps = max(sigma_eps, 1e-6)  # floor against zero flow
    residual = y_t_observed - y_t_model
    exponent = -0.5 * (residual ** 2) / sigma_eps ** 2
    return np.exp(exponent) / np.sqrt(2 * np.pi * sigma_eps ** 2)


def calculate_likelihood_ar(y_t_observed, y_t_model, epsilon_hat, sigma_eta=None,
                            phi_ar=None, beta_obs=0.2 / 3):
    """Gaussian likelihood with AR(1)-predicted structural error.

    Sign convention: epsilon = y_observed − y_model  (positive = model underestimates).
    epsilon_hat = φ·ε(t−1) + μ_m  where μ_m is any seasonal bias pre-added by the
    caller (see main_routine, monthly_bias_ar parameter).

    Supports two modes for the innovation std σ_η:

    1. Fixed (homoscedastic): pass sigma_eta as a scalar constant.

    2. Heteroscedastic (Pianosi 2016): pass phi_ar and omit sigma_eta.
       σ_η(t) = beta_obs · √(1−φ²) · |y_obs(t)|
       Derivation: raw error σ_ε = beta_obs·y_obs (20% relative error treated
       as 3σ bound → beta_obs = 0.2/3); AR(1) shrinks variance by (1−φ²),
       so σ_η = σ_ε · √(1−φ²).  This makes the likelihood sharper at low
       flows and appropriately wider at high flows.

    Args:
        y_t_observed: scalar observed discharge (or None).
        y_t_model:    scalar model discharge for this particle (or None).
        epsilon_hat:  AR(1) prediction of structural error, φ·ε(t−1) [+ optional
                      monthly bias]; computed in the main loop before calling this.
        sigma_eta:    fixed innovation std [same units as Q]. If None,
                      heteroscedastic mode is used (requires phi_ar).
        phi_ar:       AR(1) coefficient, used only in heteroscedastic mode.
        beta_obs:     proportionality constant for σ_ε = beta_obs·|y_obs|.
                      Default 0.2/3 (20% error as 3σ bound).
    """
    if y_t_observed is None or y_t_model is None:
        return 0.0
    if sigma_eta is None:
        if phi_ar is None:
            raise ValueError("Provide either sigma_eta or phi_ar.")
        sigma_eta = beta_obs * np.sqrt(1.0 - phi_ar ** 2) * abs(y_t_observed)
        # sigma_eta = beta_obs * np.sqrt(1.0 - phi_ar ** 2) * np.sqrt(abs(y_t_observed))
        sigma_eta = max(sigma_eta, 1e-6)  # floor against zero flow
    y_hat = y_t_model + epsilon_hat
    innovation = y_t_observed - y_hat   # η(t) = ε(t) − ε̂(t), should be ≈ N(0, σ_η²)
    exponent = -0.5 * (innovation ** 2) / sigma_eta ** 2
    return np.exp(exponent) / np.sqrt(2 * np.pi * sigma_eta ** 2)


def calculate_likelihood_ar_student_t(y_t_observed, y_t_model, epsilon_hat,
                                      sigma_eta=None, phi_ar=None,
                                      beta_obs=0.2 / 3, df=5.0):
    """Student-t likelihood with AR(1)-predicted structural error.

    Identical structure to calculate_likelihood_ar but uses a Student-t
    distribution for the innovation η(t), which assigns heavier probability
    to large residuals during flood events, reducing weight collapse.

    A lower degrees-of-freedom (df) gives heavier tails:
      df → ∞ : reduces to Gaussian
      df = 5 : moderately heavy tails (recommended starting point)
      df = 3 : very heavy tails

    Args:
        y_t_observed: scalar observed discharge (or None).
        y_t_model:    scalar model discharge for this particle (or None).
        epsilon_hat:  AR(1) prediction φ·ε(t−1) [+ optional monthly bias].
        sigma_eta:    fixed innovation scale [same units as Q]. If None,
                      heteroscedastic mode is used (requires phi_ar).
        phi_ar:       AR(1) coefficient, used only in heteroscedastic mode.
        beta_obs:     proportionality constant (same as Gaussian mode).
        df:           degrees of freedom for the Student-t distribution (> 0).
    """
    if y_t_observed is None or y_t_model is None:
        return 0.0
    if sigma_eta is None:
        if phi_ar is None:
            raise ValueError("Provide either sigma_eta or phi_ar.")
        sigma_eta = beta_obs * np.sqrt(1.0 - phi_ar ** 2) * abs(y_t_observed)
        sigma_eta = max(sigma_eta, 1e-6)
    y_hat = y_t_model + epsilon_hat
    # student_t.pdf(x, df, loc, scale) evaluates the scaled t-distribution
    return float(student_t.pdf(y_t_observed, df=df, loc=y_hat, scale=sigma_eta))


def systematic_resample(weights):
    """
    Mapping samples i to the new samples j all with the same weights 1/N_p
    """
    N_p = len(weights)
    # positions[j] = (j + U) / N,  U ~ Uniform(0,1),  j = 0,...,N-1 gives N evenly-spaced pointers with a single shared random offset
    positions = (np.arange(N_p) + np.random.random()) / N_p  # initialize positions from uniform distribution
    cumulative_sum = np.cumsum(weights)  # CDF of particles
    cumulative_sum[-1] = 1.0          # force exact 1.0 to avoid float drift
    indices = np.zeros(N_p, dtype=int)
    i, j = 0, 0
    # while i < N_p:
    #     if cumulative_sum[j] > positions[i]:
    #         indices[i] = j
    #         i += 1
    #     else:
    #         j += 1
    while j < N_p:
        if cumulative_sum[i] > positions[j]:  # particle i's CDF bin covers pointer j
            indices[j] = i # → select particle i
            # indices[i] = j
            j += 1
        else:
            i += 1
    return indices


def perturb_parameters(parameters, param_stds=None, perturbation_factor=0.15,
                       param_bounds=None, min_jitter_frac=0.002,
                       bound_handling="clip", perturbation_scheme="magnitude",
                       ensemble_mean=None, liu_west_delta=0.98):
    """Jitter resampled parameters, keeping them inside their prior support.

    Two perturbation schemes:

    1. "magnitude" (default). Independent Gaussian jitter around each
       particle's OWN current value, σ = η·|θ_i|. η (`perturbation_factor`) may
       be a single float shared across all parameters, or a dict {name: η} for
       parameter-specific jitter scales — e.g. a smaller η for a slow,
       memory-dependent parameter (K2) that needs to survive many days between
       informative events, and a larger one for a fast parameter (TT) that can
       afford to be reshuffled more often. Scale-dependent in both directions:
       large θ gets large jitter, which feeds back into still larger θ; small θ
       gets small jitter, which is why the jitter floor below exists.

    2. "liu_west" (Liu and West, 2001). Each particle is shrunk toward the
       CURRENT ensemble mean rather than jittered around its own value:

           a = (3·δ − 1) / (2·δ)            h² = 1 − a²
           m_i = a·θ_i + (1 − a)·ensemble_mean[name]
           θ_i_new ~ N(m_i, h²·ensemble_var[name])

       By construction Var[θ_new] = a²·Var[θ] + h²·Var[θ] = Var[θ]: the
       pre-jitter ensemble variance is preserved EXACTLY, so this scheme can
       neither runaway-collapse (as σ = η·S(θ) does: narrower ensemble →
       smaller jitter → narrower still) nor inflate over time. δ ∈ (0.95, 0.99)
       is a discount factor; closer to 1 is gentler shrinkage, closer to 0.95
       pulls particles toward the mean more aggressively. Needs `param_stds`
       (the CURRENT ensemble std per parameter) and `ensemble_mean` (same, but
       the mean) — both computed ONCE PER DATE across the whole ensemble by the
       caller, not per particle; this function only perturbs one particle.

    Two safeguards, both essential over long runs, and shared by both schemes:

    1. BOUNDS. Without clipping, repeated jitter is an unbounded random walk and
       parameters drift far outside the range declared in the configuration
       (observed: C0 reaching ~350 against bounds [0, 10]). `param_bounds` maps
       parameter name -> (lower, upper); values are clipped after perturbing.

    2. JITTER FLOOR. The "magnitude" scale σ = η·|θ| is proportional to the
       value itself, which makes θ = 0 an ABSORBING state: once a particle
       reaches zero its jitter is exactly zero and it can never move again.
       Over a long run the ensemble piles up at zero. The floor
       min_jitter_frac·(upper−lower) keeps a small range-relative jitter alive
       so zero stays escapable. The same floor rescues "liu_west" when the
       ensemble spread has collapsed after resampling (std -> 0 would
       otherwise freeze every particle at the shrunk mean, permanently).

    Args:
        parameters:         dict {name: value} for one particle.
        param_stds:         dict {name: ensemble std}. Required for
                            "liu_west"; unused by "magnitude".
        perturbation_factor: η for "magnitude" — a float, or a dict {name: η}
                            for parameter-specific jitter scales. Unused by
                            "liu_west" (its scale is set entirely by δ and the
                            current ensemble variance, not by η).
        param_bounds:       optional dict {name: (lower, upper)} enforcing support.
        min_jitter_frac:    jitter floor as a fraction of (upper−lower). Only
                            applied for parameters present in param_bounds.
        bound_handling:     how to return an out-of-bounds draw to the interval.
                            "clip"    — set it to the bound. Simple, but every
                                        overshooting particle lands on the SAME
                                        value, so probability mass piles up into
                                        an atom at the boundary
                            "reflect" — bounce it back inside. No atoms at any
                                        jitter scale, and it preserves more of the
                                        ensemble spread than clipping does.
                            Neither is a Bayesian update; both are ad-hoc ways to
                            respect a truncated prior. Reflection is the more
                            standard choice and distorts the density less.
        perturbation_scheme: "magnitude" or "liu_west", see above.
        ensemble_mean:      dict {name: ensemble mean}. Required for "liu_west".
        liu_west_delta:     δ, the Liu-West discount factor. Only used by
                            "liu_west".
    """
    if perturbation_scheme not in ("magnitude", "liu_west"):
        raise ValueError(f"perturbation_scheme must be 'magnitude' or 'liu_west', "
                         f"got {perturbation_scheme!r}.")
    if perturbation_scheme == "liu_west":
        if param_stds is None or ensemble_mean is None:
            raise ValueError("perturbation_scheme='liu_west' needs both param_stds "
                             "and ensemble_mean, computed once per date across the "
                             "whole ensemble.")
        a = (3.0 * liu_west_delta - 1.0) / (2.0 * liu_west_delta)
        h = np.sqrt(max(1.0 - a * a, 0.0))

    perturbed_parameters = {}
    for key, value in parameters.items():
        bounds = param_bounds.get(key) if param_bounds else None

        if perturbation_scheme == "liu_west":
            std = param_stds[key]
            if bounds is not None:
                lower, upper = bounds
                std = max(std, min_jitter_frac * (upper - lower))
            scale = h * std
            new_value = a * value + (1.0 - a) * ensemble_mean[key] + np.random.normal(0, scale)
        else:
            eta = perturbation_factor[key] if isinstance(perturbation_factor, dict) \
                else perturbation_factor
            # proportional to the magnitude of the parameter
            scale = eta * abs(value)
            if bounds is not None:
                lower, upper = bounds
                # Floor the jitter relative to the parameter's own range so that
                # θ→0 does not freeze the particle.
                scale = max(scale, min_jitter_frac * (upper - lower))
            new_value = value + np.random.normal(0, scale)

        if bounds is not None:
            span = upper - lower
            if span <= 0:
                new_value = lower
            elif bound_handling == "reflect":
                # Modulo 2*span folds arbitrarily large overshoots back inside,
                # so a single expression handles repeated reflections.
                t = (new_value - lower) % (2.0 * span)
                new_value = lower + (t if t <= span else 2.0 * span - t)
            elif bound_handling == "clip":
                new_value = min(max(new_value, lower), upper)
            else:
                raise ValueError(f"bound_handling must be 'clip' or 'reflect', "
                                 f"got {bound_handling!r}.")
        perturbed_parameters[key] = new_value
    return perturbed_parameters

####################


# Transport-map / parameter-transformation code lives in uqef_dynamic/utils:
#   mpart_transport.py       MParT triangular maps
#   gaussian_anamorphosis.py rank-based marginal transform
#   legacy_transport.py      the bundled transport_map.py toolbox
#   transport_timeseries.py  per-timestep driver + gaussianize_parameter_samples







####################

def main_routine(
                num_processes, number_of_particles,
                inputModelDir,
                configuration_file,
                workingDir="trial_single_run_hbvsaskmodel_7d_filtering",
                directory_for_saving_plots="trial_single_run_hbvsaskmodel_7d_filtering",
                 # ── Likelihood options ──────────────────────────────────────────────────
                 use_ar_likelihood=True,  # True → AR(1)-augmented; False → simple Gaussian
                 phi_ar=0.894,    # AR(1) coefficient
                 sigma_eta=None,  # fixed σ_η [m³/s]; None → heteroscedastic (Pianosi 2016)
                 beta_obs=0.2/3,  # σ_ε = beta_obs·|y_obs|; used when sigma_eta=None
                 # Monthly seasonal bias correction (PF sign convention: y_obs − y_model).
                 # Dict {1..12: float} of mean residuals per calendar month, estimated
                 # from an open-loop ensemble run (positive = model underestimates).
                 # None disables the correction (backward-compatible default).
                 monthly_bias_ar=None,
                 # Student-t innovation (heavier tails, reduces weight collapse at peaks).
                 use_student_t=False,  # False → Gaussian; True → Student-t innovation
                 student_t_df=5.0,     # degrees of freedom (lower = heavier tails)
                 # ── Reproducibility ─────────────────────────────────────────────────────
                 # Seeds the global numpy RNG, which drives ALL stochastic parts of the
                 # filter: the initial parameter/state draws (chaospy sampling), the
                 # systematic-resampling offset, and the parameter perturbation. Give
                 # each chain in a multi-chain study a DIFFERENT seed so the chains are
                 # independent, and record it so any chain can be reproduced exactly.
                 # None leaves the RNG untouched (non-reproducible).
                 random_seed=None,
                 # ── Parameter perturbation options ──────────────────────────────────────
                 # "magnitude" (default): sigma = eta*|theta_i| per particle, unchanged
                 # from before. eta = perturbation_factor, a float or {name: eta} dict for
                 # parameter-specific jitter scales.
                 # "liu_west": Liu and West (2001) shrinkage - each particle is pulled
                 # toward the current ensemble mean and perturbed with a scale set by
                 # liu_west_delta and the current ensemble variance, which preserves that
                 # variance exactly rather than letting it collapse or inflate. Replaces
                 # the old use_ensemble_std_perturbation=True (sigma ~ S(theta) around each
                 # particle's own value), which had no such guarantee and could collapse
                 # geometrically as the ensemble narrowed. See perturb_parameters' docstring.
                 perturbation_scheme="magnitude",
                 perturbation_factor=0.15,
                 liu_west_delta=0.98,
                 # Jitter floor, as a fraction of (upper-lower), applied to whichever
                 # scale perturb_parameters computed. Keeps a parameter that has
                 # reached 0 ("magnitude") or an ensemble whose spread has collapsed
                 # ("liu_west") from freezing permanently - see perturb_parameters'
                 # own docstring (JITTER FLOOR) for why this is necessary either way.
                 min_jitter_frac=0.002,
                 # How an out-of-bounds perturbed value is returned to its interval.
                 # "clip" pins it to the bound, which piles probability mass into an
                 # atom there (measured: ~16% of PM particles sat exactly on a bound)
                 # and stops a transport map from Gaussianizing the posterior.
                 # "reflect" bounces it back inside: no atoms at any jitter scale, and
                 # it preserves more ensemble spread. Default "clip" = existing behaviour.
                 bound_handling="clip",
                 # ── Predictive band options ─────────────────────────────────────────────
                 # Include the innovation noise η ~ N(0, σ_η²) when building the plotted
                 # percentile bands. Set False only to inspect the mean spread alone.
                 include_innovation_in_bands=False,
                 band_noise_seed=12345,   # fixed so replots are reproducible
                 # Which flow scales the heteroscedastic innovation sigma for the band:
                 #   "observed" — matches the likelihood exactly (Pianosi 2016), but the
                 #                band width then depends on y_obs(t), the value being
                 #                predicted, so it is a hindcast band.
                 #   "forecast" — scales by the ensemble-mean prediction instead, which
                 #                is known before y_obs(t); leak-free, use for forecast
                 #                skill claims.
                 # The raw-Q band plotted alongside never uses observations either way.
                 band_sigma_from="observed",
                 # ── Transport map / PCE support ─────────────────────────────────────────
                 # Persist the posterior parameter ensemble and matching model output at
                 # every timestep to <workingDir>/posterior_parameter_samples.npz. This is
                 # the raw material for building a per-timestep transport map and PCE.
                 # Size ≈ n_dates·n_particles·(n_params+1)·8 bytes.
                 save_posterior_parameter_samples=True,
                 # ── Output size control (for multi-chain runs) ───────────────────────────
                 light_output=False,
                 # Store the saved theta array as float32 instead of float64.
                 # Pairs naturally with light_output for multi-chain runs.
                 save_theta_float32=True,
                 # Backend used to map posterior parameter samples to a standard
                 # Gaussian space (the input a Hermite PCE needs):
                 #   "mpart"        → joint MParT triangular map; whitens the
                 #                    cross-correlations a marginal transform leaves
                 #   "anamorphosis" → rank-based marginal transform (Fan et al. 2016);
                 #                    much faster and robust to spiky marginals, but
                 #                    does not whiten
                 #   "legacy"       → the bundled transport_map.py toolbox
                 #   None           → skip entirely
                 transport_map_backend="legacy",
                 transport_map_max_order=2,
                 # When True, fit one map PER TIMESTEP after the filter finishes and
                 # write standard_parameter_samples.npz (z, theta, qoi per date) —
                 # the regression pairs for a per-timestep PCE. Done post-hoc from
                 # the saved samples, so it costs the filter loop nothing and can be
                 # re-run with a different backend/order without re-filtering.
                 # Requires save_posterior_parameter_samples=True.
                 map_all_timesteps=False,
                 map_all_timesteps_workers=None,   # None → os.cpu_count()
                 ):
    # Snapshot every declared argument of this call, before any local variable is
    # created. Done via introspection rather than a hand-written list so new
    # parameters are captured automatically and cannot silently drift out of sync.
    _argspec, _, _, _argvalues = inspect.getargvalues(inspect.currentframe())
    run_config = {name: _argvalues[name] for name in _argspec}

    # Seed before anything stochastic happens — the prior draws below are the first
    # consumer, and they are exactly what differs between chains.
    if random_seed is not None:
        np.random.seed(random_seed)
        print(f"RNG seeded with random_seed={random_seed}")

    # =========================================================
    # Model Related Setup
    # =========================================================

    with open(configuration_file) as _f:
        _cfg_json = json.load(_f)
    basin = _cfg_json.get("model_settings", {}).get("basin")
    if not basin:
        raise ValueError(
            f"model_settings.basin is missing from {configuration_file}; the pipeline "
            "no longer hardcodes a basin.")
    if not str(directory_for_saving_plots).endswith("/"):
        directory_for_saving_plots = str(directory_for_saving_plots) + "/"

    # Record the paths actually used (the block above overrides the arguments) plus
    # the resolved likelihood mode, so the saved config reflects the real run rather
    # than only what was passed in.
    run_config.update({
        "resolved_workingDir": workingDir,
        "resolved_inputModelDir": inputModelDir,
        "resolved_configuration_file": configuration_file,
        "resolved_basin": basin,
        "likelihood_mode": (
            ("AR(1) + Student-t" if use_student_t else "AR(1) + Gaussian")
            if use_ar_likelihood else "heteroscedastic Gaussian (no AR)"),
        "sigma_mode": "homoscedastic (fixed sigma_eta)" if sigma_eta is not None
                      else "heteroscedastic (beta_obs * |y_obs|)",
    })

    # Written now, before the expensive loop, so the settings survive a crash.
    save_run_configuration(
        directory_for_saving_plots, run_config,
        extra={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "chaospy": cp.__version__,
            "git_commit": _current_git_commit(),
            "script": os.path.abspath(__file__),
        })

    ne = number_of_particles

    # Creating Model Object
    writing_results_to_a_file = False
    plotting = False
    createNewFolder = False # create a separate folder to save results for each model run
    hbvsaskModelObject = hbvmodel.HBVSASKModel(
        configurationObject=configuration_file,
        inputModelDir=inputModelDir,
        workingDir=workingDir,
        basin=basin,
        writing_results_to_a_file=writing_results_to_a_file,
        plotting=plotting
    )

    # In case one wants to modify dates compared to those set up in the configuration object / deverge from these setting
    # start_date = '2006-03-30 00:00:00'
    # end_date = '2007-06-30 00:00:00'
    # spin_up_length = 365  # 365*3
    # start_date = pd.to_datetime(start_date)
    # end_date = pd.to_datetime(end_date)
    # # dict_with_dates_setup = {"start_date": start_date, "end_date": end_date, "spin_up_length":spin_up_length}
    # run_full_timespan = False
    # hbvsaskModelObject.set_run_full_timespan(run_full_timespan)
    # hbvsaskModelObject.set_start_date(start_date)
    # hbvsaskModelObject.set_end_date(end_date)
    # hbvsaskModelObject.set_spin_up_length(spin_up_length)
    # simulation_length = (hbvsaskModelObject.end_date - hbvsaskModelObject.start_date).days - hbvsaskModelObject.spin_up_length
    # if simulation_length <= 0:
    #     simulation_length = 365
    # hbvsaskModelObject.set_simulation_length(simulation_length)
    # hbvsaskModelObject.set_date_ranges()
    # hbvsaskModelObject.redo_input_and_measured_data_setup()

    # Get to know some of the relevant time settings, read from a json configuration file
    # print(f"start_date: {hbvsaskModelObject.start_date}")
    # print(f"start_date_predictions: {hbvsaskModelObject.start_date_predictions}")
    # print(f"end_date: {hbvsaskModelObject.end_date}")
    # print(f"full_data_range is {len(hbvsaskModelObject.full_data_range)} "
    #       f"hours including spin_up_length of {hbvsaskModelObject.spin_up_length} hours")
    # print(f"simulation_range is of length {len(hbvsaskModelObject.simulation_range)} hours")

    # Plot forcing data and observed streamflow
    hbvsaskModelObject.plot_input_data(read_measured_streamflow=True)

    list_of_dates_of_interest = list(pd.date_range(
        start=hbvsaskModelObject.start_date_predictions, end=hbvsaskModelObject.end_date, freq="1D"))
    # list_of_dates_of_interest = list(pd.date_range(
    #     start='2007-04-30 00:00:00', end='2007-05-30 00:00:00', freq="1D"))

    # =========================================================
    # Code for creating the initial parameter values and state values
    # =========================================================

    # read configuration file and save it as a configuration object
    # with open(configuration_file, 'rb') as f:
    #     configurationObject = dill.load(f)
    # simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)
    # on can as well just fetch the configurationObject from the hbvsaskModelObject
    configurationObject = hbvsaskModelObject.configurationObject

    # Sampling from the parameter space.
    # Marginal type is read per entry from the configuration ("Uniform" | "TruncNormal"),
    # see build_marginal_prior. The standard / [-1,1] companions stay uniform: they are
    # the reference spaces used by the transport map, not priors.
    list_of_single_dist = []
    list_of_single_standard_dist = []
    list_of_single_min_1_1_dist = []
    param_names = []
    for param in configurationObject["parameters"]:
        name, marginal = build_marginal_prior(param)
        param_names.append(name)
        list_of_single_dist.append(marginal)
        list_of_single_standard_dist.append(cp.Uniform(0, 1))
        list_of_single_min_1_1_dist.append(cp.Uniform(-1, 1))
    joint_params = cp.J(*list_of_single_dist)
    joint_standard_params  = cp.J(*list_of_single_standard_dist)
    joint_min_1_1_params = cp.J(*list_of_single_min_1_1_dist)
    print("Parameter priors: " + ", ".join(
        f"{p['name']}~{p['distribution']}[{p['lower']},{p['upper']}]"
        for p in configurationObject["parameters"]))

    # Bounds used to keep the perturbation step inside the prior support.
    # Without these the repeated jitter is an unbounded random walk (see
    # perturb_parameters) and parameters drift far outside their declared range.
    param_bounds = {
        p["name"]: (float(p["lower"]), float(p["upper"]))
        for p in configurationObject["parameters"]
    }

    # Sampling from the state space
    list_of_single_dist = []
    list_of_single_standard_dist = []
    list_of_single_min_1_1_dist = []
    state_names = []
    for state in configurationObject["states"]:
        name, marginal = build_marginal_prior(state)
        state_names.append(name)
        list_of_single_dist.append(marginal)
        list_of_single_standard_dist.append(cp.Uniform(0, 1))
        list_of_single_min_1_1_dist.append(cp.Uniform(-1, 1))
    joint_states = cp.J(*list_of_single_dist)
    joint_standard_states  = cp.J(*list_of_single_standard_dist)
    joint_min_1_1_states = cp.J(*list_of_single_min_1_1_dist)

    list_parameter_value_particles = joint_params.sample(number_of_particles, rule="random").T # rule can as well be: 'sobol' | 'random' | "latin_hypercube" | "halton"
    list_state_values_particles = joint_states.sample(number_of_particles, rule="random").T
    list_unique_index_model_run_list = list(range(0, len(list_state_values_particles)))

    # Plotting inital distribution of parameter values
    fig, axs = plt.subplots(1, len(param_names), figsize=(20, 10))
    fig_plotly = make_subplots(rows=1, cols=len(param_names))
    # dict_of_distriubtions_over_parameters_initial = defaultdict(list, {key:[] for key in param_names})
    for idx in range(len(param_names)):
        parameter_name = param_names[idx]
        fig_plotly.append_trace(
                go.Histogram(
                    x=list_parameter_value_particles[:,idx],
                    name=parameter_name
                ), row=1, col=idx + 1)
        axs[idx,].hist(list_parameter_value_particles[:,idx], bins=100, density=True, alpha=0.5)
        plt.setp(axs[idx,], xlabel=f'{parameter_name}')
        axs[idx,].grid()
    plt.setp(axs[0], ylabel='Histogram')
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "initial_param_distribution.png"))
    # plt.savefig(fileName)
    fig.savefig(fileName)
    plt.close(fig)   # a multi-chain driver calls this once per chain
    if not light_output:
        fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "initial_param_distribution.html"))
        pyo.plot(fig_plotly, filename=fileName, auto_open=False)

    # Save initial samples (numpy matrix) for prior-vs-posterior comparison later
    initial_param_samples = list_parameter_value_particles.copy()  # shape (N, D)

    # Create list-of-dictionaries for the parameter values and state values, instead of only matrices of values
    list_of_dict_parameter_value_particles = []
    for parameter_particle in list_parameter_value_particles:
        list_of_dict_parameter_value_particles.append(dict(zip(param_names, parameter_particle)))
    list_of_dict_state_value_particles = []
    for state_particle in list_state_values_particles:
        list_of_dict_state_value_particles.append(dict(zip(state_names, state_particle)))
    list_parameter_value_particles = list_of_dict_parameter_value_particles
    list_state_values_particles = list_of_dict_state_value_particles

    uniform_particle_weights = np.ones(number_of_particles) / number_of_particles

    # ── AR(1) structural-error machinery ────────────────────────────────────────
    # Everything below is active only when use_ar_likelihood=True. With the flag off
    # the filter is a plain bootstrap PF: no ε state, no corrected series, and no AR
    # traces or panels in any figure.
    # ε_i(0) = 0:  no structural error known at the start
    epsilon_particles = np.zeros(number_of_particles) if use_ar_likelihood else None

    # Initialize the error variance with a default value
    # Raw ensemble mean of the model output, mean(Q_i). Computed BEFORE the weights for
    # time t are applied, so it has not seen y_obs(t).
    #   OPTION A (use_ar_likelihood=False): this IS the reported forecast.
    #   OPTION C (use_ar_likelihood=True):  the reported forecast is the AR-corrected
    #                                       series below; this is kept for reference.
    final_predicted_streamflow = defaultdict(list, {key:[] for key in list_of_dates_of_interest})
    final_observed_streamflow = defaultdict(list, {key:[] for key in list_of_dates_of_interest})
    # OPTION C reported forecast: mean(Q_i) + mean(ε̂_i) — the quantity the AR likelihood
    # actually claims predicts the observation. None when the AR likelihood is off.
    final_ar_corrected_streamflow = (
        defaultdict(list, {key: [] for key in list_of_dates_of_interest}) if use_ar_likelihood else None)
    # Per-particle ε̂ vectors, so the percentile bands can be corrected particle-by-particle
    # (Q_i + ε̂_i) rather than shifted by the ensemble mean — see the plotting section.
    epsilon_hat_per_date_dict = (
        defaultdict(list, {key: [] for key in list_of_dates_of_interest}) if use_ar_likelihood else None)
    dates = []
    # Per-timestep posterior parameter ensembles and matching model outputs.
    # posterior_params_per_date[k] is (n_particles, n_params) for dates[k]: the
    # RESAMPLED-AND-PERTURBED ensemble that carries forward into date k+1 (saved
    # as "theta" below, kept for continuity with existing downstream consumers).
    # posterior_qoi_per_date[k] is (n_particles,) of Q_i(dates[k]) — this is NOT
    # aligned to posterior_params_per_date[k]. Its correctly-aligned partner is
    # posterior_theta_used_per_date[k] / posterior_state_used_per_date[k] below.
    posterior_params_per_date = []
    posterior_qoi_per_date = []
    # theta_i(t) / state_i(t) that PRODUCED y_t_model_for_date[i] this timestep,
    # i.e. the PRE-resampling ensemble, in the SAME particle order as
    # posterior_qoi_per_date[k]. This is the correctly-aligned (theta, Q) pair.
    posterior_theta_used_per_date = []
    posterior_state_used_per_date = []
    # resample_indices_per_date[k][j] = the index into theta_used[k]/qoi[k] that
    # survived resampling into slot j — i.e. which pre-resampling particle each
    # post-resampling slot is a copy of. Lets the resampled (with-duplicates)
    # ensemble be reconstructed on demand as theta_used[k][resample_indices[k]],
    # and directly shows survivor counts (n unique values = surviving particles).
    resample_indices_per_date = []
    mse = 0
    n_underflow_resets = 0         # timesteps where every likelihood underflowed to 0
    ess_per_date = []              # Effective Sample Size diagnostic
    rmse_per_date = []             # per-timestep absolute error of ensemble mean
    epsilon_hat_mean_per_date = [] # mean AR(1) prediction ε̂ = φ·ε(t−1) across particles
    epsilon_mean_per_date = []     # mean posterior ε(t) = y_obs − Q_model across particles
    epsilon_std_per_date = []      # std of posterior ε(t) across particles
    # Posterior parameter evolution: recorded after resampling, before perturbation
    param_mean_per_date  = {pn: [] for pn in param_names}
    param_p10_per_date   = {pn: [] for pn in param_names}
    param_p90_per_date   = {pn: [] for pn in param_names}
    # Posterior state evolution: recorded after resampling (x_{t+1} propagated by model)
    state_mean_per_date  = {sn: [] for sn in state_names}
    state_p10_per_date   = {sn: [] for sn in state_names}
    state_p90_per_date   = {sn: [] for sn in state_names}
    current_model_output_max = 0.0

    current_model_output_max = 0.0

    # Indices at which to save prior-vs-posterior snapshots (uniformly spaced)
    n_prior_posterior_snapshots = 5
    snapshot_indices = set(
        np.linspace(0, len(list_of_dates_of_interest) - 1, n_prior_posterior_snapshots).astype(int)
    )

    # =========================================================
    # Particle Filtering
    # =========================================================

    # Data structure to store the results
    y_t_model_per_date_dict = defaultdict(list, {key:[] for key in list_of_dates_of_interest})
    data_structure_over_dates = []

    # Outer loop that goes over all date from the configuration json
    pool = multiprocessing.Pool(processes=num_processes)
    for index_date_of_interest in range(len(list_of_dates_of_interest)):
        date_of_interest = list_of_dates_of_interest[index_date_of_interest]
        # print(f"date_of_interest - {date_of_interest}")

        new_list_parameter_value_particles = []
        new_list_state_values_particles = []
        new_list_unique_index_model_run_list = []
        y_t_model_for_date = []
        updated_weights = [] #np.zeros(len(list_parameter_value_particles))

        # AR(1) prediction of structural error for each particle at this timestep.
        # Must be rebuilt each iteration using the resampled epsilon_particles from t-1.
        # Monthly bias (PF convention: positive = model underestimates) is added when supplied.
        # Guarded: with use_ar_likelihood=False there is no ε state (epsilon_particles is None).
        if use_ar_likelihood:
            _monthly_bias = (monthly_bias_ar.get(date_of_interest.month, 0.0)
                             if monthly_bias_ar is not None else 0.0)
            epsilon_hat_by_index = {
                idx: _monthly_bias + phi_ar * epsilon_particles[pos]
                for pos, idx in enumerate(list_unique_index_model_run_list)
            }
        else:
            epsilon_hat_by_index = {}
        new_epsilon_by_index = {}   # reset each timestep; filled below

        # This part of the code is for parallel computing of independent particles, i.e., model runs
        def process_particles_concurrently(particles_to_process):
            for index_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict in \
                    pool.starmap(run_model_single_time_stamp_single_particle, \
                                 [(hbvsaskModelObject, date_of_interest, particle[0], particle[1], particle[2]) \
                                  for particle in particles_to_process]):
                yield index_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict

        row = {}
        likelihood_over_rows = []
        weights_over_rows = []
        resampling_over_rows = []

        # Iterating over the particles and model results for each partcle
        for index_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict in process_particles_concurrently(
    zip(list_parameter_value_particles, list_state_values_particles, list_unique_index_model_run_list)
        ):
            y_t_model_for_date.append(y_t_model)
            new_list_unique_index_model_run_list.append(index_run)
            new_list_state_values_particles.append(x_t_plus_1)
            new_list_parameter_value_particles.append(parameter_value_dict)

            # Record the raw residual ε_i(t) = y_obs − Q_i — the AR state carried to t+1
            if use_ar_likelihood:
                new_epsilon_by_index[index_run] = (
                    (y_t_observed - y_t_model) if y_t_observed is not None else 0.0
                )

            # Likelihood — mode selected by use_ar_likelihood / use_student_t
            if use_ar_likelihood:
                # OPTION C: likelihood centred on Q_i + ε̂_i, which is also what gets
                # reported (see final_ar_corrected_streamflow). Innovation std is
                # σ_η = σ_ε·√(1−φ²), valid because the mean has been shifted by ε̂.
                epsilon_hat_i = epsilon_hat_by_index[index_run]
                if use_student_t:
                    # AR(1) + Student-t innovation: heavier tails, fewer weight collapses
                    likelihood = calculate_likelihood_ar_student_t(
                        y_t_observed, y_t_model, epsilon_hat_i,
                        sigma_eta=sigma_eta, phi_ar=phi_ar,
                        beta_obs=beta_obs, df=student_t_df)
                else:
                    # AR(1) + Gaussian innovation (heteroscedastic when sigma_eta=None)
                    likelihood = calculate_likelihood_ar(
                        y_t_observed, y_t_model, epsilon_hat_i,
                        sigma_eta=sigma_eta, phi_ar=phi_ar,
                        beta_obs=beta_obs)
            else:
                # OPTION A: no AR term. Likelihood centred directly on Q_i, which is
                # what gets reported, using the FULL σ_ε = beta_obs·|y_obs| (not σ_η).
                likelihood = calculate_likelihood_heteroscedastic(
                    y_t_observed, y_t_model,
                    beta_obs=beta_obs, sigma_eps=sigma_eta)
            
            updated_weights.append(likelihood)

            likelihood_over_rows.append(likelihood)

        y_t_model_for_date = np.asarray(y_t_model_for_date)
        current_model_output_max = np.max(y_t_model_for_date) if np.max(y_t_model_for_date) > current_model_output_max else current_model_output_max
        y_t_model_per_date_dict[date_of_interest] = y_t_model_for_date

        # Capture the theta/state that PRODUCED y_t_model_for_date, in the same
        # particle order — BEFORE resampling reassigns list_state_values_particles
        # below. This is the correctly-aligned (theta, Q) pair for PCE work.
        if save_posterior_parameter_samples:
            posterior_theta_used_per_date.append(np.asarray(
                [[p[pn] for pn in param_names] for p in new_list_parameter_value_particles],
                dtype=np.float64))
            posterior_state_used_per_date.append(np.asarray(
                [[s.get(sn, np.nan) for sn in state_names] for s in list_state_values_particles],
                dtype=np.float64))

        updated_weights = np.asarray(updated_weights)

        # Rebuild epsilon list in the same order as new_list_unique_index_model_run_list
        # (which mirrors new_list_parameter_value_particles / new_list_state_values_particles)
        if use_ar_likelihood:
            new_epsilon_particles = [
                new_epsilon_by_index[idx]
                for idx in new_list_unique_index_model_run_list]
            # Per-particle ε̂(t), in the same particle order as y_t_model_for_date,
            # so the bands can later be corrected particle-by-particle.
            epsilon_hat_per_date_dict[date_of_interest] = np.array(
                [epsilon_hat_by_index[idx] for idx in new_list_unique_index_model_run_list])
        else:
            new_epsilon_particles = None

        # row[utility.TIME_COLUMN_NAME] = date_of_interest  # date_of_interest_over_rows
        # row['y_t_model'] = y_t_model_for_date  # y_t_model_over_rows
        # row[utility.INDEX_COLUMN_NAME] = new_list_unique_index_model_run_list  # index_run_over_rows
        # row['likelihood'] = likelihood_over_rows

        # FORECAST (prior) mean — particles entering step t are equally weighted (SIR
        # resampled at t-1), so the plain mean is E[Q_t | y_{1:t-1}]. Must stay BEFORE
        # weight normalisation so it never sees y_obs(t).
        average_predicted_streamflow = np.mean(y_t_model_for_date)  # this can as well be computed from row['y_t_model']
        if y_t_observed is not None and index_date_of_interest>0:  # disregard the zero step with allways a huge mse
            difference = average_predicted_streamflow - y_t_observed
            mse += difference ** 2
            
        # After calculating all the likelihoods and storing them in updated weights
        # Here begins likelihood/weights normalization:
        total_weight = np.sum(updated_weights)
        if abs(total_weight) < 1e-9:
            # Every particle's likelihood underflowed to ~0, so the observation
            # carries NO information at this timestep and the weights are reset to
            # uniform. This shows up in the ESS diagnostic as a spike to exactly N,
            # which is a failure, not a healthy filter. Counted so the run reports
            # how often it happened — a high count means the likelihood is too
            # sharp (lower phi_ar, or raise beta_obs / sigma_eta).
            n_underflow_resets += 1
            normalized_weights = np.ones(len(updated_weights)) / len(updated_weights)
        else:
            normalized_weights = updated_weights / total_weight
        # Now normalized_weights contains the normalized likelihoods
        normalized_weights = np.asarray(normalized_weights)

        # Overwrite the lists storing the particles (i.e., state, parameter values and unique particle indices) for the next time-stamp
        list_unique_index_model_run_list = new_list_unique_index_model_run_list
        # list_state_values_particles  = copy.deepcopy(new_list_state_values_particles) 
        # list_parameter_value_particles = copy.deepcopy(new_list_parameter_value_particles) 

        # Resample particles based on updated weights
        resample_indices = systematic_resample(normalized_weights)
        list_parameter_value_particles = [new_list_parameter_value_particles[i] for i in resample_indices]
        list_state_values_particles = [new_list_state_values_particles[i] for i in resample_indices]
        uniform_particle_weights = [uniform_particle_weights[i] for i in resample_indices]  # TODO this is probably unnecessary

        if save_posterior_parameter_samples:
            resample_indices_per_date.append(np.asarray(resample_indices, dtype=np.int64))

        if use_ar_likelihood:
            epsilon_particles = np.array([new_epsilon_particles[i] for i in resample_indices])

        # Record posterior parameter statistics (after resampling, before perturbation)
        for pn in param_names:
            vals = np.array([p[pn] for p in list_parameter_value_particles])
            param_mean_per_date[pn].append(float(np.mean(vals)))
            param_p10_per_date[pn].append(float(np.percentile(vals, 10)))
            param_p90_per_date[pn].append(float(np.percentile(vals, 90)))

        # Record posterior state statistics (x_{t+1} propagated by model, after resampling)
        for sn in state_names:
            vals = np.array([s[sn] for s in list_state_values_particles if sn in s])
            if len(vals) > 0:
                state_mean_per_date[sn].append(float(np.mean(vals)))
                state_p10_per_date[sn].append(float(np.percentile(vals, 10)))
                state_p90_per_date[sn].append(float(np.percentile(vals, 90)))

        # Perturb the parameters of resampled particles
        if perturbation_scheme == "liu_west":
            # Ensemble mean/std computed ONCE per date, across the whole
            # resampled ensemble - perturb_parameters shrinks each particle
            # toward this mean rather than jittering around its own value.
            param_stds = {
                pn: max(np.std([p[pn] for p in list_parameter_value_particles]), 1e-8)
                for pn in param_names
            }
            ensemble_mean = {
                pn: float(np.mean([p[pn] for p in list_parameter_value_particles]))
                for pn in param_names
            }
        else:
            param_stds = None
            ensemble_mean = None
        list_of_lists_with_parameter_values = []
        dict_of_distriubtions_over_parameters_for_a_date = defaultdict(list, {key:[] for key in param_names})
        for i in range(len(list_parameter_value_particles)):
            list_parameter_value_particles[i] = perturb_parameters(
                list_parameter_value_particles[i],
                param_stds,
                perturbation_factor,
                param_bounds=param_bounds,
                bound_handling=bound_handling,
                min_jitter_frac=min_jitter_frac,
                perturbation_scheme=perturbation_scheme,
                ensemble_mean=ensemble_mean,
                liu_west_delta=liu_west_delta)
            list_of_lists_with_parameter_values.append(list(list_parameter_value_particles[i].values()))

            # print(f"DEBUGGING perturbed parameters values in dict {i} - {list_parameter_value_particles[i]}")
            for parameter_name in param_names:
                dict_of_distriubtions_over_parameters_for_a_date[parameter_name].append(list_parameter_value_particles[i][parameter_name])

            for parameter_name in param_names:
                dict_of_distriubtions_over_parameters_for_a_date[parameter_name].append(list_parameter_value_particles[i][parameter_name])

        # ── Snapshot: prior vs. posterior at uniformly-spaced timesteps ──────
        if index_date_of_interest in snapshot_indices:
            date_str = date_of_interest.strftime('%Y-%m-%d')
            fig_pp, axs_pp = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4), sharey=False)
            axs_pp = np.atleast_1d(axs_pp)
            for idx, param_name in enumerate(param_names):
                prior_vals = initial_param_samples[:, idx]
                post_vals  = np.asarray(dict_of_distriubtions_over_parameters_for_a_date[param_name])
                bins = np.linspace(min(prior_vals.min(), post_vals.min()),
                                   max(prior_vals.max(), post_vals.max()), 40)
                axs_pp[idx].hist(prior_vals, bins=bins, density=True, alpha=0.4,
                                 color='gray', label='Prior')
                axs_pp[idx].hist(post_vals,  bins=bins, density=True, alpha=0.6,
                                 color='steelblue', label='Posterior')
                axs_pp[idx].set_xlabel(param_name)
                axs_pp[idx].set_title(param_name)
                axs_pp[idx].grid(True, alpha=0.3)
            axs_pp[0].set_ylabel('Density')
            axs_pp[0].legend(fontsize=8)
            fig_pp.suptitle(f'Prior vs. Posterior — {date_str}', fontsize=12)
            fig_pp.tight_layout()
            fig_pp.savefig(os.path.join(str(directory_for_saving_plots),
                           f"prior_vs_posterior_params_{date_str}.png"), dpi=150)
            plt.close(fig_pp)

        # Save one big matrix of particle values, might be used later one for transformation of the samples
        # This matrix contains resampled and perturbed parameter values for this timestep, but they do not corresponf 1-to-1 to y_t_model_for_date
        parameter_samples_matrix = list(zip(*list_of_lists_with_parameter_values))  # this should be a matrix of size number_of_particles x number_of_parameters
        parameter_samples_matrix = np.asarray(parameter_samples_matrix).T

        # Keep the POSTERIOR parameter ensemble for this timestep. Note that
        # parameter_samples_matrix above is a loop variable — it is overwritten on
        # every iteration, so only the final timestep would otherwise survive.
        # Storing it per date is what makes a per-timestep transport map + PCE
        # possible later (see save_posterior_parameter_samples below).
        if save_posterior_parameter_samples:
            posterior_params_per_date.append(parameter_samples_matrix.copy())
            posterior_qoi_per_date.append(np.asarray(y_t_model_for_date, dtype=np.float64))

        # row['weights'] = normalized_weights
        # row['resample_indices'] = resample_indices
        # data_structure_over_dates.append(row)

        print(f"Date: {date_of_interest.strftime('%Y-%m-%d')}")
        print(f"Predicted Averaged Streamflow: {average_predicted_streamflow} m^3/s")
        print(f"Observed Streamflow: {y_t_observed} m^3/s")

        final_predicted_streamflow[date_of_interest] = average_predicted_streamflow
        final_observed_streamflow[date_of_interest] = y_t_observed

        # OPTION C reported forecast: add the mean AR(1) prediction of the structural
        # error. Sign convention is ε = y_obs − y_model, so a positive ε̂ (model
        # underestimating) shifts the prediction upward, toward the observation.
        # This matches what the likelihood was centred on, keeping the reported series
        # consistent with the model the filter actually used to weight particles.
        if use_ar_likelihood:
            mean_epsilon_hat = float(np.mean(epsilon_hat_per_date_dict[date_of_interest]))
            reported_forecast = average_predicted_streamflow + mean_epsilon_hat
            final_ar_corrected_streamflow[date_of_interest] = reported_forecast
        else:
            reported_forecast = average_predicted_streamflow

        # ESS and per-timestep absolute error of the REPORTED forecast series
        ess_per_date.append(1.0 / np.sum(normalized_weights ** 2))
        if y_t_observed is not None:
            rmse_per_date.append(abs(reported_forecast - y_t_observed))
        else:
            rmse_per_date.append(np.nan)

        # Epsilon state evolution (AR prior prediction vs posterior residuals)
        if use_ar_likelihood:
            epsilon_hat_mean_per_date.append(mean_epsilon_hat)
            epsilon_mean_per_date.append(float(np.mean(epsilon_particles)))
            epsilon_std_per_date.append(float(np.std(epsilon_particles)))

        dates.append(date_of_interest)

    pool.close()
    pool.join()

    mse_total = mse / len(dates)
    print(f"Final predicted streamflow: {final_predicted_streamflow}")
    print(f"Final observed streamflow: {final_observed_streamflow}")
    print(f"Total MSE: {mse_total}; RMSE: {np.sqrt(mse_total)}")
    
    print(f"FINISH")
        
    # =========================================================
    # Creating Data Structures which will be used for further analysis (saving it) and plotting
    # =========================================================

    # Create one big DataFrame storing all the simulation over time and over different particles
    # unfolded_data_structure_over_dates = []
    # for item in data_structure_over_dates:
    #     for i in range(len(item[utility.INDEX_COLUMN_NAME])):
    #         single_row_dict = {key: value[i] if isinstance(value, list) else value for key, value in item.items()}
    #         unfolded_data_structure_over_dates.append(single_row_dict)
    # df = pd.DataFrame(unfolded_data_structure_over_dates)
    # fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "all_simulations.pkl"))
    # df.to_pickle(fileName, compression="gzip")
    # print(f"df - {df}")

    # Create DataFrames from dictionaries storing the final predicted and observed streamflow
    final_predicted_streamflow_df = pd.DataFrame.from_dict(final_predicted_streamflow, orient='index')
    final_predicted_streamflow_df.rename(columns={0: 'predicted_streamflow'}, inplace=True)
    final_observed_streamflow_df = pd.DataFrame.from_dict(final_observed_streamflow, orient='index')
    final_observed_streamflow_df.rename(columns={0: 'observed_streamflow'}, inplace=True)
    merged_df = final_predicted_streamflow_df.merge(final_observed_streamflow_df, left_index=True, right_index=True)
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "averaged_and_simulated.pkl"))
    merged_df.to_pickle(fileName, compression="gzip")

    # Create one big Matrix containing all model simulations over particles and over dates
    # Extract the lists from the dictionary
    lists = list(y_t_model_per_date_dict.values())
    # Use the zip function to transpose the lists into columns
    y_t_model_per_date_matrix = list(zip(*lists))
    assert len(y_t_model_per_date_matrix[0]) == len(list_of_dates_of_interest)

    # =========================================================
    # Plotting the results
    # =========================================================

    fig, axs = plt.subplots(1, len(param_names), figsize=(20, 10))
    fig_plotly = make_subplots(rows=1, cols=len(param_names))
    for idx in range(len(param_names)):
        parameter_name = param_names[idx]
        # Visualize data from the last dict_of_distriubtions_over_parameters_for_a_date 
        dict_of_distriubtions_over_parameters_for_a_date[parameter_name] = np.asarray(dict_of_distriubtions_over_parameters_for_a_date[parameter_name])
        min_value = dict_of_distriubtions_over_parameters_for_a_date[parameter_name].min() - abs(dict_of_distriubtions_over_parameters_for_a_date[parameter_name].min())*0.001
        max_value= dict_of_distriubtions_over_parameters_for_a_date[parameter_name].max() + abs(dict_of_distriubtions_over_parameters_for_a_date[parameter_name].max())*0.001
        t = np.linspace(min_value, max_value, 1000)
        distribution = cp.GaussianKDE(dict_of_distriubtions_over_parameters_for_a_date[parameter_name], h_mat=0.005 ** 2)
        axs[idx,].hist(dict_of_distriubtions_over_parameters_for_a_date[parameter_name], bins=100, density=True, alpha=0.5)
        axs[idx,].plot(t, distribution.pdf(t), label=f"KDE {parameter_name}")
        plt.setp(axs[idx,], xlabel=f'{parameter_name}')
        axs[idx,].grid()
        fig_plotly.append_trace(
                go.Histogram(
                    x=dict_of_distriubtions_over_parameters_for_a_date[parameter_name],
                    name=parameter_name
                ), row=1, col=idx + 1)
    plt.setp(axs[0], ylabel='PDF')
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "final_param_distribution.png"))
    # plt.savefig(fileName)
    fig.savefig(fileName)
    plt.close(fig)   # a multi-chain driver calls this once per chain
    if not light_output:
        fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "final_param_distribution.html"))
        pyo.plot(fig_plotly, filename=fileName, auto_open=False)

    # ── Persist per-timestep posterior ensembles for later transport-map + PCE ──
    # theta: (n_dates, n_particles, n_params) — RESAMPLED+PERTURBED ensemble that
    # carries forward into date k+1 (kept for continuity with existing consumers).
    # qoi: (n_dates, n_particles) of Q_i(dates[k]).
    # theta_used/state_used: the PRE-resampling theta/state that actually produced
    # qoi[k] — these are the correctly-aligned (theta, Q) regression pair a PCE
    # needs. theta[k] and qoi[k] are NOT aligned to each other: theta[k] is
    # resampled+jittered for use at k+1, while qoi[k] came from the ensemble that
    # entered step k (theta[k-1]'s resampled+jittered values) — one particle-index
    # shuffle removed from qoi[k].
    # resample_indices[k][j] = index into theta_used[k]/qoi[k] that survived into
    # post-resampling slot j. The resampled (with-duplicates) ensemble can be
    # reconstructed on demand as theta_used[k][resample_indices[k]] — this is
    # cheaper than storing that duplicated array outright, and the indices
    # themselves show survivor counts directly (n unique = particles that survived).
    param_output_corr = None
    if save_posterior_parameter_samples and posterior_params_per_date:
        theta_stack = np.stack(posterior_params_per_date, axis=0)
        qoi_stack = np.stack(posterior_qoi_per_date, axis=0)
        theta_used_stack = np.stack(posterior_theta_used_per_date, axis=0)
        state_used_stack = np.stack(posterior_state_used_per_date, axis=0)
        resample_indices_stack = np.stack(resample_indices_per_date, axis=0)
        # Computed on the correctly-aligned (theta_used, qoi) pair — NOT (theta,
        # qoi), which are one resampling-and-jitter step apart from each other.
        # ~0 means the parameter does not move the output, so no likelihood can
        # select on it and the posterior for it cannot converge.
        param_output_corr = parameter_output_correlation(theta_used_stack, qoi_stack)
        print("median |corr(parameter, Q)| across particles, on the ALIGNED "
              "(theta_used, qoi) pair (<0.1 = filter cannot learn this parameter):")
        for nm, c in zip(param_names, param_output_corr):
            print(f"    {nm:<8}{c:6.3f}")
        print(f"    {'overall':<8}{np.nanmedian(param_output_corr):6.3f}")
        # theta/theta_used/state_used dominate this file. float32 keeps ~7
        # significant digits — far more than parameter/state values carry — and
        # roughly halves their size. qoi stays float64: it is much smaller and
        # feeds the pooled streamflow statistics. resample_indices is int64 and
        # tiny (one int per particle per date) — not worth downcasting.
        if save_theta_float32:
            theta_stack = theta_stack.astype(np.float32)
            theta_used_stack = theta_used_stack.astype(np.float32)
            state_used_stack = state_used_stack.astype(np.float32)
        samples_file = os.path.abspath(os.path.join(
            str(directory_for_saving_plots), "posterior_parameter_samples.npz"))
        np.savez_compressed(
            samples_file,
            theta=theta_stack,
            qoi=qoi_stack,
            theta_used=theta_used_stack,
            state_used=state_used_stack,
            state_names=np.array(state_names, dtype=object),
            resample_indices=resample_indices_stack,
            param_names=np.array(param_names, dtype=object),
            dates=np.array([str(d) for d in dates], dtype=object),
            param_lower=np.array([param_bounds[p][0] for p in param_names]),
            param_upper=np.array([param_bounds[p][1] for p in param_names]),
        )
        print(f"Saved posterior parameter samples theta{theta_stack.shape} "
              f"theta_used{theta_used_stack.shape} state_used{state_used_stack.shape} "
              f"resample_indices{resample_indices_stack.shape} "
              f"[theta dtype={theta_stack.dtype}] -> {samples_file}")

        # ── Per-timestep Gaussianization (PCE inputs) ───────────────────────────
        # Runs on the file just written, so it is identical to doing it inside the
        # loop but costs the filter nothing and is re-runnable with other settings.
        if map_all_timesteps:
            if transport_map_backend in transport_timeseries.BATCH_METHODS or \
                    transport_map_backend in ("mpart", "1d"):
                try:
                    transport_timeseries.map_timesteps(
                        samples_file,
                        method=transport_map_backend,
                        max_order=transport_map_max_order,
                        n_workers=map_all_timesteps_workers,
                        verbose=True)
                except Exception as e:
                    print(f"WARNING: per-timestep mapping failed ({type(e).__name__}: {e}); "
                          f"the filter results are unaffected and it can be re-run "
                          f"later with transport_timeseries.map_timesteps().")
            else:
                print(f"map_all_timesteps=True ignored: transport_map_backend must be "
                      f"'mpart' or 'anamorphosis', got {transport_map_backend!r}.")
    elif map_all_timesteps:
        print("map_all_timesteps=True ignored: needs save_posterior_parameter_samples=True.")

    # ── Gaussianize the final-timestep ensemble (diagnostic plot) ───────────────
    # print(f"DEBUGGING - {parameter_samples_matrix.shape}")
    # standar_parameter_samples_matrix = gaussianize_parameter_samples(
    #     parameter_samples_matrix,
    #     method=transport_map_backend,
    #     max_order=transport_map_max_order,
    #     param_names=param_names)
    # if standar_parameter_samples_matrix is None:
    #     standar_parameter_samples_matrix = parameter_samples_matrix
    # print(f"DEBUGGING - {standar_parameter_samples_matrix.shape}")
    # # Plotting final distribution of transformed parameter values
    # fig_plotly = make_subplots(rows=1, cols=len(param_names))
    # for idx in range(len(param_names)):
    #     parameter_name = param_names[idx]
    #     fig_plotly.append_trace(
    #             go.Histogram(
    #                 x=standar_parameter_samples_matrix[:,idx],
    #                 name=parameter_name
    #             ), row=1, col=idx + 1)
    # if not light_output:
    #     fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "final_transformed_param_distribution.html"))
    #     pyo.plot(fig_plotly, filename=fileName, auto_open=False)

    # ── Percentile band ────────────────────────────────
    particle_matrix = np.array(y_t_model_per_date_matrix)  # (N_particles, N_dates)

    obs_vals = np.array([final_observed_streamflow[d] for d in dates], dtype=float)

    # OPTION C: the reported series is Q + ε̂, so the bands must describe the same
    # model — otherwise the reported mean is drawn against bands belonging to a
    # different model and the P-factor describes the wrong one. Corrected PER
    # PARTICLE (Q_i + ε̂_i), which is exactly what each particle was weighted against.
    # OPTION A: no AR term, bands are the raw particle spread.
    if use_ar_likelihood:
        eps_hat_matrix = np.column_stack(
            [np.asarray(epsilon_hat_per_date_dict[d], dtype=float) for d in dates])
        assert eps_hat_matrix.shape == particle_matrix.shape, (
            f"ε̂ matrix {eps_hat_matrix.shape} != particle matrix {particle_matrix.shape}")
        mean_matrix = particle_matrix + eps_hat_matrix
        band_label_suffix = " (AR-corrected)"
    else:
        mean_matrix = particle_matrix
        band_label_suffix = ""

    # ── Predictive bands must include the innovation/observation noise ──────────
    # mean_matrix holds only the spread of the predictive MEAN across particles.
    # The likelihood states y_obs = Q_i + ε̂_i + η with η ~ N(0, σ_η²) (or a
    # Student-t when use_student_t), so a band built from mean_matrix alone is NOT
    # the posterior predictive distribution and systematically under-covers — which
    # also makes the P-factor meaningless. Draw one η per particle per date.
    #
    # This matters most under the AR likelihood: expanding
    #   band_i(t) = Q_i(t) − φ·Q_i(t−1) + [μ + φ·y_obs(t−1)]
    # shows the particle-dependent part is scaled by (1−φ), so with φ=0.894 the
    # across-particle spread is compressed to ~11% of the raw Q spread. Without the
    # η term almost no uncertainty remains to plot.
    # Which flow scales the heteroscedastic sigma for the BAND:
    #   "observed" — sigma proportional to |y_obs(t)|, matching the likelihood
    #                (Pianosi 2016). Defensible inside the update step, but it makes
    #                the band's WIDTH depend on the observation being predicted, so
    #                the result is a hindcast band, not a forecast band.
    #   "forecast" — sigma proportional to the ensemble-mean prediction, which is
    #                available before y_obs(t) is seen. Leak-free.
    if band_sigma_from == "forecast":
        scale_flow = np.array([final_predicted_streamflow[d] for d in dates], dtype=float)
    elif band_sigma_from == "observed":
        scale_flow = obs_vals
    else:
        raise ValueError(f"band_sigma_from must be 'observed' or 'forecast', "
                         f"got {band_sigma_from!r}.")

    if sigma_eta is not None:
        sigma_per_date = np.full(len(dates), float(sigma_eta))   # homoscedastic override
    elif use_ar_likelihood:
        sigma_per_date = beta_obs * np.sqrt(1.0 - phi_ar ** 2) * np.abs(scale_flow)
    else:
        sigma_per_date = beta_obs * np.abs(scale_flow)           # Option A uses full σ_ε
    sigma_per_date = np.maximum(sigma_per_date, 1e-6)

    if include_innovation_in_bands:
        rng_bands = np.random.default_rng(band_noise_seed)
        if use_ar_likelihood and use_student_t:
            # match the Student-t innovation used in the likelihood
            noise = student_t.rvs(df=student_t_df, size=mean_matrix.shape,
                                  random_state=rng_bands)
        else:
            noise = rng_bands.standard_normal(mean_matrix.shape)
        band_matrix = mean_matrix + noise * sigma_per_date[np.newaxis, :]
        band_note = " + innovation"
    else:
        band_matrix = mean_matrix
        band_note = ""

    pct_05 = np.percentile(band_matrix, 5,  axis=0)
    pct_25 = np.percentile(band_matrix, 25, axis=0)
    pct_50 = np.percentile(band_matrix, 50, axis=0)
    pct_75 = np.percentile(band_matrix, 75, axis=0)
    pct_95 = np.percentile(band_matrix, 95, axis=0)

    # Parameter-uncertainty-only band, for reference: no innovation noise, and for
    # the AR case not compressed by (1−φ) because it uses the raw model output.
    raw_pct_05 = np.percentile(particle_matrix, 5,  axis=0)
    raw_pct_95 = np.percentile(particle_matrix, 95, axis=0)

    # P-factor: fraction of observations inside the 5-95% band of the REPORTED model
    in_band = np.sum((obs_vals >= pct_05) & (obs_vals <= pct_95))
    p_factor = in_band / len(obs_vals) if len(obs_vals) > 0 else np.nan
    raw_in_band = np.sum((obs_vals >= raw_pct_05) & (obs_vals <= raw_pct_95))
    raw_p_factor = raw_in_band / len(obs_vals) if len(obs_vals) > 0 else np.nan
    print(f"P-factor (5–95% band{band_label_suffix}{band_note}): {p_factor:.3f}")
    print(f"P-factor (5–95% raw Q, parameter uncertainty only): {raw_p_factor:.3f}")
    print(f"Mean 5–95% width: reported={np.mean(pct_95-pct_05):.2f} m³/s, "
          f"raw Q={np.mean(raw_pct_95-raw_pct_05):.2f} m³/s")

    # ── Filter health summary ───────────────────────────────────────────────────
    _ess = np.asarray(ess_per_date, dtype=float)
    print(f"ESS: median={np.median(_ess):.0f} of N={number_of_particles} | "
          f"<N/10 on {np.mean(_ess < number_of_particles/10):.1%} of steps | "
          f"<N/2 on {np.mean(_ess < number_of_particles/2):.1%}")
    if n_underflow_resets:
        print(f"WARNING: total likelihood underflow on {n_underflow_resets}/{len(dates)} "
              f"timesteps ({n_underflow_resets/len(dates):.1%}) — the observation was "
              f"IGNORED there (weights reset to uniform, ESS spikes to N). "
              f"The likelihood is too sharp: raise beta_obs (currently {beta_obs:.4f}) "
              f"or set an explicit sigma_eta.")

    # ── Persist the headline results next to the configuration ──────────────────
    # run_configuration.json says how the run was set up; this says how it went.
    # Together they make a result folder self-describing and comparable across runs.
    _abs_err = np.asarray(rmse_per_date, dtype=float)
    run_results = {
        "n_timesteps": len(dates),
        "date_first": str(dates[0]) if dates else None,
        "date_last": str(dates[-1]) if dates else None,
        "rmse_reported_forecast": float(np.sqrt(np.nanmean(_abs_err ** 2))),
        "mae_reported_forecast": float(np.nanmean(_abs_err)),
        "mse_total": float(mse_total),
        "p_factor_reported": float(p_factor),
        "p_factor_raw_q": float(raw_p_factor),
        "mean_band_width_reported": float(np.mean(pct_95 - pct_05)),
        "mean_band_width_raw_q": float(np.mean(raw_pct_95 - raw_pct_05)),
        "band_includes_innovation": bool(include_innovation_in_bands),
        "ess_median": float(np.median(_ess)),
        "ess_mean": float(np.mean(_ess)),
        "ess_frac_below_N_over_10": float(np.mean(_ess < number_of_particles / 10)),
        "ess_frac_below_N_over_2": float(np.mean(_ess < number_of_particles / 2)),
        "n_underflow_resets": int(n_underflow_resets),
        "median_abs_corr_param_qoi": (
            {nm: float(c) for nm, c in zip(param_names, param_output_corr)}
            if param_output_corr is not None else None),
        "median_abs_corr_overall": (
            float(np.nanmedian(param_output_corr))
            if param_output_corr is not None else None),
        "frac_underflow_resets": float(n_underflow_resets / len(dates)) if dates else None,
    }
    save_run_configuration(directory_for_saving_plots, run_results,
                           filename="run_summary.json")

    fig = go.Figure()

    # Raw particle band: 5–95 % of Q_i alone.
    # Drawn first (behind the others) because it is the widest.
    #
    # This is the only band that is a genuine FORECAST band. It comes purely from
    # the particle ensemble and never touches y_obs(t). The AR-corrected band does,
    # twice over: ε̂ carries φ·y_obs(t−1), and — more importantly — the innovation
    # scale σ_η(t) = β·√(1−φ²)·|y_obs(t)| is proportional to the very observation
    # being predicted. Its width is therefore informed by the answer, so it should
    # be read as a hindcast/analysis band, not a predictive one.
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=list(raw_pct_95) + list(raw_pct_05[::-1]),
        fill='toself', fillcolor='rgba(120,120,120,0.18)',
        line=dict(color='rgba(0,0,0,0)'),
        name='5–95% band (raw Q, parameter uncertainty)', hoverinfo='skip'))

    # Outer band: 5–95 %
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=list(pct_95) + list(pct_05[::-1]),
        fill='toself', fillcolor='rgba(173,216,230,0.35)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'5–95% band{band_label_suffix}{band_note}', hoverinfo='skip'))

    # Inner band: 25–75 %
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=list(pct_75) + list(pct_25[::-1]),
        fill='toself', fillcolor='rgba(70,130,180,0.35)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'25–75% band{band_label_suffix}', hoverinfo='skip'))

    # Median
    fig.add_trace(go.Scatter(
        x=dates, y=pct_50, mode='lines',
        line=dict(color='steelblue', width=1.5, dash='dash'),
        name=f'Median prediction{band_label_suffix}'))

    # Forcing data
    if PLOT_FORCING_DATA:
        reset_index_at_the_end = False
        if hbvsaskModelObject.time_series_measured_data_df.index.name != utility.TIME_COLUMN_NAME:
            hbvsaskModelObject.time_series_measured_data_df.set_index(utility.TIME_COLUMN_NAME, inplace=True)
            reset_index_at_the_end = True
        temp = hbvsaskModelObject.time_series_measured_data_df[
            hbvsaskModelObject.time_series_measured_data_df.index.isin(list_of_dates_of_interest)]
        N_max = temp['precipitation'].max()
        fig.add_trace(go.Bar(
            x=temp.index, y=temp['precipitation'],
            name='Precipitation', yaxis="y2", marker_color='rgba(31,119,180,0.5)'))
        if reset_index_at_the_end:
            hbvsaskModelObject.time_series_measured_data_df.reset_index(inplace=True)
            hbvsaskModelObject.time_series_measured_data_df.rename(
                columns={hbvsaskModelObject.time_series_measured_data_df.index.name: utility.TIME_COLUMN_NAME},
                inplace=True)

    # Observed streamflow
    fig.add_trace(go.Scatter(
        x=merged_df.index, y=merged_df['observed_streamflow'],
        name='Observed', line=dict(color='orange', width=2.5)))

    # Reported forecast. Both options are computed before the weights are applied, so
    # neither has seen y_obs(t).
    #   OPTION C: mean(Q) + mean(ε̂) — matches the AR-corrected bands above.
    #   OPTION A: mean(Q) — matches the raw bands above.
    if use_ar_likelihood:
        fig.add_trace(go.Scatter(
            x=dates, y=[final_ar_corrected_streamflow[d] for d in dates],
            name='Ensemble mean (AR-corrected)', line=dict(color='blue', width=2)))

        # Reference: the raw model mean, without the structural-error term
        fig.add_trace(go.Scatter(
            x=merged_df.index, y=merged_df['predicted_streamflow'],
            name='Ensemble mean (raw Q, uncorrected)',
            line=dict(color='grey', width=1.5, dash='dot'),
            visible='legendonly'))
    else:
        fig.add_trace(go.Scatter(
            x=merged_df.index, y=merged_df['predicted_streamflow'],
            name='Ensemble mean', line=dict(color='blue', width=2)))

    # Clip y-axis so wide initial bands don't dominate; viewer can zoom to see full range
    y_max_obs = np.nanmax(obs_vals) if len(obs_vals) > 0 else 1.0
    spinup_steps = 30  # timesteps to shade as filter warm-up period

    fig.update_xaxes(title_text="Date", type="date",
                     range=[hbvsaskModelObject.start_date_predictions, hbvsaskModelObject.end_date])
    fig.update_yaxes(title_text="Q [m³/s]", side="left", domain=[0, 0.7],
                     range=[0, y_max_obs * 1.4],
                     mirror=True, tickfont={"color": "#d62728"},
                     title=dict(font={"color": "#d62728"}))

    if len(dates) > spinup_steps:
        fig.add_vrect(
            x0=dates[0], x1=dates[spinup_steps - 1],
            fillcolor="grey", opacity=0.12, layer="below", line_width=0,
            annotation_text="Warm-up", annotation_position="top left",
            annotation_font_size=11, annotation_font_color="grey")

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=f'Particle Filter — {ne} particles  |  P-factor={p_factor:.2f}  |  RMSE={np.sqrt(mse_total):.2f} m³/s',
        showlegend=True,
        template="plotly_white",
    )
    if PLOT_FORCING_DATA:
        fig.update_layout(yaxis2=dict(
            anchor="x", domain=[0.7, 1], mirror=True,
            range=[N_max, 0], side="right",
            tickfont={"color": '#1f77b4'}, nticks=3,
            title=dict(text="N [mm/h]", font={"color": '#1f77b4'}),
            type="linear"))

    fig.show()
    if not light_output:
        pyo.plot(fig, filename=directory_for_saving_plots + "particle_filter_streamflow.html", auto_open=False)
    try:
        fig.write_image(directory_for_saving_plots + "particle_filter_streamflow.pdf",
                        width=1400, height=700)
    except Exception as e:
        print(f"PDF export skipped (install kaleido): {e}")

    # ── Prior vs. posterior parameter comparison (matplotlib) ───────────────
    fig_pp, axs_pp = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4),
                                   sharey=False)
    axs_pp = np.atleast_1d(axs_pp)
    for idx, param_name in enumerate(param_names):
        prior_vals = initial_param_samples[:, idx]
        post_vals  = np.asarray(dict_of_distriubtions_over_parameters_for_a_date[param_name])
        bins = np.linspace(min(prior_vals.min(), post_vals.min()),
                           max(prior_vals.max(), post_vals.max()), 40)
        axs_pp[idx].hist(prior_vals, bins=bins, density=True, alpha=0.4,
                         color='gray', label='Prior')
        axs_pp[idx].hist(post_vals,  bins=bins, density=True, alpha=0.6,
                         color='steelblue', label='Posterior')
        axs_pp[idx].set_xlabel(param_name)
        axs_pp[idx].set_title(param_name)
        axs_pp[idx].grid(True, alpha=0.3)
    axs_pp[0].set_ylabel('Density')
    axs_pp[0].legend(fontsize=8)
    fig_pp.suptitle('Prior vs. Posterior parameter distributions', fontsize=12)
    fig_pp.tight_layout()
    fig_pp.savefig(os.path.join(str(directory_for_saving_plots), "prior_vs_posterior_params.png"), dpi=150)
    plt.close(fig_pp)

    # ── Diagnostics: ESS, cumulative RMSE, epsilon state evolution ──────────
    abs_errors = np.array(rmse_per_date, dtype=float)
    cumulative_rmse = np.sqrt(
        np.nancumsum(abs_errors ** 2) / np.arange(1, len(abs_errors) + 1))

    # The two ε panels only exist in AR mode — a plain PF has no structural error state
    n_diag_panels = 4 if use_ar_likelihood else 2
    fig_diag, axes = plt.subplots(n_diag_panels, 1,
                                  figsize=(12, 3 * n_diag_panels), sharex=True)
    axes = np.atleast_1d(axes)
    if use_ar_likelihood:
        eps_mean = np.array(epsilon_mean_per_date)
        eps_std  = np.array(epsilon_std_per_date)
        eps_hat  = np.array(epsilon_hat_mean_per_date)
        ax_ess, ax_err, ax_eps, ax_eps_spread = axes
    else:
        ax_ess, ax_err = axes
        ax_eps = ax_eps_spread = None

    # Panel 1 — Effective Sample Size
    ax_ess.plot(dates, ess_per_date, color='steelblue', linewidth=1.2)
    ax_ess.axhline(number_of_particles / 2, color='red', linestyle='--', linewidth=0.9,
                   label=f'N/2 = {number_of_particles // 2}')
    ax_ess.set_ylabel('ESS')
    ax_ess.set_title('Effective Sample Size  (collapses below N/2)')
    ax_ess.legend(fontsize=8)
    ax_ess.grid(True, alpha=0.3)

    # Panel 2 — Per-timestep |error| + cumulative RMSE of the REPORTED forecast series
    _err_series = 'AR-corrected mean' if use_ar_likelihood else 'ensemble mean'
    ax_err.plot(dates, abs_errors, color='tomato', linewidth=1.0, alpha=0.7,
                label=f'|Error| {_err_series}')
    ax_err.plot(dates, cumulative_rmse, color='darkred', linewidth=1.8,
                label='Cumulative RMSE')
    ax_err.set_ylabel('[m³/s]')
    ax_err.set_title(f'Per-timestep absolute error and cumulative RMSE ({_err_series})')
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)

    if use_ar_likelihood:
        # Panel 3 — AR(1) prediction vs posterior mean structural error
        ax_eps.plot(dates, eps_hat,  color='purple', linewidth=1.2, linestyle='--',
                    label='ε̂ = φ·ε(t−1) + μ_m  [AR prior]')
        ax_eps.plot(dates, eps_mean, color='darkgreen', linewidth=1.2,
                    label='Mean ε(t) = y_obs − Q_model  [posterior]')
        ax_eps.axhline(0, color='black', linewidth=0.7, linestyle=':')
        ax_eps.set_ylabel('[m³/s]')
        ax_eps.set_title('Structural error state: AR prior prediction vs posterior mean')
        ax_eps.legend(fontsize=8)
        ax_eps.grid(True, alpha=0.3)

        # Panel 4 — Particle spread in epsilon (diversity of error states)
        ax_eps_spread.fill_between(dates,
                                   eps_mean - eps_std, eps_mean + eps_std,
                                   alpha=0.35, color='teal', label='ε mean ± std')
        ax_eps_spread.plot(dates, eps_mean, color='teal', linewidth=1.2)
        ax_eps_spread.axhline(0, color='black', linewidth=0.7, linestyle=':')
        ax_eps_spread.set_ylabel('[m³/s]')
        ax_eps_spread.set_xlabel('Date')
        ax_eps_spread.set_title('Particle spread in structural error state ε(t)')
        ax_eps_spread.legend(fontsize=8)
        ax_eps_spread.grid(True, alpha=0.3)
    else:
        ax_err.set_xlabel('Date')

    fig_diag.autofmt_xdate()
    fig_diag.tight_layout()
    fig_diag.savefig(os.path.join(str(directory_for_saving_plots), "diagnostics.png"), dpi=150)
    plt.close(fig_diag)

    # ── Posterior parameter evolution ────────────────────────────────────────
    n_params = len(param_names)
    fig_pe, axs_pe = plt.subplots(n_params, 1,
                                   figsize=(12, 3 * n_params), sharex=True)
    axs_pe = np.atleast_1d(axs_pe)
    for ax, pn in zip(axs_pe, param_names):
        mean_vals = np.array(param_mean_per_date[pn])
        p10_vals  = np.array(param_p10_per_date[pn])
        p90_vals  = np.array(param_p90_per_date[pn])
        ax.fill_between(dates, p10_vals, p90_vals,
                        alpha=0.30, color='steelblue', label='P10–P90')
        ax.plot(dates, mean_vals, color='steelblue', linewidth=1.4, label='Mean')
        ax.set_ylabel(pn, fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
    axs_pe[-1].set_xlabel('Date')
    fig_pe.suptitle('Posterior parameter evolution over time', fontsize=12)
    fig_pe.autofmt_xdate()
    # tight_layout() alone does not reserve space for suptitle, so it overlaps
    # the top panel once figsize grows with n_params (here 3*n_params inches
    # tall). Reserving a fixed INCH amount off the top (not a fixed fraction)
    # keeps the gap right whether n_params is 1 or 7.
    fig_pe.tight_layout(rect=[0, 0, 1, 1 - 0.4 / fig_pe.get_figheight()])
    fig_pe.savefig(os.path.join(str(directory_for_saving_plots), "param_evolution.png"), dpi=150)
    plt.close(fig_pe)

    # ── Posterior state evolution ────────────────────────────────────────────
    # Only plot states for which data was actually recorded (guards against
    # x_{t+1} dicts missing a key in early timesteps)
    plottable_states = [sn for sn in state_names if len(state_mean_per_date[sn]) == len(dates)]
    if plottable_states:
        n_states = len(plottable_states)
        fig_se, axs_se = plt.subplots(n_states, 1,
                                       figsize=(12, 3 * n_states), sharex=True)
        axs_se = np.atleast_1d(axs_se)
        for ax, sn in zip(axs_se, plottable_states):
            mean_vals = np.array(state_mean_per_date[sn])
            p10_vals  = np.array(state_p10_per_date[sn])
            p90_vals  = np.array(state_p90_per_date[sn])
            ax.fill_between(dates, p10_vals, p90_vals,
                            alpha=0.30, color='darkorange', label='P10–P90')
            ax.plot(dates, mean_vals, color='darkorange', linewidth=1.4, label='Mean')
            ax.set_ylabel(sn, fontsize=9)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
        axs_se[-1].set_xlabel('Date')
        fig_se.suptitle('Posterior state evolution over time', fontsize=12)
        fig_se.autofmt_xdate()
        # See the identical fix on fig_pe above: reserve a fixed INCH amount
        # off the top for the suptitle, not a fixed fraction, so it scales
        # correctly as figsize grows with n_states.
        fig_se.tight_layout(rect=[0, 0, 1, 1 - 0.4 / fig_se.get_figheight()])
        fig_se.savefig(os.path.join(str(directory_for_saving_plots), "state_evolution.png"), dpi=150)
        plt.close(fig_se)

    # Nothing here holds a figure the caller still wants, and a multi-chain
    # driver calls this routine once per chain — so any figure left open would
    # accumulate across chains rather than being reclaimed between them.
    plt.close("all")

    # Return the directory this run wrote into. The paths block above builds
    # workingDir internally, so callers cannot otherwise know where the output
    # landed; a multi-chain driver needs it to pass the chain directories to
    # pool_chain_results.
    return workingDir


if __name__ == "__main__":

    # Number of parallel processes
    num_processes = multiprocessing.cpu_count()
    print(f"Number of parallel processes = {num_processes}")
    number_of_particles = ne = 5000  # 50, 100, 500 2000

    # BASE_SOURCE_PATH = pathlib.Path.cwd().parents[1] # uqef_dynamic
    BASE_SOURCE_PATH = pathlib.Path(__file__).resolve().parents[2]
    hbv_model_data_path = BASE_SOURCE_PATH / "data" / "HBV-SASK-data"

    multiple_chains = True  # True → run several independent chains and pool the results; False → single chain only

    # ==========================================================================
    # Single-CHAIN RUN
    # ==========================================================================

    if not multiple_chains:
        for i in range(1,2):
            # working_dir_name=f"trial_single_run_hbvsaskmodel_7d_filtering/run_{i}"
            working_dir_name=f"hbvsaskmodel_7d_{number_of_particles}_filtering_gaussian_likelihood_heteroscedastic_one_year_Uniform /run_{i}"
            inputModelDir = hbv_model_data_path
            configuration_file = BASE_SOURCE_PATH / "data" / "configurations" / "configuration_hbv_sask_PF_one_year.json"

            with open(configuration_file) as _f:
                _cfg_json = json.load(_f)
            basin = _cfg_json.get("model_settings", {}).get("basin")
            if not basin:
                raise ValueError(
                    f"model_settings.basin is missing from {configuration_file}; the pipeline "
                    "no longer hardcodes a basin.")
            # Output folder follows the basin, so runs for different basins cannot collide.
            workingDir = (inputModelDir / "particle_filtering_model_runs"
                        / basin.lower() / working_dir_name)
            directory_for_saving_plots = workingDir

            main_routine(
                inputModelDir=inputModelDir,
                configuration_file=configuration_file,
                workingDir=workingDir,
                directory_for_saving_plots=directory_for_saving_plots,
                num_processes=num_processes, 
                number_of_particles=number_of_particles, 
                use_ar_likelihood=False,  # True → AR(1)-augmented likelihood; False → standard Gaussian likelihood
                sigma_eta=None,  # 14.2,  # fixed innovation std [m³/s]; None → heteroscedastic mode
                phi_ar=0.894,    # AR(1) coefficient — fit from error_signal_analysis.py
                beta_obs=0.2, #0.2, 0.5/3, #0.2/3, 1.0/3,  # used only when sigma_eta=None: σ_ε = beta_obs·y_obs (0.2/3 ≈ 20% as 3σ bound)
                monthly_bias_ar=None, # monthly_bias_ar=None,  # None → no monthly bias correction; otherwise a dict {month: bias} to subtract from y_obs(t) before likelihood evaluation
                use_student_t=False, # True → Student-t likelihood; False → Gaussian likelihood
                # random_seed=1000,  # seed for the random number generator (initial parameter/state draws, resampling offsets, parameter perturbation)
                perturbation_scheme="magnitude",  # "magnitude" (η·|θ_i|, default) or "liu_west" (shrink to ensemble mean, variance-preserving)
                perturbation_factor=0.15,  # η for "magnitude" — float, or {name: η} for parameter-specific jitter scales. Unused by "liu_west".
                # perturbation_factor = {
                #     "TT": 0.15, "C0": 0.15, "ETF": 0.15, "PM": 0.15,
                #     "FC": 0.106, "FRAC": 0.16, "K2": 0.087,
                # },
                liu_west_delta=0.98,  # discount factor for "liu_west"; unused by "magnitude"
                min_jitter_frac=0.002,  # floor as a fraction of (upper-lower); rescues theta=0 ("magnitude") or a collapsed ensemble ("liu_west")
                bound_handling="reflect",  # "clip" or "reflect"
                band_sigma_from="forecast",  # "forecast" → σ_η(t) ∝ |Q̄(t)|; "observed" → σ_η(t) ∝ |y_obs(t)|
                include_innovation_in_bands=False, # True → bands include η ~ N(0, σ_η²) noise; False → bands show only the across-particle spread of Q_i + ε̂_i
                light_output=True,  # True → skip HTML plots and large .npz files; False → save everything
                save_theta_float32=True,  # True → save θ in float32 (halves size, still ~7 sig digits); False → save θ in float64
                map_all_timesteps=False, # True → run transport_timeseries.map_timesteps() on the saved posterior samples; False → skip it
                transport_map_backend="mpart",
                transport_map_max_order=2,
                map_all_timesteps_workers=num_processes,
                )
    else:
        # ==========================================================================
        # MULTI-CHAIN RUN — average out the influence of the initial sample
        # ==========================================================================

        # Each chain is an independent estimate of the same posterior, differing only
        # in its random seed (initial parameter/state draws, resampling offsets,
        # parameter perturbation). Pooling several reduces the effect of one unlucky
        # initial ensemble. Keep EVERY other argument identical across chains.
        #
        n_chains = 5 #10
        base_name = f"hbvsaskmodel_7d_{number_of_particles}_filtering_gaussian_likelihood_heteroscedastic_two_years_Uniform_{n_chains}_chains"
        chain_dirs = []
        for i in range(n_chains):
            working_dir_name = f"{base_name}/run_{i}"
            inputModelDir = hbv_model_data_path
            configuration_file = BASE_SOURCE_PATH / "data" / "configurations" / "configuration_hbv_sask_PF_two_years.json" #"configuration_hbv_sask_PF_three_years.json"
            with open(configuration_file) as _f:
                _cfg_json = json.load(_f)
            basin = _cfg_json.get("model_settings", {}).get("basin")
            if not basin:
                raise ValueError(
                    f"model_settings.basin is missing from {configuration_file}; the pipeline "
                    "no longer hardcodes a basin.")
            # Output folder follows the basin, so runs for different basins cannot collide.
            workingDir = (inputModelDir / "particle_filtering_model_runs"
                        / basin.lower() / working_dir_name)
            directory_for_saving_plots = workingDir
            chain_dir = main_routine(          # main_routine returns its workingDir
                inputModelDir=inputModelDir,
                configuration_file=configuration_file,
                workingDir=workingDir,
                directory_for_saving_plots=directory_for_saving_plots,
                num_processes=num_processes,
                number_of_particles=number_of_particles,
                random_seed=1000 + i,          # <- the ONLY thing that differs per chain
                use_ar_likelihood=False,
                sigma_eta=None,
                phi_ar=0.894,
                beta_obs=0.2,
                monthly_bias_ar=None,
                use_student_t=False,
                perturbation_scheme="magnitude",
                perturbation_factor=0.15,
                liu_west_delta=0.98,
                min_jitter_frac=0.002,
                bound_handling="reflect",  # "clip" or "reflect"
                band_sigma_from="forecast",
                include_innovation_in_bands=False,
                save_posterior_parameter_samples=True,   # REQUIRED for pooling
                light_output=True,
                save_theta_float32=True,
            )
            chain_dirs.append(chain_dir)
        
        # Observed series, read back from any chain (identical across chains)
        merged = pd.read_pickle(pathlib.Path(chain_dirs[0]) / "averaged_and_simulated.pkl",
                                compression="gzip")
        observed = merged["observed_streamflow"].to_numpy()
        
        # Pool the PARTICLES (never average the per-chain quantiles — see the
        # docstring of pool_chain_results). Writes pooled_chains.npz and
        # pooled_chains_summary.json next to the run_* folders.
        pooled = pool_chain_results(chain_dirs, observed=observed)
        print("between/within:", pooled["between_over_within_median"],
            " MC floor 1/N:", 1.0 / pooled["n_particles_per_chain"])