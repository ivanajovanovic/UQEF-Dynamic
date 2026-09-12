"""Offline transformation of particle-filter posteriors to a standard Gaussian.

Reads what a particle_filtering_pipeline run saved and maps the posterior
parameter samples into a standard-Gaussian reference space, writing everything a
polynomial chaos expansion needs later. Nothing here re-runs the model or the
filter, so map settings can be changed freely without re-filtering.

Three mappings, differing in what they can and cannot do:

  "mpart_joint"   joint triangular transport map over all parameters at once.
                  The only option that WHITENS the cross-correlations between
                  parameters. Most expensive; struggles when the posterior is
                  spiky or heavily clustered, and a higher order can make that
                  worse rather than better.
  "mpart_1d"      one independent 1-D transport map per parameter. Far fewer
                  coefficients and no triangular conditioning issues, but it
                  leaves cross-correlations untouched.
  "anamorphosis"  rank-based marginal transform, z = G^-1(F(y)) (Fan et al. 2016).
                  Fastest by a wide margin, cannot fail to converge, and is
                  indifferent to marginal shape - but also leaves correlations.

A Hermite PCE assumes jointly independent standard normal inputs. Only
"mpart_joint" moves toward that; the other two give standard normal MARGINALS.
Whether the residual correlation matters depends on the PCE order and on how
correlated the posterior actually is - the summary reports it either way.

Scope:
  "all"   one mapping per timestep  -> z of shape (n_dates, n_particles, n_params)
  "last"  the final timestep only   -> z of shape (1, n_particles, n_params)

The second stage fits one PCE per timestep by regression on those samples,
Q_i(t) ~ sum_j c_j(t) Phi_j(z_i(t)), giving per-date coefficients, moments and
Sobol indices. Both stages parallelise across timesteps.

Tasks, selected with `task`:
  "transform"      map the posterior only, writing standard_parameter_samples_*.npz
  "pce"            fit PCEs on an EXISTING saved transform, read back from disk
  "transform+pce"  both, feeding the transform straight into the PCE

Read R2 in the PCE summary before anything else: it says whether the parameters
actually drive the output spread at each date. When it is near zero the surrogate
has fitted noise, and the Sobol indices describe only that small explained
fraction rather than a meaningful sensitivity.

ALIGNMENT: the transform stage reads theta/qoi via
transport_timeseries.load_aligned_theta_qoi, which pairs each theta with the Q
it actually produced (theta_used<->qoi on a fixed-pipeline run; the shifted
theta[:-1]<->qoi[1:] fallback, dropping one date at each end, on an older file
without theta_used). See that function's docstring for the full derivation of
why posterior_parameter_samples.npz's own same-index theta/qoi are not already
a matching pair. The PCE stage itself needs no such handling — it fits on
whatever (z, qoi) the transform stage already wrote, which is aligned by
construction.

Usage:
    python -m uqef_dynamic.scientific_pipelines.offline_parameter_transform_and_pce_learning \\
        --working-dir  <run folder written by main_routine> \\
        --task         transform+pce \\
        --method       mpart_joint \\
        --scope        all \\
        --max-order    2 \\
        --pce-order    2

or from python:

    from uqef_dynamic.scientific_pipelines import (
        offline_parameter_transform_and_pce_learning as opl)

    opl.run_offline(run_dir, task="transform", method="anamorphosis")
    opl.run_offline(run_dir, task="pce",
                    transform_file=".../standard_parameter_samples_anamorphosis.npz")
    opl.run_offline(run_dir, task="transform+pce", method="anamorphosis", pce_order=3)
"""

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import chaospy as cp

import time
import multiprocessing
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpoly")
# warnings.filterwarnings("ignore")

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from uqef_dynamic.utils import transport_timeseries
from uqef_dynamic.utils import mpart_transport
from uqef_dynamic.utils import gaussian_anamorphosis
from uqef_dynamic.utils import utility


__all__ = ["run_offline", "run_offline_transform", "run_pce_learning",
           "load_transform_output", "load_pce_output", "TASKS", "METHODS"]

# Canonical names come from transport_timeseries so the vocabulary is shared.
METHODS = transport_timeseries.BATCH_METHODS

# What run_offline can be asked to do.
TASKS = ("transform", "pce", "transform+pce")


def _stem_from_name(name):
    """Strip the .npz suffix and the shared prefix, leaving the part that
    distinguishes one run from another (e.g. "anamorphosis_stride10"). Output
    files are named from this so runs of the same method cannot overwrite
    each other."""
    stem = name[:-4] if name.lower().endswith(".npz") else name
    for prefix in ("standard_parameter_samples_", "pce_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem


def _fit_last_only(theta_last, method, max_order, param_names, verbose):
    """Map just the final timestep. Returns (z, ok, max_abs_mean, max_abs_cov)."""
    if method == "anamorphosis":
        fitted = gaussian_anamorphosis.fit_anamorphosis(theta_last, param_names=param_names)
        z = fitted.forward(theta_last)
        ok = bool(np.isfinite(z).all())
    elif method == "mpart_1d":
        # dtype=float64 explicitly: theta_last may now be float32 (see the note
        # in run_offline_transform), but f1.forward computes in float64, so
        # empty_like(theta_last) without this would silently truncate z.
        z = np.empty_like(theta_last, dtype=np.float64)
        ok = True
        for j in range(theta_last.shape[1]):
            f1 = mpart_transport.fit_transport_map(theta_last[:, [j]],
                                                   max_order=max_order, verbose=False)
            z[:, [j]] = f1.forward(theta_last[:, [j]])
            ok = ok and bool(getattr(f1.optimizer, "success", True))
        ok = ok and bool(np.isfinite(z).all())
    else:
        fitted = mpart_transport.fit_transport_map(theta_last, max_order=max_order,
                                                   param_names=param_names,
                                                   verbose=verbose)
        z = fitted.forward(theta_last)
        ok = bool(getattr(fitted.optimizer, "success", True)) and bool(np.isfinite(z).all())
    mean = z.mean(axis=0)
    cov = np.atleast_2d(np.cov(z, rowvar=False))
    n = theta_last.shape[1]
    return z, ok, float(np.max(np.abs(mean))), float(np.max(np.abs(cov - np.eye(n))))


def run_offline_transform(working_dir, out_dir=None, method="mpart_joint",
                          scope="all", max_order=2, stride=1, n_workers=None,
                          configuration_file=None, out_name=None, verbose=True):
    """Map a saved filtering posterior to a standard Gaussian space.

    Args:
        working_dir:  run folder containing posterior_parameter_samples.npz
                      (written when save_posterior_parameter_samples=True).
        out_dir:      where to write; defaults to working_dir.
        method:       one of METHODS.
        scope:        "all" (every timestep) or "last" (final timestep only).
        max_order:    polynomial order for the mpart methods; ignored by
                      "anamorphosis".
        stride:       with scope="all", map every stride-th date. Useful to probe
                      cost and quality before committing to the full record.
        n_workers:    processes for scope="all"; None -> os.cpu_count().
        configuration_file: optional path recorded in the summary for provenance.
        out_name:     output .npz name; defaults to
                      standard_parameter_samples_<method>.npz. The JSON summary is
                      named to match (transform_summary_<stem>.json), so running the
                      same method twice with different out_name — say GA at stride 1
                      and stride 10 — keeps both summaries rather than overwriting.

    Returns:
        dict with z, dates, param_names, ok, diagnostics and the output paths.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    if scope not in ("all", "last"):
        raise ValueError(f"scope must be 'all' or 'last', got {scope!r}.")
    if method.startswith("mpart") and not mpart_transport.is_available():
        raise ImportError(
            f"method={method!r} needs MParT. Install with:\n"
            "  export CMAKE_POLICY_VERSION_MINIMUM=3.5\n"
            "  pip install mpart --no-cache-dir\n"
            "or use method='anamorphosis', which has no such dependency.")

    working_dir = str(working_dir)
    samples_file = os.path.join(working_dir, "posterior_parameter_samples.npz")
    if not os.path.isfile(samples_file):
        raise FileNotFoundError(
            f"{samples_file} not found. Re-run the filter with "
            "save_posterior_parameter_samples=True.")
    out_dir = str(out_dir or working_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_name = out_name or f"standard_parameter_samples_{method}.npz"
    out_path = os.path.join(out_dir, out_name)

    # load_aligned_theta_qoi returns a genuinely matching (theta, qoi) pair:
    # theta_used<->qoi directly on a fixed run, or the shifted theta[:-1]<->
    # qoi[1:] fallback on an older, pre-fix file — see its docstring. Every
    # consumer of posterior_parameter_samples.npz in this module goes through
    # it so the alignment logic lives in exactly one place.
    aligned = transport_timeseries.load_aligned_theta_qoi(samples_file, verbose=verbose)
    # Keep theta in whatever dtype it was saved with (float32 when the pipeline
    # ran with save_theta_float32=True). Every consumer below - mpart_transport,
    # transport_timeseries.map_timesteps, fit_anamorphosis - casts to float64
    # internally itself, so upcasting here only doubles RAM and doubles the
    # theta copy written into the output .npz for no benefit.
    theta = np.asarray(aligned["theta"])
    qoi = np.asarray(aligned["qoi"], dtype=np.float64)
    dates = aligned["dates"]
    names = aligned["param_names"]
    lower, upper = aligned["param_lower"], aligned["param_upper"]
    pairing = aligned["pairing"]

    if verbose:
        print(f"Read {theta.shape[0]} dates x {theta.shape[1]} particles x "
              f"{theta.shape[2]} params from {samples_file} (pairing={pairing!r})")
        print(f"Method {method!r}, scope {scope!r}"
              + (f", order {max_order}" if method.startswith("mpart") else ""))

    if scope == "last":
        z, ok, m_mean, m_cov = _fit_last_only(theta[-1], method, max_order,
                                              names, verbose)
        z = z[np.newaxis, ...]
        theta_out, qoi_out = theta[-1:][...], qoi[-1:]
        dates_out = dates[-1:]
        ok = np.array([ok]); m_mean = np.array([m_mean]); m_cov = np.array([m_cov])
    else:
        r = transport_timeseries.map_timesteps(
            samples_file, out_path=os.path.join(out_dir, "_tmp_map.npz"),
            max_order=max_order, stride=stride, n_workers=n_workers,
            verbose=verbose, method=method)
        z = r["z"]; ok = r["ok"]; m_mean = r["max_abs_mean"]; m_cov = r["max_abs_cov"]
        dates_out = r["dates"]
        keep = [dates.index(dt) for dt in dates_out]
        theta_out, qoi_out = theta[keep], qoi[keep]
        tmp = os.path.join(out_dir, "_tmp_map.npz")
        if os.path.isfile(tmp):
            os.remove(tmp)

    # Residual cross-correlation is the number that says whether the PCE's
    # independence assumption is respected.
    def _max_offdiag(a):
        if a.shape[0] < 3:
            return np.nan
        c = np.atleast_2d(np.corrcoef(a, rowvar=False))
        return float(np.max(np.abs(c - np.eye(c.shape[0]))))

    corr_before = np.array([_max_offdiag(theta_out[k]) for k in range(len(dates_out))])
    corr_after = np.array([_max_offdiag(z[k]) if ok[k] else np.nan
                           for k in range(len(dates_out))])

    # TODO maybe this won't be needed one building a PCE comes here, but for now it's a convenient way to get the transformed samples out of the run folder.
    np.savez_compressed(
        out_path, z=z, theta=theta_out, qoi=qoi_out,
        dates=np.array(dates_out, dtype=object),
        param_names=np.array(names, dtype=object),
        param_lower=lower, param_upper=upper,
        ok=ok, max_abs_mean=m_mean, max_abs_cov=m_cov,
        max_offdiag_corr_before=corr_before, max_offdiag_corr_after=corr_after,
        method=method, scope=scope, max_order=max_order, stride=stride,
        pairing=pairing)

    summary = {
        "working_dir": working_dir, "samples_file": samples_file,
        "output_file": out_path, "configuration_file": configuration_file,
        "method": method, "scope": scope, "max_order": max_order, "stride": stride,
        "pairing": pairing,
        "n_dates_mapped": int(len(dates_out)),
        "n_particles": int(theta.shape[1]), "n_params": int(theta.shape[2]),
        "param_names": names,
        "n_converged": int(np.sum(ok)),
        "median_max_abs_mean": float(np.nanmedian(m_mean[ok])) if np.any(ok) else None,
        "median_max_abs_cov_minus_I": float(np.nanmedian(m_cov[ok])) if np.any(ok) else None,
        "median_max_offdiag_corr_before": float(np.nanmedian(corr_before)),
        "median_max_offdiag_corr_after": float(np.nanmedian(corr_after)),
    }
    # Derive the summary name from out_name so it cannot collide when the same
    # method is run more than once (e.g. GA at stride 1 and stride 10). Naming it
    # after the method alone meant the second run silently overwrote the first
    # summary while both .npz files survived.
    summary_path = os.path.join(out_dir,
                                f"transform_summary_{_stem_from_name(out_name)}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"  converged {summary['n_converged']}/{len(dates_out)}")
        print(f"  |mean(z)|max      median {summary['median_max_abs_mean']}")
        print(f"  |cov(z)-I|max     median {summary['median_max_abs_cov_minus_I']}")
        print(f"  max|offdiag corr| {summary['median_max_offdiag_corr_before']:.4f} "
              f"-> {summary['median_max_offdiag_corr_after']:.4f}"
              + ("  (joint map whitens)" if method == "mpart_joint"
                 else "  (marginal transform: correlations left as-is)"))
        print(f"  -> {out_path}")
        print(f"  -> {summary_path}")

    return {"z": z, "theta": theta_out, "qoi": qoi_out, "dates": dates_out,
            "param_names": names, "ok": ok, "summary": summary,
            "out_path": out_path, "summary_path": summary_path}


# ---------------------------------------------------------------------------
# PCE learning
# ---------------------------------------------------------------------------
#
# One PCE per timestep, fitted by regression on the transformed samples:
#
#     Q_i(t)  ~  sum_j  c_j(t) * Phi_j( z_i(t) )
#
# with z the standard-Gaussian parameter samples from the transform step and Q
# the particle streamflow at that date. The basis is the normed (orthonormal)
# Hermite expansion, which is what makes the moments below exact.
#
# Statistics come from the COEFFICIENTS, not from cp.E / cp.Var. Two reasons.
# For an orthonormal basis under N(0, I) with Phi_0 = 1 the identities
#
#     E[PCE] = c_0        Var[PCE] = sum_{j>0} c_j^2
#
# are exact and essentially free, whereas cp.E integrates numerically. These are
# POPULATION quantities under N(0, I), the same thing cp.Var means.
#
# In this environment (numpy 2.4 with numpoly 1.2.14) anything routed through
# dist.mom() raises TypeError, because numpoly still calls numpy.reshape with
# the `newshape` keyword that numpy 2.1 removed. cp.FirstOrderSobol and
# cp.TotalOrderSobol are coefficient-based and unaffected, so they are used
# directly.
#
# Two ways to get the coefficients, matching parallel_statistics.py /
# time_dependent_statistics.py's regression/quadrature split:
#
#   regression=True (default)   cp.fit_regression on the particles directly.
#   regression=False            cp.fit_quadrature, treating the particles as
#                                quadrature nodes with `weights_quad` (default:
#                                uniform 1/N, since a SIR filter's particles are
#                                already equally weighted after resampling).
#
# The two are NOT interchangeable in quality here. fit_quadrature is exact
# pseudo-spectral projection when nodes/weights form a genuine quadrature rule
# (e.g. a Gauss-Hermite tensor grid); with scattered particle samples and
# uniform weights it degrades to a crude Monte Carlo estimate of each
# coefficient, c_j = E[Q * Phi_j], which is unbiased but has much higher
# variance per coefficient than a least-squares fit on the same points -
# measured on this ensemble, order-2 quadrature gave R2 = -2.45 against
# regression's 0.02 at the same date. Prefer regression unless `weights_quad`
# carries genuine quadrature weights.

# Populated once per worker process by _pce_init, so the polynomial expansion is
# built n_workers times rather than pickled into all n_dates tasks (at order 3
# the expansion is ~120 KB, which would dominate the transfer cost).
_PCE_CTX = {}


def _pce_init(n_params, order, regression_model_type, poly_rule: str = 'three_terms_recurrence', cross_truncation: float = 1.0, poly_normed: bool = True, compute_sobol: bool = False, regression: bool = True):
    dist = cp.J(*[cp.Normal(0, 1) for _ in range(n_params)])
    # Distinct Normal instances are required: cp.J(*[cp.Normal(0,1)] * n) passes
    # the SAME object n times, which chaospy reads as a stochastically dependent
    # joint and rejects.
    # expansion, _ = utility.generate_polynomial_expansion(dist, order, rule=poly_rule, poly_normed=poly_normed, cross_truncation=cross_truncation)
    expansion, norms = cp.generate_expansion(
            order=order, dist=dist, rule=poly_rule, normed=poly_normed,
            graded=True, reverse=True, cross_truncation=cross_truncation, retall=True)
    _PCE_CTX["expansion"] = expansion
    _PCE_CTX["model"] = utility.generate_regression_model(regression_model_type, expansion)
    _PCE_CTX["compute_sobol"] = compute_sobol
    _PCE_CTX["norms"] = norms
    _PCE_CTX["regression"] = regression
    return expansion


def _pce_one_date(task):
    """Fit one timestep. Module-level and self-contained so it survives the
    spawn-based pickling macOS uses. Returns a tuple keyed by the date index."""
    k, z_k, q_k, w_k = task
    expansion = _PCE_CTX["expansion"]
    n_terms, n_params = len(expansion), z_k.shape[1]
    try:
        # abscissas/nodes want (n_params, n_samples), z_k is (n_samples, n_params).
        if _PCE_CTX["regression"]:
            gpce, coeff = cp.fit_regression(
                polynomials=expansion, abscissas=z_k.T, evals=q_k,
                retall=True, model=_PCE_CTX["model"])
        else:
            gpce, coeff = cp.fit_quadrature(
                orth=expansion, nodes=z_k.T, weights=w_k, solves=q_k,
                retall=True, norms=_PCE_CTX["norms"])
        coeff = np.asarray(coeff, dtype=np.float64).ravel()
        pred = np.asarray(gpce(*z_k.T), dtype=np.float64)
        if not (np.all(np.isfinite(coeff)) and np.all(np.isfinite(pred))):
            raise FloatingPointError("non-finite PCE coefficients or predictions")

        E = float(coeff[0])
        var = float(np.sum(coeff[1:] ** 2))
        resid = q_k - pred
        denom = float(np.sum((q_k - q_k.mean()) ** 2))
        r2 = float(1.0 - np.sum(resid ** 2) / denom) if denom > 0 else np.nan
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        # Percentiles of the surrogate response. Taken over the fitted sample
        # rather than via cp.Perc, which is on the broken numpoly path; since
        # z is standard normal by construction the two agree closely.
        p10, p90 = (float(x) for x in np.percentile(pred, [10, 90]))

        if _PCE_CTX["compute_sobol"]:
            sm = np.asarray(cp.FirstOrderSobol(expansion, coeff), dtype=np.float64).ravel()
            st = np.asarray(cp.TotalOrderSobol(expansion, coeff), dtype=np.float64).ravel()
        else:
            sm = st = np.full(n_params, np.nan)
        return k, coeff, E, var, r2, rmse, p10, p90, sm, st, True
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [pce] date index {k} failed: {type(exc).__name__}: {exc}")
        nan_p = np.full(n_params, np.nan)
        return (k, np.full(n_terms, np.nan), np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, nan_p, nan_p, False)


def run_pce_learning(transform_path, out_dir=None, method=None, pce_order=2,
                     regression_model_type=None, cross_truncation=1.0, compute_sobol=True,
                     regression=True, weights_quad=None,
                     n_workers=None, out_name=None, configuration_file=None,
                     verbose=True):
    """Fit one PCE per timestep on an already-transformed posterior.

    Args:
        transform_path: the .npz written by the transform step, or the folder
                        holding it (then `method` picks the file).
        out_dir:        where to write; defaults to the transform file's folder.
        pce_order:      total polynomial order.
        regression_model_type: None for plain least squares, or "ols" / "lars"
                        to go through sklearn (see utility.generate_regression_model).
                        "lars" is worth trying when terms outnumber useful signal.
        cross_truncation: float, optional
            Cross-truncation parameter for the polynomial expansion.
        compute_sobol:  first- and total-order Sobol indices per date.
        regression:     True (default) fits by least squares (cp.fit_regression).
                        False fits by pseudo-spectral projection (cp.fit_quadrature),
                        treating the particles as quadrature nodes weighted by
                        `weights_quad`. See the module note above the PCE section:
                        with scattered particle samples rather than a genuine
                        quadrature rule this is a much noisier estimator of each
                        coefficient, so leave this True unless `weights_quad`
                        carries real quadrature weights.
        weights_quad:   only used when regression=False. None (default) uses
                        uniform weights 1/n_particles per date, valid because a
                        SIR filter's saved particles are already equally weighted
                        after resampling. Otherwise an array of shape
                        (n_particles,) - applied at every date - or
                        (n_dates, n_particles) for per-date weights.
        n_workers:      processes; None -> os.cpu_count(). Each date is one task.
        out_name:       output .npz name; defaults to pce_<stem>.npz, where the
                        stem is taken from the transform file.

    Returns:
        dict with coefficients, per-date statistics, diagnostics and paths.

    Note:
        R2 is the number to read first. The PCE maps parameters to output, so a
        low R2 means the posterior parameters barely drive the output spread at
        that date — in which case the Sobol indices describe only that small
        explained fraction and should not be interpreted as sensitivities.
    """
    if pce_order < 1:
        raise ValueError(f"pce_order must be >= 1, got {pce_order}.")

    path = str(transform_path)
    if os.path.isdir(path):
        name = (f"standard_parameter_samples_{method}.npz" if method
                else "standard_parameter_samples.npz")
        path = os.path.join(path, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run the transform step first "
            "(task='transform' or task='transform+pce').")

    d = load_transform_output(path)
    z = np.asarray(d["z"], dtype=np.float64)
    qoi = np.asarray(d["qoi"], dtype=np.float64)
    dates = [str(x) for x in d["dates"]]
    names = [str(x) for x in d["param_names"]]
    ok_transform = np.asarray(d["ok"]).astype(bool)
    n_dates, n_particles, n_params = z.shape

    out_dir = str(out_dir or os.path.dirname(path))
    os.makedirs(out_dir, exist_ok=True)
    stem = _stem_from_name(os.path.basename(path))
    out_name = out_name or f"pce_{stem}.npz"
    out_path = os.path.join(out_dir, out_name)

    # Build the basis first, then read its true size off it: math.comb(pce_order
    # + n_params, n_params) only predicts the term count at cross_truncation=1.0
    # (full total-degree expansion); at cross_truncation<1 chaospy drops some
    # interaction terms, so that formula overcounts (e.g. order 3, 7 params,
    # cross_truncation=0.7: predicts 120, actual is 43) and would wrongly
    # reject - or assert-fail against - a perfectly fittable request.
    expansion = _pce_init(
        n_params, order=pce_order, regression_model_type=regression_model_type,
        cross_truncation=cross_truncation, poly_normed=True,
        compute_sobol=compute_sobol, regression=regression)
    n_terms = len(expansion)
    if n_terms > n_particles:
        raise ValueError(
            f"order {pce_order} over {n_params} parameters (cross_truncation="
            f"{cross_truncation}) needs {n_terms} terms but only {n_particles} "
            "particles are available; the regression is underdetermined. Lower "
            "pce_order, lower cross_truncation, or use regression_model_type='lars'.")

    # Per-date quadrature weights (unused when regression=True, built regardless
    # since it is cheap and keeps the task tuple shape uniform). Default: uniform
    # 1/n_particles, valid because these particles are already equally weighted
    # after the filter's systematic resampling.
    if weights_quad is None:
        weights_per_date = np.full((n_dates, n_particles), 1.0 / n_particles)
    else:
        w = np.asarray(weights_quad, dtype=np.float64)
        if w.ndim == 1:
            if w.shape[0] != n_particles:
                raise ValueError(
                    f"weights_quad has {w.shape[0]} entries but there are "
                    f"{n_particles} particles per date.")
            weights_per_date = np.tile(w, (n_dates, 1))
        elif w.ndim == 2:
            if w.shape != (n_dates, n_particles):
                raise ValueError(
                    f"weights_quad shape {w.shape} does not match "
                    f"(n_dates={n_dates}, n_particles={n_particles}).")
            weights_per_date = w
        else:
            raise ValueError(f"weights_quad must be 1-D or 2-D, got shape {w.shape}.")

    if verbose:
        print(f"PCE order {pce_order}: {n_terms} terms, {n_particles} samples/date, "
              f"{n_dates} dates from {os.path.basename(path)}")
        if n_terms > n_particles // 10:
            print(f"  note: {n_terms} terms against {n_particles} samples is a thin "
                  "ratio; check R2 for overfitting")

    # Dates whose transform failed carry no usable z, so they are not fitted.
    idx = [k for k in range(n_dates) if ok_transform[k]]
    if not idx:
        raise ValueError("No timestep has a converged transform; nothing to fit.")
    if verbose and len(idx) < n_dates:
        print(f"  skipping {n_dates - len(idx)} date(s) whose transform did not converge")

    coeffs = np.full((n_dates, n_terms), np.nan)
    E = np.full(n_dates, np.nan); var = np.full(n_dates, np.nan)
    r2 = np.full(n_dates, np.nan); rmse = np.full(n_dates, np.nan)
    p10 = np.full(n_dates, np.nan); p90 = np.full(n_dates, np.nan)
    sobol_m = np.full((n_dates, n_params), np.nan)
    sobol_t = np.full((n_dates, n_params), np.nan)
    ok = np.zeros(n_dates, dtype=bool)

    def _collect(results):
        for k, c, e_, v_, r_, rm_, p1_, p9_, sm_, st_, good in results:
            coeffs[k] = c; E[k] = e_; var[k] = v_; r2[k] = r_; rmse[k] = rm_
            p10[k] = p1_; p90[k] = p9_; sobol_m[k] = sm_; sobol_t[k] = st_
            ok[k] = good

    tasks = [(k, z[k], qoi[k], weights_per_date[k]) for k in idx]
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(tasks)))
    # Each worker re-imports chaospy and mpart, measured at 2-4 s, while one
    # order-2 fit is ~15 ms. So a pool only pays once the serial cost is well
    # past that. The estimate below is calibrated on 7 parameters / 2000
    # particles: order 2 over 366 dates is ~8 s serial (a pool saves nothing),
    # order 3 over the same dates is ~44 s (a pool halves it).
    est_serial = len(tasks) * n_terms * 6e-4
    if est_serial < 15.0:
        n_workers = 1

    t0 = time.perf_counter()
    if verbose:
        why = ("" if n_workers > 1 else
               f" (serial: ~{est_serial:.0f}s of work is below the pool's startup cost)")
        print(f"  fitting {len(tasks)} dates on {n_workers} worker(s), "
              f"via {'regression' if regression else 'quadrature'}{why}")
    if n_workers == 1:
        _collect(map(_pce_one_date, tasks))
    else:
        try:
            # Positional, in _pce_init's declared order - initializer/initargs is
            # always called positionally, so this must track that signature.
            with ProcessPoolExecutor(
                    max_workers=n_workers, initializer=_pce_init,
                    initargs=(n_params, pce_order, regression_model_type,
                              'three_terms_recurrence', cross_truncation, True,
                              compute_sobol, regression)) as ex:
                _collect(ex.map(_pce_one_date, tasks, chunksize=1))
        except BrokenProcessPool:
            # macOS spawns workers by re-importing __main__; that fails when the
            # caller is an interactive session or a stdin script. Serial is slower
            # but always available.
            print("  process pool unavailable (spawn could not re-import "
                  "__main__); falling back to serial")
            _collect(map(_pce_one_date, tasks))
    elapsed = time.perf_counter() - t0

    std = np.sqrt(var)
    n_ok = int(np.sum(ok))
    med_r2 = float(np.nanmedian(r2[ok])) if n_ok else None

    np.savez_compressed(
        out_path,
        gpce_coeff=coeffs.astype(np.float32), E=E, Var=var, StdDev=std,
        Sobol_m=sobol_m, Sobol_t=sobol_t, R2=r2, RMSE=rmse, P10=p10, P90=p90,
        ok=ok, dates=np.array(dates, dtype=object),
        param_names=np.array(names, dtype=object),
        qoi_mean=qoi.mean(axis=1), qoi_std=qoi.std(axis=1),
        pce_order=pce_order, n_terms=n_terms, n_particles=n_particles,
        regression_model_type=str(regression_model_type),
        fit_method=("regression" if regression else "quadrature"),
        transform_file=path)

    summary = {
        "transform_file": path, "output_file": out_path,
        "configuration_file": configuration_file,
        "pce_order": pce_order,
        "cross_truncation": cross_truncation,
        "n_terms": int(n_terms),
        "fit_method": "regression" if regression else "quadrature",
        "regression_model_type": regression_model_type,
        "n_dates": int(n_dates), "n_dates_fitted": int(len(tasks)),
        "n_converged": n_ok,
        "n_particles": int(n_particles), "n_params": int(n_params),
        "param_names": names,
        "median_R2": med_r2,
        "min_R2": float(np.nanmin(r2[ok])) if n_ok else None,
        "max_R2": float(np.nanmax(r2[ok])) if n_ok else None,
        "n_dates_R2_above_0.5": int(np.sum(r2[ok] > 0.5)) if n_ok else 0,
        "median_RMSE": float(np.nanmedian(rmse[ok])) if n_ok else None,
        "median_qoi_std": float(np.median(qoi.std(axis=1))),
        "elapsed_seconds": round(elapsed, 2),
        "n_workers": int(n_workers),
    }
    if compute_sobol and n_ok:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            summary["median_Sobol_m"] = {
                nm: float(np.nanmedian(sobol_m[ok, j])) for j, nm in enumerate(names)}
            summary["median_Sobol_t"] = {
                nm: float(np.nanmedian(sobol_t[ok, j])) for j, nm in enumerate(names)}

    summary_path = os.path.join(out_dir, f"pce_summary_{_stem_from_name(out_name)}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"  fitted {n_ok}/{len(tasks)} dates in {elapsed:.1f}s")
        if med_r2 is not None:
            print(f"  R2  median {med_r2:.4f}  (min {summary['min_R2']:.4f}, "
                  f"max {summary['max_R2']:.4f}); "
                  f"{summary['n_dates_R2_above_0.5']}/{n_ok} dates above 0.5")
            if med_r2 < 0.1:
                print(f"  !! median R2 {med_r2:.3f}: the parameters explain almost "
                      "none of the particle spread in the output, so this surrogate\n"
                      "     carries little signal and the Sobol indices below "
                      "describe only that small explained fraction.\n")
        if compute_sobol and n_ok:
            top = sorted(summary["median_Sobol_t"].items(), key=lambda kv: -kv[1])[:3]
            print("  median Sobol_t top: "
                  + ", ".join(f"{k}={v:.3f}" for k, v in top))
        print(f"  -> {out_path}")
        print(f"  -> {summary_path}")

    return {"gpce_coeff": coeffs, "E": E, "Var": var, "StdDev": std,
            "Sobol_m": sobol_m, "Sobol_t": sobol_t, "R2": r2, "RMSE": rmse,
            "P10": p10, "P90": p90, "ok": ok, "dates": dates,
            "param_names": names, "summary": summary,
            "out_path": out_path, "summary_path": summary_path}


def run_offline(working_dir, task="transform+pce", out_dir=None,
                method="mpart_joint", scope="all", max_order=2, stride=1,
                pce_order=2, regression_model_type=None, cross_truncation=1.0, compute_sobol=True,
                regression=True, weights_quad=None,
                transform_file=None, out_name=None, pce_out_name=None,
                n_workers=None, configuration_file=None, verbose=True):
    """Run the offline stage: transform, PCE, or both.

    Args:
        working_dir: run folder written by particle_filtering_pipeline.main_routine.
        task:        "transform"     map the posterior to standard Gaussian only;
                     "pce"           fit PCEs on an EXISTING transform, read from
                                     `transform_file` (or derived from `method`);
                     "transform+pce" both, feeding the transform straight into
                                     the PCE without a second read.
        transform_file: only for task="pce" — which saved .npz to read. Defaults
                     to standard_parameter_samples_<method>.npz in out_dir.

    Everything else is passed through to run_offline_transform / run_pce_learning;
    both stages parallelise across timesteps.

    Returns:
        dict with "task", and "transform" / "pce" entries for the stages that ran.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}.")

    results = {"task": task}
    transform_result = None

    if task in ("transform", "transform+pce"):
        if verbose:
            print("=" * 78)
            print(f"[1] parameter transformation  ({method}, scope={scope}, stride={stride})")
        transform_result = run_offline_transform(
            working_dir, out_dir=out_dir, method=method, scope=scope,
            max_order=max_order, stride=stride, n_workers=n_workers,
            configuration_file=configuration_file, out_name=out_name,
            verbose=verbose)
        results["transform"] = transform_result

    if task in ("pce", "transform+pce"):
        if transform_result is not None:
            src = transform_result["out_path"]
        elif transform_file is not None:
            src = str(transform_file)
        else:
            # Default to what a transform run with these settings would have written.
            src = os.path.join(str(out_dir or working_dir),
                               out_name or f"standard_parameter_samples_{method}.npz")
        if verbose:
            print("=" * 78)
            print(f"[2] PCE learning  (order {pce_order}) from {os.path.basename(src)}")
        results["pce"] = run_pce_learning(
            src, out_dir=out_dir, method=method, pce_order=pce_order,
            regression_model_type=regression_model_type,
            cross_truncation=cross_truncation,
            compute_sobol=compute_sobol, regression=regression,
            weights_quad=weights_quad, n_workers=n_workers,
            out_name=pce_out_name, configuration_file=configuration_file,
            verbose=verbose)

    return results


def load_pce_output(path, stem=None):
    """Load a PCE output .npz into a dict of arrays. `path` may be the file or
    its directory (with `stem` naming which file to pick up)."""
    if os.path.isdir(str(path)):
        path = os.path.join(str(path), f"pce_{stem}.npz" if stem else "pce.npz")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_transform_output(path, method=None):
    """Load an output .npz into a dict of arrays. `path` may be the file or its
    directory (with `method` naming which file to pick up)."""
    if os.path.isdir(str(path)):
        name = (f"standard_parameter_samples_{method}.npz" if method
                else "standard_parameter_samples.npz")
        path = os.path.join(str(path), name)
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _cli():
    p = argparse.ArgumentParser(
        description="Offline stage for a particle-filter run: map the posterior "
                    "parameter samples to a standard Gaussian space and/or fit a "
                    "per-timestep PCE surrogate on them.")
    p.add_argument("--working-dir", required=True,
                   help="run folder containing posterior_parameter_samples.npz")
    p.add_argument("--task", default="transform+pce", choices=list(TASKS),
                   help="transform only, pce only (reads a saved transform), or both")
    p.add_argument("--out-dir", default=None, help="default: the working dir")
    # transform stage
    p.add_argument("--method", default="mpart_joint", choices=list(METHODS))
    p.add_argument("--scope", default="all", choices=["all", "last"])
    p.add_argument("--max-order", type=int, default=2,
                   help="transport-map order (mpart methods only)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out-name", default=None,
                   help="transform output .npz name")
    # pce stage
    p.add_argument("--pce-order", type=int, default=2,
                   help="PCE total order (7 params: 36 terms at 2, 120 at 3)")
    p.add_argument("--regression-model-type", default=None,
                   choices=["ols", "lars"],
                   help="default: plain least squares inside chaospy")
    p.add_argument("--no-sobol", action="store_true",
                   help="skip Sobol indices")
    p.add_argument("--quadrature", action="store_true",
                   help="fit by pseudo-spectral projection (cp.fit_quadrature, "
                   "uniform particle weights) instead of regression. Usually "
                   "noisier on particle-filter samples - see the module docstring")
    p.add_argument("--transform-file", default=None,
                   help="task=pce: the saved transform .npz to fit on")
    p.add_argument("--pce-out-name", default=None,
                   help="PCE output .npz name")
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--configuration-file", default=None,
                   help="recorded in the summary for provenance")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    run_offline(a.working_dir, task=a.task, out_dir=a.out_dir, method=a.method,
                scope=a.scope, max_order=a.max_order, stride=a.stride,
                pce_order=a.pce_order,
                regression_model_type=a.regression_model_type,
                compute_sobol=not a.no_sobol, regression=not a.quadrature,
                transform_file=a.transform_file, out_name=a.out_name,
                pce_out_name=a.pce_out_name, n_workers=a.n_workers,
                configuration_file=a.configuration_file,
                verbose=not a.quiet)


if __name__ == "__main__":
    # _cli()
    # (method, stride, output filename)
    # D = ("data/HBV-SASK-data/particle_filtering_model_runs/banff_basin/hbvsaskmodel_7d_2000_filtering_gaussian_likelihood_heteroscedastic_one_year_Uniform/run_0")
    D = "data/HBV-SASK-data/particle_filtering_model_runs/banff_basin/hbvsaskmodel_7d_2000_filtering_gaussian_likelihood_heteroscedastic_one_year_Uniform/run_0"

    nw = multiprocessing.cpu_count()

    # JOBS = [
    # ("anamorphosis", 1,  "standard_parameter_samples_anamorphosis.npz"),
    # ("anamorphosis", 10, "standard_parameter_samples_anamorphosis_stride10.npz"),
    # ("mpart_joint",  10, "standard_parameter_samples_mpart_joint_stride10.npz"),
    # ("mpart_joint",  1, "standard_parameter_samples_mpart_joint_stride1.npz"),
    # ("mpart_1d",     10, "standard_parameter_samples_mpart_1d_stride10.npz"),
    # ]
    # for method, stride, out_name in JOBS:
    #     print("=" * 78)
    #     print(f"### {method}  stride={stride}")
    #     t0 = time.perf_counter()
    #     r = run_offline_transform(D, method=method, scope="all", stride=stride,
    #                                   max_order=2, n_workers=nw, out_name=out_name,
    #                                   configuration_file="data/configurations/configuration_hbv_sask_PF_one_year.json",
    #                                   verbose=True)
    #     s = r["summary"]
    #     print(f"  elapsed {time.perf_counter()-t0:.1f}s | {s['n_dates_mapped']} dates | "
    #           f"converged {s['n_converged']}")

    from uqef_dynamic.scientific_pipelines import designed_sample_pce as dsp

    # Designed-sample PCE, single reference date (already run - see
    # designed_pce/pce_after_particle_filter_streamflow.pdf): fit the theta<->z
    # map ONCE from the posterior at reference_date, draw 2000 designed
    # samples, invert to theta, run HBV-SASK once per sample from a common
    # 3-year-warmup state, fit one order-3 (cross_truncation=0.7) PCE per date
    # over every daily date from Nov 2004 to Oct 2005, then plot the result.
    # ~29s. Uncomment to re-run.
    #
    # out_dir = os.path.join(D, "designed_pce")
    # os.makedirs(out_dir, exist_ok=True)
    # target_dates = [str(d.date()) for d in
    #                 pd.date_range("2004-11-01", "2005-10-01", freq="1D")]
    # reference_date = "2005-04-15"
    # t0 = time.perf_counter()
    # result = dsp.run_designed_sample_pce(
    #     working_dir=D,
    #     configuration_file="data/configurations/configuration_hbv_sask_PF_one_year.json",
    #     inputModelDir="data/HBV-SASK-data",
    #     model_working_dir=out_dir,
    #     reference_date=reference_date,
    #     target_dates=target_dates,
    #     n_samples=2000, design="random", method="mpart_joint", max_order=2,
    #     pce_order=3, cross_truncation=0.7, warmup_years=3,
    #     out_dir=out_dir, n_workers=nw, seed=0, verbose=True,
    # )
    # print(f"\nTOTAL elapsed: {time.perf_counter() - t0:.1f}s")
    # dsp.plot_pce_after_particle_filter(result, working_dir=D, out_dir=out_dir)

    # =========================================================================
    # Designed-sample PCE, EVOLVING posterior (uses the PF's per-date posterior
    # directly, not one frozen reference date - see run_designed_sample_pce_evolving's
    # docstring for why: each date gets its own transport map fit on that
    # date's own PF posterior, so the design genuinely tracks how the PF's
    # belief about theta narrowed/shifted over the assimilation window, rather
    # than reusing one snapshot everywhere).


    # RUN_FULL_THREE_YEAR_EVOLVING = True

    # D_pooled = ("data/HBV-SASK-data/particle_filtering_model_runs/banff_basin/"
    #            "hbvsaskmodel_7d_2000_filtering_gaussian_likelihood_heteroscedastic_"
    #            "three_years_Uniform_10_chains")
    # cfg_three_years = "data/configurations/configuration_hbv_sask_PF_three_years.json"
    # out_dir_evolving = os.path.join(D_pooled, "designed_pce_evolving")
    # os.makedirs(out_dir_evolving, exist_ok=True)

    # if RUN_FULL_THREE_YEAR_EVOLVING:
    #     target_dates_evolving = [str(d.date()) for d in
    #                              pd.date_range("2004-10-01", "2007-10-01", freq="1D")]
    #     n_samples_evolving = 2000
    # else:
    #     target_dates_evolving = ["2005-01-15", "2005-10-01", "2006-07-01", "2007-04-01"]
    #     n_samples_evolving = 300

    # t0 = time.perf_counter()
    # result_evolving = dsp.run_designed_sample_pce_evolving(
    #     working_dir=D_pooled,
    #     configuration_file=cfg_three_years,
    #     inputModelDir="data/HBV-SASK-data",
    #     model_working_dir=out_dir_evolving,
    #     target_dates=target_dates_evolving,
    #     n_samples=n_samples_evolving, design="random", method="mpart_joint", max_order=2,
    #     pce_order=3, cross_truncation=0.7, warmup_years=3,
    #     out_dir=out_dir_evolving, n_workers=nw, seed=0, verbose=True,
    # )
    # print(f"\nTOTAL elapsed: {time.perf_counter() - t0:.1f}s")
    # dsp.plot_pce_after_particle_filter(
    #     result_evolving, working_dir=D_pooled, out_dir=out_dir_evolving,
    #     filename="pce_after_particle_filter_streamflow_evolving")

    ##################################

    D = "data/HBV-SASK-data/particle_filtering_model_runs/banff_basin/hbvsaskmodel_7d_2000_filtering_gaussian_likelihood_heteroscedastic_one_year_Uniform/run_0"
    # D = D_pooled = ("data/HBV-SASK-data/particle_filtering_model_runs/banff_basin/"
    #            "hbvsaskmodel_7d_2000_filtering_gaussian_likelihood_heteroscedastic_"
    #            "three_years_Uniform_10_chains")
    # cfg_three_years = "data/configurations/configuration_hbv_sask_PF_three_years.json"
    # out_dir_evolving = os.path.join(D_pooled, "designed_pce_evolving")
    # os.makedirs(out_dir_evolving, exist_ok=True)
    # PF-particle transform+PCE (not the designed-sample approach above) - kept
    # for reference; median R2 ~0.04, since PF particles carry per-particle
    # state alongside theta and a theta-only PCE can't fit that (see this
    # module's docstring and designed_sample_pce.py's for the diagnosis).
    result_dict = run_offline(working_dir=D, task="transform+pce",
                method="mpart_joint", scope="all", max_order=2, stride=10,
                pce_order=3, cross_truncation=0.7, regression_model_type=None, compute_sobol=True,
                out_name="standard_parameter_samples_mpart_joint_stride10.npz",
                n_workers=nw,
                configuration_file="data/configurations/configuration_hbv_sask_PF_one_year.json",
                verbose=True
                )
    s = result_dict["transform"]["summary"]
    pce_s = result_dict["pce"]["summary"]
    print(f" {s['n_dates_mapped']} dates | "
          f"converged {s['n_converged']} | "
          f"PCE converged {pce_s['n_converged']} | "
          f"PCE median_R2 {pce_s['median_R2']} ")
