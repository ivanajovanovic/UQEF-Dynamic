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

Usage:
    python -m uqef_dynamic.scientific_pipelines.offline_parameter_transform \\
        --working-dir  <run folder written by main_routine> \\
        --method       mpart_joint \\
        --scope        all \\
        --max-order    2

or from python:

    from uqef_dynamic.scientific_pipelines import offline_parameter_transform as opt
    res = opt.run_offline_transform(run_dir, method="anamorphosis", scope="all")
"""

import os
import json
import argparse
import numpy as np

from uqef_dynamic.utils import transport_timeseries
from uqef_dynamic.utils import mpart_transport
from uqef_dynamic.utils import gaussian_anamorphosis


__all__ = ["run_offline_transform", "load_transform_output"]

# Canonical names come from transport_timeseries so the vocabulary is shared.
METHODS = transport_timeseries.BATCH_METHODS


def _fit_last_only(theta_last, method, max_order, param_names, verbose):
    """Map just the final timestep. Returns (z, ok, max_abs_mean, max_abs_cov)."""
    if method == "anamorphosis":
        fitted = gaussian_anamorphosis.fit_anamorphosis(theta_last, param_names=param_names)
        z = fitted.forward(theta_last)
        ok = bool(np.isfinite(z).all())
    elif method == "mpart_1d":
        z = np.empty_like(theta_last)
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
                      standard_parameter_samples_<method>.npz

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

    d = np.load(samples_file, allow_pickle=True)
    theta = np.asarray(d["theta"], dtype=np.float64)
    qoi = np.asarray(d["qoi"], dtype=np.float64)
    dates = [str(x) for x in d["dates"]]
    names = [str(x) for x in d["param_names"]]
    lower, upper = np.asarray(d["param_lower"]), np.asarray(d["param_upper"])

    if verbose:
        print(f"Read {theta.shape[0]} dates x {theta.shape[1]} particles x "
              f"{theta.shape[2]} params from {samples_file}")
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
        method=method, scope=scope, max_order=max_order, stride=stride)

    summary = {
        "working_dir": working_dir, "samples_file": samples_file,
        "output_file": out_path, "configuration_file": configuration_file,
        "method": method, "scope": scope, "max_order": max_order, "stride": stride,
        "n_dates_mapped": int(len(dates_out)),
        "n_particles": int(theta.shape[1]), "n_params": int(theta.shape[2]),
        "param_names": names,
        "n_converged": int(np.sum(ok)),
        "median_max_abs_mean": float(np.nanmedian(m_mean[ok])) if np.any(ok) else None,
        "median_max_abs_cov_minus_I": float(np.nanmedian(m_cov[ok])) if np.any(ok) else None,
        "median_max_offdiag_corr_before": float(np.nanmedian(corr_before)),
        "median_max_offdiag_corr_after": float(np.nanmedian(corr_after)),
    }
    summary_path = os.path.join(out_dir, f"transform_summary_{method}.json")
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
        description="Offline transformation of particle-filter posterior parameter "
                    "samples to a standard Gaussian space, ready for PCE.")
    p.add_argument("--working-dir", required=True,
                   help="run folder containing posterior_parameter_samples.npz")
    p.add_argument("--out-dir", default=None, help="default: the working dir")
    p.add_argument("--method", default="mpart_joint", choices=list(METHODS))
    p.add_argument("--scope", default="all", choices=["all", "last"])
    p.add_argument("--max-order", type=int, default=2)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--configuration-file", default=None,
                   help="recorded in the summary for provenance")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    run_offline_transform(a.working_dir, out_dir=a.out_dir, method=a.method,
                          scope=a.scope, max_order=a.max_order, stride=a.stride,
                          n_workers=a.n_workers,
                          configuration_file=a.configuration_file,
                          verbose=not a.quiet)


if __name__ == "__main__":
    _cli()
