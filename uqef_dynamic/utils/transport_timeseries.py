"""Per-timestep Gaussianization of particle-filter posterior parameter samples.

Runs POST-HOC on the posterior_parameter_samples.npz written by
particle_filtering_pipeline.main_routine. Everything needed is already stored
per timestep, so fitting the maps afterwards gives results identical to doing
it inside the filter loop, while keeping the filter fast, letting the map order
be changed without re-running the filter, and allowing the dates to be fitted
in parallel.

Output: standard_parameter_samples.npz containing

    z            (n_dates, n_particles, n_params)  standard-Gaussian samples
    theta        (n_dates, n_particles, n_params)  the ALIGNED inputs (see below)
    qoi          (n_dates, n_particles)            matching model outputs
    ok           (n_dates,)  bool, whether that date's map fit succeeded
    max_abs_mean (n_dates,)  |mean(z)|max          diagnostic, ~0 when good
    max_abs_cov  (n_dates,)  |cov(z) - I|max       diagnostic, ~0 when good
    pairing      str, "theta_used" or "shifted" — see load_aligned_theta_qoi

(z[k], qoi[k]) are then the regression pairs for a PCE at dates[k], with z
distributed as a standard Gaussian so a probabilist-Hermite basis applies.

ALIGNMENT — read load_aligned_theta_qoi's docstring before touching this file.
posterior_parameter_samples.npz's own "theta"/"qoi" fields are NOT a matching
(parameter, output) pair: "theta"[k] is the resampled+perturbed ensemble that
becomes the input at date k+1, so it is "theta"[k] that produced "qoi"[k+1],
not "qoi"[k]. Every function here that consumes that file goes through
load_aligned_theta_qoi so this is handled in exactly one place.
"""

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from uqef_dynamic.utils import mpart_transport
from uqef_dynamic.utils import gaussian_anamorphosis
from uqef_dynamic.utils import legacy_transport


__all__ = ["map_timesteps", "load_standard_samples", "load_aligned_theta_qoi",
           "gaussianize_parameter_samples", "METHODS", "normalize_method"]


def load_aligned_theta_qoi(samples_npz, verbose=True):
    """Load posterior_parameter_samples.npz and return a genuinely aligned
    (theta, qoi) pair: theta[k] is the parameter ensemble that PRODUCED qoi[k],
    in the same particle order, for every k.

    Two schemas are handled:

    - NEW schema (file has "theta_used", written by main_routine since the
      alignment fix): theta_used[k] was captured inside the per-particle
      model-run loop as the exact ensemble that produced qoi[k]. Used
      directly — no shift, no dates dropped.

    - OLD schema (no "theta_used", pre-fix runs): "theta"[k] is the
      RESAMPLED-AND-PERTURBED ensemble that becomes the input at date k+1, so
      "theta"[k] paired with "qoi"[k+1] is the valid correspondence, not same-
      index k with k (see particle_filtering_pipeline.py's main_routine for
      the full derivation). Falls back to theta[:-1] <-> qoi[1:], with dates
      taken from qoi's own index (dates[1:]) since that is the date the pair
      actually describes. This drops qoi at the very first date (produced by
      the initial prior draw, never saved as an array in the old schema) and
      theta at the very last date (nothing comes after it to pair with) — one
      date lost at each end, not the whole record.

    Args:
        samples_npz: path to posterior_parameter_samples.npz, or its directory.
        verbose:     print a one-line notice when the old-schema fallback fires.

    Returns:
        dict: theta (n_dates', n_particles, n_params), qoi (n_dates', n_particles),
        dates (list[str], length n_dates'), param_names, param_lower, param_upper,
        pairing ("theta_used" or "shifted").
    """
    if os.path.isdir(str(samples_npz)):
        samples_npz = os.path.join(str(samples_npz), "posterior_parameter_samples.npz")
    d = np.load(samples_npz, allow_pickle=True)
    dates_raw = [str(x) for x in d["dates"]]
    names = [str(x) for x in d["param_names"]]
    lower = np.asarray(d["param_lower"])
    upper = np.asarray(d["param_upper"])
    qoi_raw = np.asarray(d["qoi"], dtype=np.float64)

    if "theta_used" in d.files:
        theta = np.asarray(d["theta_used"])
        qoi, dates, pairing = qoi_raw, dates_raw, "theta_used"
    else:
        theta_raw = np.asarray(d["theta"])
        theta = theta_raw[:-1]
        qoi = qoi_raw[1:]
        dates = dates_raw[1:]
        pairing = "shifted"
        if verbose:
            print(f"WARNING {os.path.basename(str(samples_npz))}: no 'theta_used' "
                  f"field (written before the theta/qoi alignment fix). Falling "
                  f"back to the shifted pairing theta[:-1] <-> qoi[1:] "
                  f"({len(dates_raw)} dates -> {len(dates)} usable pairs). "
                  f"Re-run the filter to get the direct, unshifted pairing.")

    return {"theta": theta, "qoi": qoi, "dates": dates, "param_names": names,
            "param_lower": lower, "param_upper": upper, "pairing": pairing}


# One vocabulary for every entry point, so a name valid in one place cannot
# silently mean something else in another.
METHODS = ("mpart_joint", "mpart_1d", "anamorphosis", "legacy")

# Names accepted for backward compatibility / convenience, mapped to canonical ones.
_ALIASES = {"mpart": "mpart_joint", "joint": "mpart_joint",
            "1d": "mpart_1d", "mpart1d": "mpart_1d",
            "ga": "anamorphosis"}

# Only these support a per-timestep sweep; "legacy" has no batch path.
BATCH_METHODS = ("mpart_joint", "mpart_1d", "anamorphosis")


def normalize_method(method, allowed=METHODS, caller="this function"):
    """Canonicalise a method name and reject anything unrecognised.

    Previously an unknown name fell through to the mpart branch, so a typo — or
    a name valid elsewhere, like "legacy" — silently produced an MParT map while
    the saved metadata recorded the name that was asked for. Fail loudly instead.
    """
    if method is None:
        return None
    m = _ALIASES.get(str(method).strip().lower(), str(method).strip().lower())
    if m not in allowed:
        extra = ""
        if m == "legacy":
            extra = (" The bundled 'legacy' toolbox has no per-timestep path; use "
                     "gaussianize_parameter_samples for a single ensemble.")
        raise ValueError(
            f"{caller}: unknown method {method!r}. Valid: {sorted(allowed)} "
            f"(aliases: {sorted(_ALIASES)}).{extra}")
    return m


def _fit_one(args):
    """Fit one date. Returns (index, z, ok, max_abs_mean, max_abs_cov)."""
    k, theta_k, max_order, gtol, maxiter, method = args
    n_particles, n_dim = theta_k.shape
    try:
        if method == "anamorphosis":
            fitted = gaussian_anamorphosis.fit_anamorphosis(theta_k)
            z = fitted.forward(theta_k)
            ok = np.isfinite(z).all()
        elif method == "mpart_1d":
            # One independent 1-D map per parameter: far fewer coefficients and no
            # triangular conditioning issues, but it cannot whiten cross-correlations.
            z = np.empty_like(theta_k)
            ok = True
            for j in range(n_dim):
                f1 = mpart_transport.fit_transport_map(
                    theta_k[:, [j]], max_order=max_order, gtol=gtol,
                    maxiter=maxiter, verbose=False)
                z[:, [j]] = f1.forward(theta_k[:, [j]])
                ok = ok and bool(getattr(f1.optimizer, "success", True))
            ok = ok and np.isfinite(z).all()
        else:                                    # "mpart_joint" (validated upstream)
            fitted = mpart_transport.fit_transport_map(
                theta_k, max_order=max_order, gtol=gtol, maxiter=maxiter, verbose=False)
            z = fitted.forward(theta_k)
            ok = bool(getattr(fitted.optimizer, "success", True)) and np.isfinite(z).all()
        mean = z.mean(axis=0)
        cov = np.atleast_2d(np.cov(z, rowvar=False))
        return (k, z, ok, float(np.max(np.abs(mean))),
                float(np.max(np.abs(cov - np.eye(n_dim)))))
    except Exception:
        # A collapsed ensemble (zero-variance parameter) or a failed optimisation
        # should not abort the whole sweep; mark the date and carry on.
        return k, np.full((n_particles, n_dim), np.nan), False, np.nan, np.nan


def map_timesteps(samples_npz, out_path=None, max_order=2, stride=1,
                  n_workers=None, gtol=1e-3, maxiter=500, verbose=True,
                  method="mpart_joint"):
    """Fit one transport map per timestep and save the standard-Gaussian samples.

    Args:
        samples_npz: path to posterior_parameter_samples.npz (or its directory).
        out_path:    output .npz; defaults to standard_parameter_samples.npz
                     beside the input.
        max_order:   polynomial order of each map. 1 is affine and fast; 2 adds
                     skew and mild nonlinearity. Cost grows steeply with order
                     and dimension.
        stride:      fit every stride-th date. Use >1 to probe cost/quality
                     before committing to the full record.
        n_workers:   processes to use; None -> os.cpu_count(). Each date is an
                     independent fit, so this scales close to linearly.
        method:      "mpart"        -> joint triangular transport map. Whitens
                                       cross-correlations; needs mpart; struggles
                                       with spiky/clustered marginals.
                     "anamorphosis" -> rank-based marginal transform (Fan et al.
                                       2016). ~500x faster, indifferent to
                                       marginal shape, cannot fail to converge,
                                       but leaves cross-correlations untouched.
                                       max_order/gtol/maxiter are ignored.
        gtol,maxiter: optimiser controls passed through to fit_transport_map.

    theta/qoi are read via load_aligned_theta_qoi, so this always fits on a
    genuinely matching (parameter, output) pair — see that function's docstring
    for what "aligned" means and the old-schema fallback it falls back to.

    Returns:
        dict with z, dates, param_names, ok, max_abs_mean, max_abs_cov, pairing
        ("theta_used" or "shifted"), out_path.
    """
    method = normalize_method(method, BATCH_METHODS, "map_timesteps")
    aligned = load_aligned_theta_qoi(samples_npz, verbose=verbose)
    theta_all = np.asarray(aligned["theta"], dtype=np.float64)   # (T, N, P)
    qoi_all = np.asarray(aligned["qoi"], dtype=np.float64)
    dates = aligned["dates"]
    names = aligned["param_names"]
    pairing = aligned["pairing"]

    idx = list(range(0, theta_all.shape[0], stride))
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(idx)))

    if verbose:
        what = ("anamorphoses" if method == "anamorphosis"
                else f"transport maps (order {max_order})")
        print(f"Fitting {len(idx)} {what} "
              f"({theta_all.shape[1]} particles, {theta_all.shape[2]}-D) "
              f"on {n_workers} workers")

    tasks = [(k, theta_all[k], max_order, gtol, maxiter, method) for k in idx]
    z_all = np.full((len(idx), theta_all.shape[1], theta_all.shape[2]), np.nan)
    ok = np.zeros(len(idx), dtype=bool)
    m_mean = np.full(len(idx), np.nan)
    m_cov = np.full(len(idx), np.nan)
    pos = {k: i for i, k in enumerate(idx)}

    def _collect(results):
        done = 0
        for k, z, good, mm, mc in results:
            i = pos[k]
            z_all[i], ok[i], m_mean[i], m_cov[i] = z, good, mm, mc
            done += 1
            if verbose and done % max(1, len(idx) // 10) == 0:
                print(f"  {done}/{len(idx)} dates")

    if n_workers == 1:
        _collect(map(_fit_one, tasks))
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                _collect(ex.map(_fit_one, tasks, chunksize=1))
        except BrokenProcessPool:
            # 'spawn' (the macOS default) re-imports __main__ in each worker, which
            # fails from a notebook or a piped stdin script. Fall back rather than
            # lose the run; call from a file under `if __name__ == "__main__":`
            # to get the parallel path.
            print("WARNING: process pool unavailable (are you in a notebook or "
                  "piped script?) — falling back to serial.")
            _collect(map(_fit_one, tasks))

    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(samples_npz)),
                                "standard_parameter_samples.npz")
    np.savez_compressed(
        out_path, z=z_all, theta=theta_all[idx], qoi=qoi_all[idx],
        dates=np.array([dates[k] for k in idx], dtype=object),
        param_names=np.array(names, dtype=object),
        ok=ok, max_abs_mean=m_mean, max_abs_cov=m_cov,
        max_order=max_order, stride=stride, method=method, pairing=pairing)

    if verbose:
        good = ok.sum()
        print(f"  converged on {good}/{len(idx)} dates")
        if good:
            print(f"  |mean(z)|max   median {np.nanmedian(m_mean[ok]):.4f}  "
                  f"(0 = perfectly centred)")
            print(f"  |cov(z)-I|max  median {np.nanmedian(m_cov[ok]):.4f}  "
                  f"(0 = perfectly whitened)")
        print(f"  -> {out_path}")
    return {"z": z_all, "dates": [dates[k] for k in idx], "param_names": names,
            "ok": ok, "max_abs_mean": m_mean, "max_abs_cov": m_cov,
            "pairing": pairing, "out_path": out_path}


def load_standard_samples(path):
    """Load standard_parameter_samples.npz into a dict of arrays."""
    if os.path.isdir(str(path)):
        path = os.path.join(str(path), "standard_parameter_samples.npz")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def gaussianize_parameter_samples(parameter_samples_matrix, method="legacy",
                                  max_order=2, param_names=None, verbose=True,
                                  backend=None):
    """Map a posterior parameter ensemble to a standard-Gaussian reference space.

    This is the step that makes a polynomial chaos expansion possible: PCE needs
    inputs in a standard space (independent standard normals for a Hermite basis),
    but the particle-filter posterior is correlated, bounded and non-Gaussian.

    Args:
        parameter_samples_matrix: (n_particles, n_params) posterior ensemble.
        backend: "mpart"        → joint MParT triangular map. Whitens the
                                  cross-correlations between parameters, which a
                                  marginal transform cannot; needs mpart.
                 "anamorphosis" → rank-based marginal transform (Fan et al. 2016).
                                  Far faster, indifferent to marginal shape, cannot
                                  fail to converge, but leaves cross-correlations.
                 "legacy"       → the bundled uqef_dynamic/utils/transport_map.py.
                 None           → skip, return None.
        max_order:   polynomial order of the map (mpart backend only).
        param_names: optional names, used in diagnostics.
        verbose:     print fit diagnostics.

    Returns:
        (n_particles, n_params) array in the reference space, or None when the
        transform is skipped or unavailable. Callers should handle None.
    """
    # `backend` is the old name for `method`; accepted so existing calls keep working.
    if backend is not None:
        method = backend
    method = normalize_method(method, METHODS, "gaussianize_parameter_samples")
    if method is None:
        return None

    if method == "mpart_joint":
        if not mpart_transport.is_available():
            print("WARNING: method='mpart_joint' but MParT is not installed "
                  "(pip install mpart). Skipping the Gaussianization.")
            return None
        fitted = mpart_transport.fit_transport_map(
            parameter_samples_matrix, max_order=max_order,
            param_names=param_names, verbose=verbose)
        return fitted.forward(parameter_samples_matrix)

    if method == "anamorphosis":
        fitted = gaussian_anamorphosis.fit_anamorphosis(
            parameter_samples_matrix, param_names=param_names)
        z = fitted.forward(parameter_samples_matrix)
        if verbose:
            d = fitted.diagnostics(parameter_samples_matrix)
            # marginal normality is automatic here, so report the joint number
            print(f"[anamorphosis] max|offdiag corr| after transform = "
                  f"{d['max_abs_offdiag_corr']:.4f} (a marginal transform does "
                  f"not whiten; use backend='mpart' if that matters)")
        return z

    if method == "mpart_1d":
        z = np.empty_like(parameter_samples_matrix, dtype=np.float64)
        for j in range(z.shape[1]):
            f1 = mpart_transport.fit_transport_map(
                parameter_samples_matrix[:, [j]], max_order=max_order, verbose=False)
            z[:, [j]] = f1.forward(parameter_samples_matrix[:, [j]])
        return z

    if method == "legacy":
        return legacy_transport.transform_samples_with_transport_map(
            parameter_samples_matrix)

    raise AssertionError(f"unreachable: {method!r} passed validation")  # pragma: no cover
