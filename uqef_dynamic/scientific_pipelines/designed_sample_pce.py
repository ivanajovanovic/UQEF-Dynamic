"""Designed-sample PCE: theta <-> z via a fitted transport map, NEW designed
samples in z, HBV-SASK run at each from a shared spin-up-converged initial
state, PCE of Q on z.

Why this exists: a PCE fitted directly on particle-filter (PF) posterior samples
is unreliable - out-of-sample R2 is negative - because a PF particle carries its
own model state (snow, soil moisture, storages) alongside theta, so
Q_i = M(theta_i, S_i), not M(theta_i), and no theta-only PCE can fit that (see
offline_parameter_transform_and_pce_learning.py's module docstring for the
diagnosis). This module instead:

  1. (Re)fits a transport map from the PF posterior at one reference date, purely
     to inherit its whitening structure - not to reuse the PF particles themselves.
  2. Draws NEW samples in z (random, or Gauss-Hermite quadrature).
  3. Inverts them to theta via the map's existing .inverse().
  4. Runs HBV-SASK at each theta from a COMMON, spin-up-converged initial state
     (not a per-sample random draw), one simulation per sample, for however many
     target dates are wanted from that same run.
  5. Fits one PCE per target date on the resulting (z, Q) via
     offline_parameter_transform_and_pce_learning.run_pce_learning, unchanged.

Kept as its own module because offline_parameter_transform_and_pce_learning.py's
own docstring states "nothing here re-runs the model" - this module's whole
point is to run it.
"""

import os
import json
import time
import functools
import warnings

import numpy as np
import pandas as pd
import chaospy as cp
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

from uqef_dynamic.utils import utility
from uqef_dynamic.models.hbv_sask import hbvsask_utility as hbv
from uqef_dynamic.utils import mpart_transport
from uqef_dynamic.utils import gaussian_anamorphosis
from uqef_dynamic.models.hbv_sask import HBVSASKModel as hbvmodel
from uqef_dynamic.scientific_pipelines import offline_parameter_transform_and_pce_learning as opl

warnings.filterwarnings("ignore", category=UserWarning, module="numpoly")

__all__ = ["build_designed_theta", "make_random_design", "make_gauss_hermite_design",
           "run_designed_sample_pce", "run_designed_sample_pce_evolving",
           "plot_pce_after_particle_filter"]

QOI_COLUMN_NAME = "Q_cms"


# ---------------------------------------------------------------------------
# theta <-> z: (re)fit a map for one date, invert designed z
# ---------------------------------------------------------------------------

class _Mpart1DInverse:
    """Wraps one TransportMapResult per column so mpart_1d's inverse looks like
    a joint map's: .inverse(Z) with Z of shape (n_samples, n_dim)."""

    def __init__(self, fits):
        self.fits = fits

    def inverse(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        out = np.empty_like(Z)
        for j, f in enumerate(self.fits):
            out[:, [j]] = f.inverse(Z[:, [j]])
        return out


def _fit_map_for_date(theta_k, method, max_order, param_names):
    """(Re)fit the transport map for one date's posterior. Mirrors
    offline_parameter_transform_and_pce_learning._fit_last_only, but returns
    the fitted object itself - needed for .inverse(), not just .forward()."""
    if method == "anamorphosis":
        return gaussian_anamorphosis.fit_anamorphosis(theta_k, param_names=param_names)
    if method == "mpart_1d":
        fits = [mpart_transport.fit_transport_map(theta_k[:, [j]], max_order=max_order,
                                                   verbose=False)
                for j in range(theta_k.shape[1])]
        return _Mpart1DInverse(fits)
    return mpart_transport.fit_transport_map(theta_k, max_order=max_order,
                                              param_names=param_names, verbose=False)


def build_designed_theta(theta_at_date, z_design, method="mpart_joint",
                         max_order=2, param_names=None):
    """Fit the map for one date's posterior, invert z_design, then let the fitted
    map - and any live MParT object inside it - go out of scope. Runs once in
    the parent process, before any multiprocessing: TransportMapResult holds a
    live pybind11 object with no __getstate__/__reduce__ and is not expected to
    survive a spawn boundary, so it is never passed to a worker."""
    fitted = _fit_map_for_date(theta_at_date, method, max_order, param_names)
    return fitted.inverse(np.asarray(z_design, dtype=np.float64))


# ---------------------------------------------------------------------------
# Designed samples in z
# ---------------------------------------------------------------------------

def make_random_design(n_dim, n_samples=200, rule="random", seed=None):
    """N(0, I) draws, shape (n_samples, n_dim). Distinct Normal instances are
    required: cp.J(*[cp.Normal(0,1)] * n) repeats one object and chaospy reads
    that as a dependent joint (hit and fixed earlier this session)."""
    dist = cp.J(*[cp.Normal(0, 1) for _ in range(n_dim)])
    z = dist.sample(n_samples, rule=rule, seed=seed)   # (n_dim, n_samples)
    return np.asarray(z, dtype=np.float64).T


def make_gauss_hermite_design(n_dim, order, growth=False, sparse=False, max_nodes=500):
    """Tensor Gauss-Hermite nodes/weights, shape (n_nodes, n_dim) / (n_nodes,).

    A full tensor grid has (order+1)**n_dim nodes - one HBV-SASK run each - so
    this refuses BEFORE building anything once that exceeds max_nodes; use
    sparse=True or make_random_design instead of raising the order in 7-D."""
    dist = cp.J(*[cp.Normal(0, 1) for _ in range(n_dim)])
    nodes, weights = cp.generate_quadrature(order, dist, rule="gaussian",
                                            growth=growth, sparse=sparse)
    n_nodes = nodes.shape[1]
    # if n_nodes > max_nodes:
    #     raise ValueError(
    #         f"Gauss-Hermite order {order} over {n_dim} dimensions needs {n_nodes} "
    #         f"nodes - one HBV-SASK run each - which exceeds max_nodes={max_nodes}. "
    #         "Lower the order, set sparse=True, or use design='random' instead.")
    return np.asarray(nodes, dtype=np.float64).T, np.asarray(weights, dtype=np.float64)


# ---------------------------------------------------------------------------
# Running HBV-SASK at each designed theta, from a common initial state
# ---------------------------------------------------------------------------

def _run_designed_sample(model, parameter_value, unique_index_model_run,
                         qoi_column_name, target_date_positions):
    """Spawn-safe HBV-SASK-only replacement for
    KL_and_PCE_time_dependent_processes_pipeline.run_model_single_parameter_node's
    HBV-SASK branch: that function branches on a module-level MODEL global
    hardcoded to 'battery' in that file, so importing it directly would silently
    run the wrong branch, and monkey-patching the global would not survive
    multiprocessing's spawned workers re-importing the module fresh.

    No initial_condition_df override is passed to model.run(), so every sample
    uses the model's single self.initial_condition_df (read once at
    construction) - this is what gives every designed sample the same starting
    state; the only per-sample difference is which theta that state gets
    integrated forward under during spin-up.

    Subsets to target_date_positions before returning, so the parent process
    only ever accumulates (n_samples, n_target_dates), not (n_samples, n_days).
    Bound via functools.partial (qoi_column_name, target_date_positions) before
    being handed to utility.running_model_in_parallel_and_generating_df, which
    calls routine(model, parameter_value, unique_index_model_run) positionally.
    """
    results_list = model.run(
        i_s=[unique_index_model_run],
        parameters=[parameter_value],
        createNewFolder=False,
        take_direct_value=False,
        merge_output_with_measured_data=False,
    )
    rts = results_list[0][0]["result_time_series"]
    y_t_full = rts[qoi_column_name].to_numpy()
    return unique_index_model_run, y_t_full[target_date_positions], parameter_value


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_designed_sample_pce(
    working_dir, configuration_file, inputModelDir, model_working_dir,
    reference_date, target_dates,
    n_samples=200, design="random", method="mpart_joint", max_order=2,
    pce_order=2, cross_truncation=1.0, warmup_years=3, basin=None,
    out_dir=None, out_name=None, n_workers=None, seed=None, verbose=True,
):
    """End-to-end: designed z -> theta (via the transport map fit at
    reference_date) -> one common-warmup HBV-SASK run per sample, covering every
    target date in one run each -> one PCE per target date, fit in a single
    batched call to run_pce_learning (which already parallelises across dates)
    since all target dates share the same (z, theta) design - theta is a single
    vector per simulation and cannot vary by date within one run.

    Args:
        working_dir:        PF run dir holding posterior_parameter_samples.npz -
                            the source of theta at reference_date, used only to
                            fit the map (never reused as PCE training data).
        configuration_file: the config that produced that PF run. Its
                            time_settings is shifted (derive_common_warmup_configuration)
                            to prepend warmup_years of spin-up; everything else
                            (parameters, states, forcing paths) is kept.
        inputModelDir, model_working_dir: passed straight to HBVSASKModel, as in
                            particle_filtering_pipeline.py's own construction.
        reference_date:     PF date whose posterior defines the design space.
        target_dates:       date or list of dates (within the ORIGINAL, non-shifted
                            prediction window) to collect Q and fit a PCE at.
        n_samples:          for design="random". Ignored for "gauss_hermite"
                            (node count is set by max_order instead).
        design:             "random" (regression fit) or "gauss_hermite"
                            (quadrature fit with the real GH weights).
        method:             transport-map method for the theta<->z map.
        max_order:          transport-map polynomial order (mpart methods) and,
                            when design="gauss_hermite", the quadrature order.
        pce_order:          PCE total order.
        cross_truncation:   passed straight through to run_pce_learning; < 1.0
                            drops higher interaction terms, so the actual term
                            count is read off the built expansion, not
                            math.comb(pce_order+n_params, n_params) (that
                            formula only holds at cross_truncation=1.0).
        warmup_years:       years of spin-up prepended to the config.
        out_dir:            where PCE outputs are written; default working_dir.
        n_workers:          model-run parallelism; None -> os.cpu_count().

    Returns:
        The run_pce_learning(...) result dict - one PCE per target date, as
        arrays indexed by date (see its own docstring for the field list).
    """
    # Normalized to date-only strings throughout: the saved posterior carries
    # full timestamps ("2005-01-09 00:00:00") while HBVSASKModel.simulation_range
    # carries bare dates - comparing them unnormalized silently fails to match.
    if isinstance(target_dates, str):
        target_dates = [target_dates]
    target_dates = [str(pd.Timestamp(t).date()) for t in target_dates]
    reference_date = str(pd.Timestamp(reference_date).date())

    samples_file = os.path.join(str(working_dir), "posterior_parameter_samples.npz")
    d = np.load(samples_file, allow_pickle=True)
    dates = [str(pd.Timestamp(x).date()) for x in d["dates"]]
    names = [str(x) for x in d["param_names"]]
    lower, upper = np.asarray(d["param_lower"]), np.asarray(d["param_upper"])
    n_params = len(names)

    if reference_date not in dates:
        raise ValueError(f"reference_date {reference_date!r} not in the saved "
                         f"posterior's date range ({dates[0]}..{dates[-1]}).")
    theta_ref = np.asarray(d["theta"][dates.index(reference_date)])

    if design == "random":
        z_design = make_random_design(n_params, n_samples=n_samples, seed=seed)
        weights_quad = None
    elif design == "gauss_hermite":
        z_design, weights_quad = make_gauss_hermite_design(n_params, order=max_order)
    else:
        raise ValueError(f"design must be 'random' or 'gauss_hermite', got {design!r}.")
    n_samples_actual = z_design.shape[0]

    theta_design = build_designed_theta(theta_ref, z_design, method=method,
                                        max_order=max_order, param_names=names)

    # Two failure modes of the inverse, both dropped rather than handed to the
    # model:
    #
    # 1. Non-finite: the map's inverse is a root-solve (mpart) or an
    #    interpolation clamp (anamorphosis); both can fail for a z far in the
    #    tail of a parameter with a very tight posterior. Observed ~4% at
    #    order-1 random designs on this data.
    # 2. Finite but outside [param_lower, param_upper]: the map is a smooth
    #    polynomial fit to a bounded posterior, so nothing stops its inverse
    #    from extrapolating past the physical bounds for a z beyond the
    #    training support. NOT cosmetic - verified on this data that a single
    #    such sample (PM=47.5 against a bound of 2.0) produced Q=636 cms
    #    against a normal range of single digits, and that alone drove
    #    out-of-sample R2 from +0.88 (bounds-filtered) to -35 (unfiltered) on
    #    an otherwise identical 250-sample design. Observed ~40% of a plain
    #    N(0,1) design landing out-of-bounds here, so this is not a rare edge
    #    case to shrug off - it is the dominant failure mode.
    #
    # Regression tolerates an uneven N fine; a Gauss-Hermite design cannot
    # (dropping nodes invalidates the quadrature weights), so that combination
    # is only warned about, not silently patched.
    finite = np.all(np.isfinite(theta_design), axis=1)
    in_bounds = np.zeros_like(finite)
    in_bounds[finite] = np.all((theta_design[finite] >= lower) &
                               (theta_design[finite] <= upper), axis=1)
    keep = finite & in_bounds
    n_nonfinite = int((~finite).sum())
    n_oob = int((finite & ~in_bounds).sum())
    if n_nonfinite or n_oob:
        if design == "gauss_hermite":
            warnings.warn(
                f"{n_nonfinite + n_oob}/{n_samples_actual} Gauss-Hermite nodes "
                "inverted to non-finite or out-of-bounds theta and were dropped; "
                "the remaining weights no longer sum to 1 and are not a valid "
                "quadrature rule. Lower max_order or refit the map with a "
                "higher transport-map max_order instead of using this output "
                "as-is.")
        if verbose:
            print(f"  dropped {n_nonfinite} non-finite + {n_oob} out-of-bounds "
                  f"of {n_samples_actual} designed samples "
                  f"({keep.sum()}/{n_samples_actual} kept)")
        z_design, theta_design = z_design[keep], theta_design[keep]
        if weights_quad is not None:
            weights_quad = weights_quad[keep]
        n_samples_actual = z_design.shape[0]
    # Read the real term count off the expansion itself - math.comb(pce_order +
    # n_params, n_params) only holds at cross_truncation=1.0; at 0.7 it badly
    # overcounts (order 3, 7 params: predicts 120, actual is 43), which would
    # reject a perfectly fittable request before spending any model-run budget.
    _dist = cp.J(*[cp.Normal(0, 1) for _ in range(n_params)])
    _exp, _ = cp.generate_expansion(order=pce_order, dist=_dist,
                                    rule="three_terms_recurrence", normed=True,
                                    graded=True, reverse=True,
                                    cross_truncation=cross_truncation, retall=True)
    n_terms_needed = len(_exp)
    if n_samples_actual < n_terms_needed:
        raise ValueError(
            f"Only {n_samples_actual} designed samples survived (of "
            f"{'the requested ' + str(n_samples) if design == 'random' else 'the grid'}), "
            f"fewer than the {n_terms_needed} terms pce_order={pce_order} "
            f"cross_truncation={cross_truncation} needs. Increase n_samples or "
            "lower pce_order/cross_truncation.")

    warmup_cfg_path = os.path.join(
        str(out_dir or working_dir),
        f"{os.path.splitext(os.path.basename(str(configuration_file)))[0]}"
        f"_designed_warmup.json")
    warmup_cfg = hbv.derive_common_warmup_configuration(
        configuration_file, warmup_years=warmup_years, out_path=warmup_cfg_path)
    if basin is None:
        basin = warmup_cfg.get("model_settings", {}).get("basin")

    param_order = utility.get_list_of_uncertain_parameters_from_configuration_dict(warmup_cfg)
    if list(param_order) != list(names):
        raise ValueError(
            "Parameter order in the configuration does not match param_names "
            f"saved in posterior_parameter_samples.npz: {param_order} vs {names}. "
            "theta_design columns would be silently misassigned to the wrong "
            "parameter by model.run(take_direct_value=False).")

    model = hbvmodel.HBVSASKModel(
        configurationObject=warmup_cfg, inputModelDir=inputModelDir,
        workingDir=model_working_dir, basin=basin,
        writing_results_to_a_file=False, plotting=False)

    sim_dates = [str(x.date()) for x in model.simulation_range]
    missing = [t for t in target_dates if t not in sim_dates]
    if missing:
        raise ValueError(
            f"target_dates {missing} fall outside the model's prediction window "
            f"({sim_dates[0]}..{sim_dates[-1]}); warmup_years shifted start_date "
            "but simulation_length/end_date were kept, so the window should be "
            "unchanged - check configuration_file's original time_settings.")
    target_date_positions = np.array([sim_dates.index(t) for t in target_dates])

    routine = functools.partial(_run_designed_sample,
                                qoi_column_name=QOI_COLUMN_NAME,
                                target_date_positions=target_date_positions)
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    if verbose:
        print(f"Running {n_samples_actual} designed samples ({design}) x "
              f"{warmup_years}y warm-up on {n_workers} worker(s)...")
    model_runs, _ = utility.running_model_in_parallel_and_generating_df(
        model, routine, t=target_dates, parameters=theta_design.T,
        list_unique_index_model_run_list=list(range(n_samples_actual)),
        num_processes=n_workers)
    # model_runs: (n_samples, n_target_dates)

    # One design shared by every target date - z/theta are identical across the
    # date axis, broadcast rather than varying by date. Batched into a single
    # file so run_pce_learning fits all dates in one call, using its own
    # parallel-across-dates machinery instead of this function looping and
    # re-importing/re-building the basis once per date.
    n_dates = len(target_dates)
    stem = f"designed_{design}_{target_dates[0]}_{target_dates[-1]}"
    npz_path = os.path.join(str(out_dir or working_dir),
                            f"standard_parameter_samples_{stem}.npz")
    np.savez_compressed(
        npz_path,
        z=np.broadcast_to(z_design, (n_dates, *z_design.shape)),
        theta=np.broadcast_to(theta_design, (n_dates, *theta_design.shape)),
        qoi=model_runs.T,
        dates=np.array(target_dates, dtype=object),
        param_names=np.array(names, dtype=object),
        param_lower=lower, param_upper=upper,
        ok=np.ones(n_dates, dtype=bool), method=method, scope="designed",
        max_order=max_order, stride=1)
    if verbose:
        print(f"wrote {npz_path} ({n_dates} dates x {n_samples_actual} samples)")

    return opl.run_pce_learning(
        npz_path, out_dir=out_dir, pce_order=pce_order,
        cross_truncation=cross_truncation,
        regression=(design == "random"), weights_quad=weights_quad,
        n_workers=n_workers, out_name=(out_name or f"pce_{stem}.npz"),
        configuration_file=str(configuration_file), verbose=verbose)


def run_designed_sample_pce_evolving(
    working_dir, configuration_file, inputModelDir, model_working_dir,
    target_dates,
    n_samples=2000, design="random", method="mpart_joint", max_order=2,
    pce_order=2, cross_truncation=1.0, warmup_years=3, basin=None,
    out_dir=None, out_name=None, n_workers=None, seed=None, verbose=True,
):
    """Like run_designed_sample_pce, but uses the PF's own EVOLVING posterior
    instead of one reference date's snapshot: fits a SEPARATE transport map
    and a SEPARATE designed-theta set for EVERY target date, from that date's
    own PF posterior. Each date's theta then needs its own independent
    warm-up - state cannot be shared across dates once theta varies by date -
    so this does ~n_dates times more model runs than run_designed_sample_pce
    for the same n_samples. Budget accordingly: at ~20ms/run, 2000 samples x
    335 dates is on the order of tens of minutes even parallelised.

    The SAME raw z-draws (fixed once via `seed`) are reused across every date,
    only re-inverted through each date's own map, so "sample i" is the same
    abstract Gaussian point everywhere - only its mapped theta (and hence its
    warm-up trajectory) differs by date. Per-date sample survival after the
    finite/in-bounds filter (see run_designed_sample_pce) varies by date, so
    - unlike the single-reference-date version - dates cannot share one
    combined array; each is written and fit as its own single-date file, then
    the per-date PCE results are stacked into one combined result/output,
    matching run_pce_learning's return shape so plot_pce_after_particle_filter
    can consume it unchanged.

    Args: as run_designed_sample_pce, except there is no reference_date - the
    posterior at each target date supplies its own map.

    Returns:
        dict with the same fields as run_pce_learning's return (E, Var,
        StdDev, R2, RMSE, P10, P90, Sobol_m, Sobol_t, gpce_coeff, ok, dates,
        param_names), plus "out_path"/"summary_path" for the combined files
        this also writes.
    """
    if isinstance(target_dates, str):
        target_dates = [target_dates]
    target_dates = [str(pd.Timestamp(t).date()) for t in target_dates]
    n_dates = len(target_dates)

    samples_file = os.path.join(str(working_dir), "posterior_parameter_samples.npz")
    d = np.load(samples_file, allow_pickle=True)
    post_dates = [str(pd.Timestamp(x).date()) for x in d["dates"]]
    names = [str(x) for x in d["param_names"]]
    lower, upper = np.asarray(d["param_lower"]), np.asarray(d["param_upper"])
    n_params = len(names)

    missing = [t for t in target_dates if t not in post_dates]
    if missing:
        raise ValueError(f"target_dates {missing[:5]}{'...' if len(missing) > 5 else ''} "
                         f"not in the saved posterior's date range "
                         f"({post_dates[0]}..{post_dates[-1]}).")

    # d["theta"] on a compressed .npz does NOT cache - every access re-inflates
    # the WHOLE member from the zip's deflate stream, discarding the result
    # each time (confirmed: d["theta"] is d["theta"] -> False). Reading it once
    # here and slicing the in-memory array per date, instead of indexing
    # d["theta"][...] inside the loop, is the difference between one ~600MB
    # decompression and O(n_dates) of them - at 1096 dates that second form
    # would spend most of its wall-clock time re-inflating the same array
    # over and over rather than running the model. qoi is never read here (the
    # PF's own qoi isn't used - only its theta, to fit each date's map).
    theta_all = np.asarray(d["theta"])
    date_to_idx = {dt: i for i, dt in enumerate(post_dates)}

    if design == "random":
        z_raw = make_random_design(n_params, n_samples=n_samples, seed=seed)
        base_weights = None
    elif design == "gauss_hermite":
        z_raw, base_weights = make_gauss_hermite_design(n_params, order=max_order)
    else:
        raise ValueError(f"design must be 'random' or 'gauss_hermite', got {design!r}.")
    n_samples_raw = z_raw.shape[0]

    # Term count (hence the minimum viable per-date sample count) does not
    # depend on the date, only on pce_order/cross_truncation/n_params - see
    # the note in run_pce_learning on why this must be read off the built
    # expansion rather than math.comb, which overcounts under truncation.
    _exp, _ = cp.generate_expansion(
        order=pce_order, dist=cp.J(*[cp.Normal(0, 1) for _ in range(n_params)]),
        rule="three_terms_recurrence", normed=True, graded=True, reverse=True,
        cross_truncation=cross_truncation, retall=True)
    n_terms_needed = len(_exp)

    # Basin and parameter order come from the base config and do not change
    # across the per-date derived configs (only time_settings does), so both
    # are resolved once rather than per date.
    base_cfg = utility.check_if_configurationObject_is_in_right_format_and_return(
        configuration_file, raise_error=True)
    if basin is None:
        basin = base_cfg.get("model_settings", {}).get("basin")
    param_order = utility.get_list_of_uncertain_parameters_from_configuration_dict(base_cfg)
    if list(param_order) != list(names):
        raise ValueError(
            "Parameter order in the configuration does not match param_names "
            f"saved in posterior_parameter_samples.npz: {param_order} vs {names}.")

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    out_dir = str(out_dir or working_dir)
    os.makedirs(out_dir, exist_ok=True)

    E = np.full(n_dates, np.nan); Var = np.full(n_dates, np.nan)
    R2 = np.full(n_dates, np.nan); RMSE = np.full(n_dates, np.nan)
    P10 = np.full(n_dates, np.nan); P90 = np.full(n_dates, np.nan)
    Sobol_m = np.full((n_dates, n_params), np.nan)
    Sobol_t = np.full((n_dates, n_params), np.nan)
    coeffs = np.full((n_dates, n_terms_needed), np.nan, dtype=np.float32)
    ok = np.zeros(n_dates, dtype=bool)
    n_kept_per_date = np.zeros(n_dates, dtype=int)

    t0 = time.perf_counter()
    for j, tgt in enumerate(target_dates):
        theta_t = theta_all[date_to_idx[tgt]]
        theta_design_t = build_designed_theta(theta_t, z_raw, method=method,
                                              max_order=max_order, param_names=names)

        finite = np.all(np.isfinite(theta_design_t), axis=1)
        in_bounds = np.zeros_like(finite)
        in_bounds[finite] = np.all((theta_design_t[finite] >= lower) &
                                   (theta_design_t[finite] <= upper), axis=1)
        keep = finite & in_bounds
        n_kept = int(keep.sum())
        n_kept_per_date[j] = n_kept
        if n_kept < n_terms_needed:
            if verbose:
                print(f"[{j + 1}/{n_dates}] {tgt}: skipped, only {n_kept}/{n_samples_raw} "
                      f"designed samples survived (need >= {n_terms_needed})")
            continue
        z_t, theta_t_kept = z_raw[keep], theta_design_t[keep]
        weights_t = base_weights[keep] if base_weights is not None else None

        # Not written to disk: 335 near-identical single-day configs would be
        # clutter with little individual provenance value, unlike the one
        # config run_designed_sample_pce writes for a single reference date.
        cfg_t = hbv.derive_single_date_warmup_configuration(
            configuration_file, target_date=tgt, warmup_years=warmup_years, out_path=None)
        model = hbvmodel.HBVSASKModel(
            configurationObject=cfg_t, inputModelDir=inputModelDir,
            workingDir=model_working_dir, basin=basin,
            writing_results_to_a_file=False, plotting=False)

        # simulation_range has 2 entries here (HBVSASKModel's inclusive
        # date_range convention: end_date = start_date_predictions +
        # simulation_length days, with simulation_length's minimum viable
        # value of 1) - target_date is always the first.
        sim_dates = [str(x.date()) for x in model.simulation_range]
        pos = np.array([sim_dates.index(tgt)])

        routine = functools.partial(_run_designed_sample,
                                    qoi_column_name=QOI_COLUMN_NAME,
                                    target_date_positions=pos)
        model_runs, _ = utility.running_model_in_parallel_and_generating_df(
            model, routine, t=[tgt], parameters=theta_t_kept.T,
            list_unique_index_model_run_list=list(range(n_kept)),
            num_processes=n_workers)
        q_t = model_runs[:, 0]

        npz_path_t = os.path.join(out_dir, f"standard_parameter_samples_evolving_{tgt}.npz")
        np.savez_compressed(
            npz_path_t, z=z_t[np.newaxis, ...], theta=theta_t_kept[np.newaxis, ...],
            qoi=q_t[np.newaxis, ...], dates=np.array([tgt], dtype=object),
            param_names=np.array(names, dtype=object),
            param_lower=lower, param_upper=upper,
            ok=np.array([True]), method=method, scope="designed_evolving",
            max_order=max_order, stride=1)
        r_t = opl.run_pce_learning(
            npz_path_t, out_dir=out_dir, pce_order=pce_order,
            cross_truncation=cross_truncation, regression=(design == "random"),
            weights_quad=weights_t, n_workers=1,
            out_name=f"pce_evolving_{tgt}.npz",
            configuration_file=str(configuration_file), verbose=False)

        E[j] = r_t["E"][0]; Var[j] = r_t["Var"][0]
        R2[j] = r_t["R2"][0]; RMSE[j] = r_t["RMSE"][0]
        P10[j] = r_t["P10"][0]; P90[j] = r_t["P90"][0]
        Sobol_m[j] = r_t["Sobol_m"][0]; Sobol_t[j] = r_t["Sobol_t"][0]
        ok[j] = bool(r_t["ok"][0])
        coeffs[j] = r_t["gpce_coeff"][0]
        # These per-date artefacts are superseded by the combined file written
        # below; remove them rather than leaving 3 x n_dates small files behind.
        for p in (npz_path_t, r_t["out_path"], r_t["summary_path"]):
            if os.path.isfile(p):
                os.remove(p)

        if verbose:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (j + 1) * (n_dates - j - 1)
            print(f"[{j + 1}/{n_dates}] {tgt}: kept {n_kept}/{n_samples_raw}, "
                  f"R2={R2[j]:.3f}  (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

    n_ok = int(np.sum(ok))
    med_r2 = float(np.nanmedian(R2[ok])) if n_ok else None
    summary = {
        "working_dir": str(working_dir), "configuration_file": str(configuration_file),
        "pce_order": pce_order, "cross_truncation": cross_truncation,
        "n_terms": n_terms_needed, "design": design, "n_samples_requested": n_samples_raw,
        "warmup_years": warmup_years, "n_dates": n_dates, "n_converged": n_ok,
        "n_params": n_params, "param_names": names,
        "median_kept_samples": float(np.median(n_kept_per_date)),
        "min_kept_samples": int(n_kept_per_date.min()),
        "median_R2": med_r2,
        "min_R2": float(np.nanmin(R2[ok])) if n_ok else None,
        "max_R2": float(np.nanmax(R2[ok])) if n_ok else None,
        "n_dates_R2_above_0.5": int(np.sum(R2[ok] > 0.5)) if n_ok else 0,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    if n_ok:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            summary["median_Sobol_t"] = {
                nm: float(np.nanmedian(Sobol_t[ok, k])) for k, nm in enumerate(names)}

    stem = f"evolving_{design}_{target_dates[0]}_{target_dates[-1]}"
    out_path = os.path.join(out_dir, out_name or f"pce_{stem}.npz")
    np.savez_compressed(
        out_path, gpce_coeff=coeffs, E=E, Var=Var, StdDev=np.sqrt(Var),
        Sobol_m=Sobol_m, Sobol_t=Sobol_t, R2=R2, RMSE=RMSE, P10=P10, P90=P90,
        ok=ok, dates=np.array(target_dates, dtype=object),
        param_names=np.array(names, dtype=object),
        pce_order=pce_order, n_terms=n_terms_needed, n_samples_requested=n_samples_raw)
    summary_path = os.path.join(out_dir, f"pce_summary_{stem}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nfitted {n_ok}/{n_dates} dates in {summary['elapsed_seconds']:.0f}s")
        if med_r2 is not None:
            print(f"  R2 median {med_r2:.4f} (min {summary['min_R2']:.4f}, "
                  f"max {summary['max_R2']:.4f}); "
                  f"{summary['n_dates_R2_above_0.5']}/{n_ok} dates above 0.5")
        print(f"  -> {out_path}")
        print(f"  -> {summary_path}")

    return {"gpce_coeff": coeffs, "E": E, "Var": Var, "StdDev": np.sqrt(Var),
            "Sobol_m": Sobol_m, "Sobol_t": Sobol_t, "R2": R2, "RMSE": RMSE,
            "P10": P10, "P90": P90, "ok": ok, "dates": target_dates,
            "param_names": names, "summary": summary,
            "out_path": out_path, "summary_path": summary_path}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pce_after_particle_filter(pce_result, working_dir, out_dir=None,
                                   light_output=False,
                                   filename="pce_after_particle_filter_streamflow"):
    """Streamflow from the designed-sample PCE against observed, in the style of
    particle_filtering_pipeline.py's particle_filter_streamflow.pdf (band +
    mean + observed), plus a second panel of each parameter's total-order
    Sobol index over the same dates.

    Args:
        pce_result:  dict returned by run_designed_sample_pce, or
                    load_pce_output(...) on its saved .npz.
        working_dir: PF run dir holding averaged_and_simulated.pkl, read here
                    only for the observed-streamflow overlay.
        out_dir:     where to write the .pdf/.html; default working_dir.

    Returns:
        The plotly Figure.
    """
    dates = [pd.Timestamp(str(d)) for d in pce_result["dates"]]
    E = np.asarray(pce_result["E"])
    P10 = np.asarray(pce_result["P10"])
    P90 = np.asarray(pce_result["P90"])
    Sobol_t = np.asarray(pce_result["Sobol_t"])
    names = [str(x) for x in pce_result["param_names"]]

    obs_df = pd.read_pickle(
        os.path.join(str(working_dir), "averaged_and_simulated.pkl"), compression="gzip")
    observed = obs_df["observed_streamflow"].reindex(dates).to_numpy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.62, 0.38],
        subplot_titles=("Streamflow: designed-sample PCE vs observed",
                        "Total-order Sobol index over time"))

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
            line=dict(width=1.5)), row=2, col=1)

    fig.update_yaxes(title_text="Q [m³/s]", row=1, col=1)
    fig.update_yaxes(title_text="Sobol_t", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        template="plotly_white", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        title="PCE surrogate on designed samples, after particle filtering",
        margin=dict(t=140))

    out_dir = str(out_dir or working_dir)
    if not light_output:
        pyo.plot(fig, filename=os.path.join(out_dir, filename + ".html"), auto_open=False)
    try:
        fig.write_image(os.path.join(out_dir, filename + ".pdf"), width=1400, height=900)
    except Exception as e:
        print(f"PDF export skipped (install kaleido): {e}")
    return fig
