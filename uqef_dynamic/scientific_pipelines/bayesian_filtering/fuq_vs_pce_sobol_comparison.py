"""Compare first-order (main-effect) Sobol indices from two pipelines on the
same HBV-SASK model/basin:

  FUQ  - a pure forward uncertainty propagation.
  PCE  - built from a particle filter's evolving posterior: at each date,
         theta is drawn from the PF's belief at that date, and Sobol_m is
         read off the fitted PCE coefficients.

These measure sensitivity to theta under two different distributions (prior
vs. data-informed posterior), so agreement or disagreement between them is
not simply "one is right and one is wrong".
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from uqef_dynamic.utils import utility
from uqef_dynamic.scientific_pipelines.bayesian_filtering.particle_filtering_and_pce_plotting import (
    plot_sobol_overlay)

__all__ = ["load_fuq_sobol", "load_pce_sobol", "generalized_index_from_var_and_sobol",
          "compute_rank_agreement", "compute_pointwise_gof", "compute_generalized_comparison",
          "compute_generalized_sobol_s1",
          "plot_sobol_overlay", "compare_fuq_and_pce_sobol"]

TIME_COLUMN = "TimeStamp"
DEFAULT_GOF_LIST = ["RMSE", "CorrelationCoefficient", "PBIAS"]


def load_fuq_sobol(fuq_dir, qoi_column="Q_cms", restrict_dates=None):
    """Load the FUQ study's per-date Var/Sobol_m into a DataFrame.

    Args:
        fuq_dir:     directory holding statistics_dictionary_qoi_<qoi_column>.pkl
                    and configurationObject (for parameter order/names).
        qoi_column:  which saved QoI's statistics dict to read.
        restrict_dates: optional (start, end) pair (anything pd.Timestamp
                    accepts); dates outside [start, end] are dropped
                    immediately, before anything else touches this data -
                    e.g., use this to cut the FUQ run's 3-year record down to
                    whatever window the PCE run actually covers.

    Returns:
        (df, param_names): df has columns [TIME_COLUMN, "Var"] plus one
        "Sobol_m_<param>" column per parameter, in the order recorded in
        configurationObject; param_names is that same list.
    """
    with open(os.path.join(str(fuq_dir), "configurationObject"), "rb") as f:
        cfg = pickle.load(f)
    param_names = utility.get_list_of_uncertain_parameters_from_configuration_dict(cfg)

    with open(os.path.join(str(fuq_dir), f"statistics_dictionary_qoi_{qoi_column}.pkl"), "rb") as f:
        stat = pickle.load(f)

    rows = []
    for date, entry in stat.items():
        row = {TIME_COLUMN: pd.Timestamp(date), "Var": float(entry["Var"])}
        sobol_m = np.asarray(entry["Sobol_m"], dtype=np.float64)
        for j, name in enumerate(param_names):
            row[f"Sobol_m_{name}"] = float(sobol_m[j])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(TIME_COLUMN).reset_index(drop=True)

    if restrict_dates is not None:
        start, end = pd.Timestamp(restrict_dates[0]), pd.Timestamp(restrict_dates[1])
        n_before = len(df)
        df = df[(df[TIME_COLUMN] >= start) & (df[TIME_COLUMN] <= end)].reset_index(drop=True)
        print(f"load_fuq_sobol: restricted {n_before} -> {len(df)} dates "
              f"({start.date()} .. {end.date()}).")
    return df, param_names


def load_pce_sobol(pce_dir, pce_stem):
    """Load the PCE run's per-date Var/Sobol_m into the same DataFrame shape.

    Returns:
        (df, param_names) - same shape as load_fuq_sobol's return.
    """
    d = np.load(os.path.join(str(pce_dir), f"pce_{pce_stem}.npz"), allow_pickle=True)
    param_names = [str(x) for x in d["param_names"]]
    dates = pd.to_datetime([str(x) for x in d["dates"]])
    var = np.asarray(d["Var"], dtype=np.float64)
    sobol_m = np.asarray(d["Sobol_m"], dtype=np.float64)
    ok = np.asarray(d["ok"], dtype=bool)

    row = {TIME_COLUMN: dates, "Var": var}
    for j, name in enumerate(param_names):
        row[f"Sobol_m_{name}"] = sobol_m[:, j]
    df = pd.DataFrame(row)
    df = df[ok].sort_values(TIME_COLUMN).reset_index(drop=True)
    return df, param_names


def _resolution_days(ts_a, ts_b):
    return (pd.Timestamp(ts_a) - pd.Timestamp(ts_b)).days


def generalized_index_from_var_and_sobol(dates, var_t, sobol_t, look_back_window_size="whole"):
    """Time-averaged, variance-weighted running index computed directly from a
    (Var(t), Sobol(t)) time series - no PCE coefficients needed.

    This reproduces the exact trapezoidal weighting
    utility.computing_generalized_sobol_indices_from_poly_expan_single_timesample
    uses (uniform count-based h, endpoints halved, running cumulative window),
    generalized to skip the coefficient step: for any orthonormal PCE,
    Var_total(t) * Sobol_j(t) IS the partial variance attributable to
    parameter j at date t by definition of what a Sobol index means - so this
    is the SAME formula, just fed a precomputed numerator instead of deriving
    it from coefficients. That makes it usable on a pure Monte-Carlo Sobol
    estimate that never fit a PCE at all (this module's actual use case), and
    it can also serve as a coefficient-free cross-check against
    offline_parameter_transform_and_pce_learning.compute_generalized_sobol_indices
    when both are available - the two should agree closely wherever they do.

    Args:
        dates:   length n_dates, chronologically ordered.
        var_t:   (n_dates,) total variance of the QoI at each date.
        sobol_t: (n_dates, n_params) Sobol (main-effect, typically) index at
                each date.
        look_back_window_size: 'whole' (default) or an int number of days.

    Returns:
        (n_dates, n_params) array; out[k] is the running generalized index
        using every date up to and including date k (or just the last
        look_back_window_size days of them).
    """
    var_t = np.asarray(var_t, dtype=np.float64)
    sobol_t = np.asarray(sobol_t, dtype=np.float64)
    n_dates, n_params = sobol_t.shape
    ts = pd.to_datetime([str(d) for d in dates])
    out = np.full((n_dates, n_params), np.nan)

    for k in range(n_dates):
        if look_back_window_size == "whole":
            idx = np.arange(k + 1)
        else:
            days_back = np.array([_resolution_days(ts[k], ts[j]) for j in range(k + 1)])
            idx = np.where((days_back >= 0) & (days_back <= look_back_window_size))[0]
        n = len(idx)
        if n > 1:
            w = np.full(n, 1.0 / (n - 1))
            w[0] /= 2.0
            w[-1] /= 2.0
        else:
            w = np.array([1.0])
        var_window = var_t[idx]
        denom = float(np.dot(var_window, w))
        if denom <= 0:
            continue
        for j in range(n_params):
            numer_window = var_window * sobol_t[idx, j]
            out[k, j] = float(np.dot(numer_window, w) / denom)
    return out


def compute_rank_agreement(df, param_names):
    """Spearman rank correlation between the two methods' median-over-time
    parameter importance ranking - one number for "do the two methods agree
    on which parameters matter," robust to the estimator/distribution
    differences between them.

    Returns:
        dict: {"spearman_r": float, "p_value": float,
              "median_FUQ": {param: value}, "median_PCE": {param: value}}
    """
    med_fuq = {p: float(df[f"Sobol_m_{p}_FUQ"].median()) for p in param_names}
    med_pce = {p: float(df[f"Sobol_m_{p}_PCE"].median()) for p in param_names}
    r, p = spearmanr([med_fuq[p] for p in param_names], [med_pce[p] for p in param_names])
    return {"spearman_r": float(r), "p_value": float(p),
            "median_FUQ": med_fuq, "median_PCE": med_pce}


def compute_pointwise_gof(df, param_names, gof_list=DEFAULT_GOF_LIST):
    """Per-parameter GoF between the two Sobol_m(t) series, reusing the same
    utility.calculateGoodnessofFit_simple machinery as pf_vs_pce_comparison.py.

    Returns:
        dict: {param: {metric: value}}
    """
    out = {}
    for p in param_names:
        out[p] = utility.calculateGoodnessofFit_simple(
            measuredDF=df[[TIME_COLUMN, f"Sobol_m_{p}_FUQ"]],
            simulatedDF=df[[TIME_COLUMN, f"Sobol_m_{p}_PCE"]],
            gof_list=list(gof_list),
            measuredDF_time_column_name=TIME_COLUMN, simulatedDF_time_column_name=TIME_COLUMN,
            measuredDF_column_name=f"Sobol_m_{p}_FUQ", simulatedDF_column_name=f"Sobol_m_{p}_PCE")
    return out


def compute_generalized_comparison(df, param_names, look_back_window_size="whole"):
    """The final (whole-window) generalized Sobol_m for FUQ and PCE, per
    parameter, via generalized_index_from_var_and_sobol on each source's own
    (Var, Sobol_m) columns.

    Returns:
        dict: {param: {"FUQ": float, "PCE": float}}
    """
    fuq_sobol = df[[f"Sobol_m_{p}_FUQ" for p in param_names]].to_numpy()
    pce_sobol = df[[f"Sobol_m_{p}_PCE" for p in param_names]].to_numpy()
    gen_fuq = generalized_index_from_var_and_sobol(
        df[TIME_COLUMN], df["Var_FUQ"].to_numpy(), fuq_sobol, look_back_window_size)
    gen_pce = generalized_index_from_var_and_sobol(
        df[TIME_COLUMN], df["Var_PCE"].to_numpy(), pce_sobol, look_back_window_size)
    return {p: {"FUQ": float(gen_fuq[-1, j]), "PCE": float(gen_pce[-1, j])}
            for j, p in enumerate(param_names)}


def compute_generalized_sobol_s1(fuq_dir, pce_dir, pce_stem, qoi_column="Q_cms",
                                 agg="median", look_back_window_size="whole"):
    """A single forward-GSA Sobol S1 per parameter from the FUQ run, summarized
    across time via the generalized (variance-weighted) running index rather
    than the raw per-date Sobol_m — the form
    particle_filtering_and_pce_plotting.plot_sensitivity_vs_identifiability's
    own sobol_s1 argument expects.

    Restricts the FUQ record to the PCE run's own date range first, same as
    compare_fuq_and_pce_sobol, so the two stay comparable even when the FUQ
    run spans a longer period. Unlike compare_fuq_and_pce_sobol (which needs
    the two parameter SETS to match exactly, since it plots one against the
    other), this only needs param_names to be a SUBSET of the FUQ run's own
    parameters — e.g. a PCE fit without PM as an uncertain parameter can still
    reuse a 7-parameter FUQ study; PM's own S1 is simply not returned.

    Args:
        fuq_dir, pce_dir, pce_stem, qoi_column: see load_fuq_sobol/load_pce_sobol.
        agg: "median" (default — robust to any single date's window still
            settling) or "last" (matches compare_fuq_and_pce_sobol's own
            "generalized" field, which is the running index's value at the
            final date only).
        look_back_window_size: passed to generalized_index_from_var_and_sobol.

    Returns:
        dict {param_name: float}, one entry per PCE parameter.
    """
    pce_df, param_names = load_pce_sobol(pce_dir, pce_stem)
    date_range = (pce_df[TIME_COLUMN].iloc[0], pce_df[TIME_COLUMN].iloc[-1])
    fuq_df, fuq_param_names = load_fuq_sobol(fuq_dir, qoi_column=qoi_column,
                                             restrict_dates=date_range)
    missing = [p for p in param_names if p not in fuq_param_names]
    if missing:
        raise ValueError(f"PCE parameter(s) {missing} not found among the FUQ "
                         f"run's own parameters {fuq_param_names}.")

    sobol_m_fuq = fuq_df[[f"Sobol_m_{p}" for p in param_names]].to_numpy()
    gen_fuq = generalized_index_from_var_and_sobol(
        fuq_df[TIME_COLUMN], fuq_df["Var"].to_numpy(), sobol_m_fuq, look_back_window_size)

    if agg == "median":
        return {p: float(np.nanmedian(gen_fuq[:, j])) for j, p in enumerate(param_names)}
    if agg == "last":
        return {p: float(gen_fuq[-1, j]) for j, p in enumerate(param_names)}
    raise ValueError(f"agg must be 'median' or 'last', got {agg!r}.")


def compare_fuq_and_pce_sobol(fuq_dir, pce_dir, pce_stem, qoi_column="Q_cms",
                              out_dir=None, gof_list=DEFAULT_GOF_LIST,
                              light_output=False, verbose=True):
    """Orchestrates everything: load PCE first (to get its date range), load
    FUQ restricted to that SAME range up front, align, rank agreement,
    pointwise GoF, generalized-index comparison, overlay plot. Writes
    sobol_comparison_summary.json and the overlay plot to out_dir.

    The two runs' parameter SETS need not match exactly — only param_names
    (the PCE's own) must each exist in the FUQ run's own parameters; a FUQ
    study with extra parameters the PCE never estimated (e.g. PM, when the
    PF was run without it as an uncertain parameter) simply has those extra
    parameters excluded from the comparison, printed as a note rather than
    raising, and recorded under "fuq_only_params_excluded" in the summary.

    Returns:
        dict with "df" (aligned DataFrame), "param_names", "rank_agreement",
        "pointwise_gof", "generalized", "overlay_fig".
    """
    pce_df, param_names = load_pce_sobol(pce_dir, pce_stem)
    date_range = (pce_df[TIME_COLUMN].iloc[0], pce_df[TIME_COLUMN].iloc[-1])
    if verbose:
        print(f"PCE run: {len(pce_df)} dates, {date_range[0].date()} .. {date_range[1].date()}")

    fuq_df, fuq_param_names = load_fuq_sobol(fuq_dir, qoi_column=qoi_column,
                                             restrict_dates=date_range)
    missing = [p for p in param_names if p not in fuq_param_names]
    if missing:
        raise ValueError(f"PCE parameter(s) {missing} not found among the FUQ "
                         f"run's own parameters {fuq_param_names}.")
    fuq_only = [p for p in fuq_param_names if p not in param_names]
    if fuq_only and verbose:
        print(f"Note: FUQ run has {fuq_only} which the PCE was not fit over "
              "(not an uncertain parameter there); excluded from this comparison.")

    df = fuq_df.merge(pce_df, on=TIME_COLUMN, suffixes=("_FUQ", "_PCE"), how="inner")
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    if verbose:
        print(f"Aligned {len(df)} overlapping dates: {df[TIME_COLUMN].iloc[0].date()} .. "
              f"{df[TIME_COLUMN].iloc[-1].date()}")

    rank_agreement = compute_rank_agreement(df, param_names)
    if verbose:
        print(f"\nRank agreement (Spearman over median Sobol_m across parameters): "
              f"r={rank_agreement['spearman_r']:.3f} (p={rank_agreement['p_value']:.3f})")
        print("  median Sobol_m  FUQ vs PCE:")
        for p in param_names:
            print(f"    {p:<6} {rank_agreement['median_FUQ'][p]:.4f}  vs  "
                  f"{rank_agreement['median_PCE'][p]:.4f}")

    pointwise_gof = compute_pointwise_gof(df, param_names, gof_list=gof_list)
    if verbose:
        print("\nPer-parameter pointwise GoF (FUQ='measured', PCE='simulated'):")
        for p, metrics in pointwise_gof.items():
            print(f"  {p:<6} " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    generalized = compute_generalized_comparison(df, param_names)
    if verbose:
        print("\nGeneralized (time-averaged, variance-weighted) Sobol_m, FUQ vs PCE:")
        for p, vals in generalized.items():
            print(f"  {p:<6} {vals['FUQ']:.4f}  vs  {vals['PCE']:.4f}")

    out_dir = str(out_dir or pce_dir)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "sobol_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "date_range": [str(date_range[0].date()), str(date_range[1].date())],
            "n_dates": len(df),
            "param_names": param_names,
            "fuq_only_params_excluded": fuq_only,
            "rank_agreement": rank_agreement,
            "pointwise_gof": pointwise_gof,
            "generalized": generalized,
        }, f, indent=2, default=float)
    if verbose:
        print(f"\n  -> {summary_path}")

    overlay_fig = plot_sobol_overlay(df, param_names, out_dir, light_output=light_output)
    if verbose:
        print(f"  -> {os.path.join(out_dir, 'fuq_vs_pce_sobol_m_overlay.pdf')}")

    return {"df": df, "param_names": param_names, "rank_agreement": rank_agreement,
            "pointwise_gof": pointwise_gof, "generalized": generalized,
            "overlay_fig": overlay_fig}


if __name__ == "__main__":
    fuq_dir = ("data/HBV-SASK-data/paper_uqef_dynamic_sim/hbv_uq_cm4.0333")
    D = ("data/HBV-SASK-data/particle_filtering_model_runs/oldman_basin/"
         "hbvsaskmodel_7d_5000_filtering_gaussian_likelihood_heteroscedastic_two_years_Uniform_5_chains")
    pce_dir = os.path.join(D, "designed_pce_evolving")
    out_dir = os.path.join(pce_dir, "fuq_vs_pce_sobol_comparison")

    result = compare_fuq_and_pce_sobol(
        fuq_dir=fuq_dir, pce_dir=pce_dir,
        pce_stem="evolving_random_2004-10-01_2006-10-01",
        qoi_column="Q_cms", out_dir=out_dir, verbose=True)
