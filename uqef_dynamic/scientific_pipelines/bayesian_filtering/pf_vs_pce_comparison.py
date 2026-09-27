"""Compare the two mean streamflow signals produced for the same particle-filter
run: the PF's own pooled ensemble mean (a one-step-ahead FORECAST, computed
before each day's observation updates the weights) and a PCE
surrogate's mean (built from the PF's per-date posterior over theta, run from a
common warm-up state - informed by data through date t, but not a sequential
forecast the way the PF mean is).

These are not the same kind of quantity, so "which is more accurate" is not
quite the right question - this module answers three, reusing the GoF suite
already in utility.py/objectivefunctions.py rather than new metric code:

  1. observed vs PF mean       - the PF's own forecast skill.
  2. observed vs PCE mean      - does the model, given the PF's inferred theta
                                 at each date, reproduce observed flow.
  3. PF mean vs PCE mean       - how much do the two independently-derived
                                 pipelines agree with EACH OTHER, separate from
                                 either one's accuracy against truth.

Data sources (must already share one common date grid - true for a pooled PF
run and the evolving-posterior PCE built from it):
  pooled_dir/pooled_chains.npz          -> "pooled_mean", "dates"
  pooled_dir/averaged_and_simulated.pkl -> "observed_streamflow"
  pce_dir/pce_<pce_stem>.npz            -> "E", "dates"
"""

import os
import json

import numpy as np
import pandas as pd

from uqef_dynamic.utils import utility
from uqef_dynamic.scientific_pipelines.bayesian_filtering.particle_filtering_and_pce_plotting import (
    plot_overlay, plot_residuals)

__all__ = ["load_pf_pce_observed", "compute_pairwise_gof", "compute_monthly_gof",
          "plot_overlay", "plot_residuals", "compare_pf_and_pce_means"]

TIME_COLUMN = "TimeStamp"
DEFAULT_GOF_LIST = ["RMSE", "NSE", "KGE", "PBIAS", "CorrelationCoefficient"]

# Preference order for PF's own uncertainty band: 10-90 matches the PCE's
# fixed P10/P90 exactly, so it is used whenever pool_chain_results happened
# to save it; otherwise fall back to the widest inner band actually saved.
_PF_BAND_PREFERENCE = ((10, 90), (5, 95), (25, 75))


def _pick_pf_band(available_files):
    """Which (lo, hi) percentile pair in pooled_chains.npz to use as PF's
    uncertainty band - see _PF_BAND_PREFERENCE."""
    for lo, hi in _PF_BAND_PREFERENCE:
        if f"pct_{lo}" in available_files and f"pct_{hi}" in available_files:
            return lo, hi
    raise ValueError(
        f"pooled_chains.npz has none of {_PF_BAND_PREFERENCE}; re-pool with "
        "percentiles including at least one such pair to get a PF band.")


def load_pf_pce_observed(pooled_dir, pce_dir, pce_stem):
    """Load and align the PF pooled mean+band, PCE mean+band, and observed streamflow.

    Args:
        pooled_dir: directory holding pooled_chains.npz and averaged_and_simulated.pkl
                   (a pool_chain_results output, or a single chain's own working_dir).
        pce_dir:    directory holding pce_<pce_stem>.npz (a run_pce_learning /
                   run_designed_sample_pce_evolving output).
        pce_stem:   the stem naming that file, e.g. "evolving_random_2004-10-01_2006-10-01".

    Returns:
        (df, pf_band):
          df: DataFrame with columns [TIME_COLUMN, "observed", "PF_mean",
              "PF_P_lo", "PF_P_hi", "PCE_mean", "PCE_P10", "PCE_P90"],
              inner-joined on date (so it is safe even if the two runs' date
              ranges do not match exactly - only the overlap is kept) and
              sorted by date. Rows with any NaN in observed/PF_mean/PCE_mean
              are dropped, since the GoF functions below are not uniformly
              NaN-safe.
          pf_band: (lo, hi) - the percentile pair actually backing
              PF_P_lo/PF_P_hi (see _pick_pf_band).
    """
    pf = np.load(os.path.join(str(pooled_dir), "pooled_chains.npz"), allow_pickle=True)
    lo, hi = _pick_pf_band(pf.files)
    pf_df = pd.DataFrame({
        TIME_COLUMN: pd.to_datetime([str(x) for x in pf["dates"]]),
        "PF_mean": np.asarray(pf["pooled_mean"], dtype=np.float64),
        "PF_P_lo": np.asarray(pf[f"pct_{lo}"], dtype=np.float64),
        "PF_P_hi": np.asarray(pf[f"pct_{hi}"], dtype=np.float64),
    })

    pce = np.load(os.path.join(str(pce_dir), f"pce_{pce_stem}.npz"), allow_pickle=True)
    pce_df = pd.DataFrame({
        TIME_COLUMN: pd.to_datetime([str(x) for x in pce["dates"]]),
        "PCE_mean": np.asarray(pce["E"], dtype=np.float64),
        "PCE_P10": np.asarray(pce["P10"], dtype=np.float64),
        "PCE_P90": np.asarray(pce["P90"], dtype=np.float64),
    })

    obs_df = pd.read_pickle(os.path.join(str(pooled_dir), "averaged_and_simulated.pkl"),
                            compression="gzip")
    obs_df = obs_df[["observed_streamflow"]].reset_index()
    obs_df.columns = [TIME_COLUMN, "observed"]
    obs_df[TIME_COLUMN] = pd.to_datetime(obs_df[TIME_COLUMN])

    df = obs_df.merge(pf_df, on=TIME_COLUMN, how="inner").merge(pce_df, on=TIME_COLUMN, how="inner")
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    n_before = len(df)
    df = df.dropna(subset=["observed", "PF_mean", "PCE_mean"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"load_pf_pce_observed: dropped {n_before - len(df)}/{n_before} "
              f"date(s) with a NaN in observed/PF_mean/PCE_mean.")
    return df, (lo, hi)


def _gof_pair(df, col_a, col_b, gof_list):
    """One pairwise GoF dict, col_a treated as 'measured', col_b as 'simulated'."""
    return utility.calculateGoodnessofFit_simple(
        measuredDF=df[[TIME_COLUMN, col_a]], simulatedDF=df[[TIME_COLUMN, col_b]],
        gof_list=list(gof_list),
        measuredDF_time_column_name=TIME_COLUMN, simulatedDF_time_column_name=TIME_COLUMN,
        measuredDF_column_name=col_a, simulatedDF_column_name=col_b)


def compute_pairwise_gof(df, gof_list=DEFAULT_GOF_LIST, burn_in_days=0):
    """The three pairwise GoF tables over the aligned record.

    Args:
        burn_in_days: drop this many dates from the START of `df` before
            computing (default 0, i.e. no burn-in). NOT a cosmetic knob: date 0
            of a fresh particle filter run is drawn straight from the prior with
            no assimilation history yet, and can produce a wildly unrepresentative
            forecast that dominates a squared-error metric on an otherwise-good
            record

    Returns:
        dict: {"observed_vs_PF": {...}, "observed_vs_PCE": {...}, "PF_vs_PCE": {...}}
    """
    if burn_in_days:
        df = df.iloc[burn_in_days:]
    return {
        "observed_vs_PF": _gof_pair(df, "observed", "PF_mean", gof_list),
        "observed_vs_PCE": _gof_pair(df, "observed", "PCE_mean", gof_list),
        "PF_vs_PCE": _gof_pair(df, "PF_mean", "PCE_mean", gof_list),
    }


def compute_monthly_gof(df, gof_list=DEFAULT_GOF_LIST, burn_in_days=0):
    """The same three pairs, recomputed within each calendar month (year+month
    - NOT month-of-year. One aggregate number over the whole record hides exactly the
    regime-dependence this kind of comparison is usually run to find.

    burn_in_days: see compute_pairwise_gof - applied once, up front, before
    grouping by month (so it only ever affects the first month's group, same
    as it does for the whole-record version).

    NSE/KGE/CorrelationCoefficient can be extremely unstable within a single
    low-flow month: their denominator is the observed variance WITHIN that one
    narrow window, which can be tiny, so a small absolute error still produces
    huge negative NSE or even -inf. RMSE/PBIAS stay interpretable there; treat
    monthly NSE/KGE as informative mainly in months with real flow variability.

    Returns:
        DataFrame indexed by month (Timestamp, month start) with one column
        per "<pair>_<metric>", e.g. "observed_vs_PF_RMSE".
    """
    if burn_in_days:
        df = df.iloc[burn_in_days:]
    rows = {}
    for period, group in df.groupby(df[TIME_COLUMN].dt.to_period("M")):
        gof = compute_pairwise_gof(group, gof_list=gof_list)
        rows[period.to_timestamp()] = {
            f"{pair}_{metric}": value
            for pair, metrics in gof.items() for metric, value in metrics.items()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def compare_pf_and_pce_means(pooled_dir, pce_dir, pce_stem, out_dir=None,
                             gof_list=DEFAULT_GOF_LIST, burn_in_days=0,
                             plot_forcing_data=False, configuration_file=None,
                             inputModelDir=None, basin=None,
                             light_output=False, verbose=True):
    """Orchestrates all four pieces: load+align, whole-record GoF, monthly GoF,
    overlay plot, residual plot. Writes gof_summary.json, monthly_gof.csv, and
    the two plots (.html + .pdf) to out_dir.

    burn_in_days: see compute_pairwise_gof. Applied ONLY to the saved/returned
    GoF numbers, never to `df`/the plots - the plots always show the full
    record (including any early pathological date) so nothing is hidden; only
    the aggregate metrics, which a single outlier can dominate through a
    squared-error term, are computed with it excluded. Both the with- and
    without-burn-in whole-record GoF are printed and saved side by side
    specifically so a flattering number is never reported alone.

    plot_forcing_data, configuration_file, inputModelDir, basin: forwarded to
    plot_overlay (see its docstring) — add temperature/precipitation "wall"
    rows ahead of the streamflow overlay panel.

    Returns:
        dict with "df" (the aligned DataFrame, full record), "pf_band" (the
        percentile pair backing the PF band — see load_pf_pce_observed),
        "gof" (whole-record pairwise dict, with burn_in_days applied),
        "gof_full_record" (same, with NO burn-in, for comparison),
        "monthly_gof" (DataFrame), "overlay_fig", "residual_fig".
    """
    out_dir = str(out_dir or pce_dir)
    os.makedirs(out_dir, exist_ok=True)

    df, pf_band = load_pf_pce_observed(pooled_dir, pce_dir, pce_stem)
    if verbose:
        print(f"Aligned {len(df)} dates: {df[TIME_COLUMN].iloc[0].date()} .. "
              f"{df[TIME_COLUMN].iloc[-1].date()}")

    gof_full = compute_pairwise_gof(df, gof_list=gof_list, burn_in_days=0)
    gof = (compute_pairwise_gof(df, gof_list=gof_list, burn_in_days=burn_in_days)
          if burn_in_days else gof_full)
    if verbose:
        print("\nWhole-record GoF (ALL dates):")
        for pair, metrics in gof_full.items():
            print(f"  {pair}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if burn_in_days:
            print(f"\nWhole-record GoF (first {burn_in_days} date(s) excluded):")
            for pair, metrics in gof.items():
                print(f"  {pair}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    gof_path = os.path.join(out_dir, "gof_summary.json")
    with open(gof_path, "w") as f:
        json.dump({"burn_in_days": burn_in_days, "all_dates": gof_full,
                   f"excl_first_{burn_in_days}_dates": gof}, f, indent=2, default=float)
    if verbose:
        print(f"  -> {gof_path}")

    monthly_gof = compute_monthly_gof(df, gof_list=gof_list, burn_in_days=burn_in_days)
    monthly_path = os.path.join(out_dir, "monthly_gof.csv")
    monthly_gof.to_csv(monthly_path)
    if verbose:
        print(f"  -> {monthly_path}")

    overlay_fig = plot_overlay(df, out_dir, pf_band=pf_band, light_output=light_output,
                               plot_forcing_data=plot_forcing_data,
                               configuration_file=configuration_file,
                               inputModelDir=inputModelDir, basin=basin)
    residual_fig = plot_residuals(df, out_dir, light_output=light_output)
    if verbose:
        print(f"  -> {os.path.join(out_dir, 'pf_vs_pce_streamflow_overlay.pdf')}")
        print(f"  -> {os.path.join(out_dir, 'pf_vs_pce_residuals.pdf')}")

    return {"df": df, "pf_band": pf_band, "gof": gof, "gof_full_record": gof_full,
            "monthly_gof": monthly_gof, "overlay_fig": overlay_fig, "residual_fig": residual_fig}


if __name__ == "__main__":
    D = ("data/HBV-SASK-data/particle_filtering_model_runs/oldman_basin/"
         "hbvsaskmodel_7d_5000_filtering_gaussian_likelihood_heteroscedastic_two_years_Uniform_5_chains")
    pce_dir = os.path.join(D, "designed_pce_evolving")
    out_dir = os.path.join(pce_dir, "pf_vs_pce_comparison")

    result = compare_pf_and_pce_means(
        pooled_dir=D, pce_dir=pce_dir,
        pce_stem="evolving_random_2004-10-01_2006-10-01",
        out_dir=out_dir, burn_in_days=1,
        plot_forcing_data=True,
        configuration_file="data/configurations/configuration_hbv_sask_PF_two_years.json",
        inputModelDir="data/HBV-SASK-data",
        verbose=True)
