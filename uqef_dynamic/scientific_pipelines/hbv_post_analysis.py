"""
HBV-SASK UQ Post-Analysis
====================================
Reads the output of a completed UQEF-Dynamic UQ run (Monte Carlo, PCE, Saltelli,
or any other UQ analysis applied to the HBV-SASK hydrological model),
reconstructs the statistics object, and produces a full suite of UQ analysis
plots together with a time-series analysis of the model error signal (mean
prediction E − observed streamflow).

All figures are written to SAVE_DIR both as PDF (for publication) and as
interactive HTML files (via Plotly).

Expected input (working_dir)
-----------------------------
The script reads the standard artefact files produced by a UQEF-Dynamic
simulation run. These are located with
``utility.get_dict_with_output_file_paths_based_on_workingDir(working_dir)``
and include:

  args.pkl                     – argparse namespace of the original run
  configuration_object.pkl     – dill-pickled configuration dict
  time_info.txt                – (optional) wall-clock timing of the run
  simulation_nodes.pkl         – chaospy SimulationNodes object (quadrature
                                 points / samples and joint distributions)
  df_all_index_parameter_values.pkl       – parameter sample matrix (one row
                                            per model run)
  df_all_index_parameter_gof_values.pkl   – same rows + GoF metrics columns
  statistics_*.pkl (or per-timestep files) – pre-computed UQ statistics
                                             (mean, variance, percentiles, …)

Optionally:
  df_all_simulations.pkl                  – full (N_runs × T) output matrix;
                                            large, loaded only with --read_sims
  df_index_parameter_conditioned.pkl      – parameter samples after GoF
  df_index_parameter_gof_conditioned.pkl    conditioning (if computed)

Outputs (save_dir)
-------------------
Parameter / GoF analysis
  prior_dist_of_params.*          – histograms of all parameter priors
  pdf_of_gofs.* / cdf_of_gofs.*  – KDE-smoothed PDF and CDF of each GoF metric
  dist_params_{gof}_{cmp}.*      – conditional parameter distributions
  parallel_params_vs_{gof}.*     – parallel-coordinate plots coloured by GoF

UQ time-series (--plot_time_series)
  simulation_big_plot_{qoi}_2std.*    – mean ± 2σ and P10/P90 vs observed
  forcing_measured_and_mean_data.*    – forcing + mean prediction overview

Error signal analysis
  error_time_series.*         – mean error over time + 30-day rolling mean/std
  error_distribution.*        – histogram, monthly boxplot, empirical CDF
  error_acf_pacf.*            – ACF and PACF up to lag 60
  error_psd.*                 – Welch power spectral density
  error_vs_forcing.*          – cross-correlation and scatter vs precipitation
                                and temperature
  error_vs_spread.*           – |bias| vs UQ spread over time

Console output
  GoF summary statistics, best-parameter-set rows, P-factor, R-factor,
  error summary (bias, MAE, RMSE, autocorrelations), stationarity test results.

Dependencies
------------
  numpy, pandas, matplotlib, chaospy, plotly, scipy, statsmodels, dill
  uqef_dynamic (this project)
  kaleido  (optional, required for PDF export of Plotly figures)

Usage examples
--------------
Run with paths hard-coded at the top of this file::

    python hbv_post_analysis.py

Override paths on the command line::

    python hbv_post_analysis.py \\
        --working_dir /work/ga45met/paper_uqef_dynamic_sim/hbvsask_runs/hbv_uq_cm4.0328 \\
        --save_dir    /work/ga45met/paper_uqef_dynamic_sim/hbv_sask/oldman_2004_2007/ \\
        --hbv_data    /work/ga45met/Hydro_Models/HBV-SASK-data \\
        --qoi         Q_cms

Include parameter analysis and UQ time-series plots::

    python hbv_post_analysis.py \\
        --working_dir /path/to/run \\
        --save_dir    /path/to/output \\
        --plot_parameter_analysis \\
        --plot_time_series

Load individual simulation trajectories (warning: large file)::

    python hbv_post_analysis.py --working_dir /path/to/run --read_sims
"""

import argparse
import pathlib
import pickle
import warnings

import dill
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import chaospy as cp
import plotly.offline as pyo
from scipy import signal as sig, stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

from uqef_dynamic.utils import utility, uqef_dynamic_utils, create_stat_object
# from uqef_dynamic.models.hbv_sask import hbvsask_utility as hbv

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — edit these paths before running
# =============================================================================
BASE_SOURCE_PATH = pathlib.Path.cwd().parents[1]  # assumes this script is in uqef_dynamic/scientific_pipelines
print(f"Base source path: {BASE_SOURCE_PATH}")
HBV_MODEL_DATA_PATH = BASE_SOURCE_PATH / "data" / "HBV-SASK-data"
BASIS_WORKING_DIR   = HBV_MODEL_DATA_PATH / "paper_uqef_dynamic_sim"
WORKING_DIR         = BASIS_WORKING_DIR / "hbv_uq_cm4.0327"   # Banff, 6 years, ensemble analysis
SAVE_DIR            = HBV_MODEL_DATA_PATH / "banff_post_analysis_output"

# Which QoI to use for error analysis
QOI_COLUMN = "Q_cms"

# Loading full simulation results can require a lot of memory — set to True only if needed
READ_SIMULATION_RESULTS = False
READ_STATE_RESULTS      = False
# =============================================================================


def _save_plotly(fig, name, save_dir, width=1200, height=None):
    pyo.plot(fig, filename=str(save_dir / f"{name}.html"), auto_open=False)
    kwargs = {"format": "pdf", "width": width}
    if height:
        kwargs["height"] = height
    try:
        fig.write_image(str(save_dir / f"{name}.pdf"), **kwargs)
    except Exception as e:
        print(f"  [warn] PDF export failed for {name}: {e}")


def _save_mpl(fig, name, save_dir):
    fig.savefig(save_dir / f"{name}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 1. Load saved run artefacts
# =============================================================================

def load_run_artefacts(working_dir):
    """Read the core artefact files produced by a UQEF-Dynamic run.

    Parameters
    ----------
    working_dir : pathlib.Path
        Root directory of the simulation run. Must contain ``args.pkl``
        and ``configuration_object.pkl`` (names resolved by
        ``utility.get_dict_with_output_file_paths_based_on_workingDir``).

    Returns
    -------
    uqsim_args_dict : dict
        Flat dict of the original argparse namespace (model name, sampling
        method, number of runs, QoI settings, …).
    configuration : dict
        UQEF-Dynamic configuration object (time settings, model settings,
        simulation settings, parameter bounds, …).
    file_paths : dict
        Mapping of logical name → pathlib.Path for every standard output
        file in working_dir (nodes, parameter dataframes, statistics, …).
    """
    print("Loading run artefacts...")
    file_paths = utility.get_dict_with_output_file_paths_based_on_workingDir(working_dir)

    with open(file_paths["args_file"], "rb") as f:
        uqsim_args = pickle.load(f)
    uqsim_args_dict = vars(uqsim_args)

    with open(file_paths["configuration_object_file"], "rb") as f:
        configuration = dill.load(f)

    if file_paths["time_info_file"].is_file():
        print("  Time info:", file_paths["time_info_file"].read_text().strip())

    return uqsim_args_dict, configuration, file_paths


# =============================================================================
# 2. Load parameter / GoF data frames
# =============================================================================

def load_parameter_data(file_paths, working_dir, configuration):
    """Load simulation nodes, parameter samples, and GoF dataframes.

    All files are optional — missing ones are handled gracefully and the
    corresponding return value is set to ``None``.

    Parameters
    ----------
    file_paths : dict
        As returned by ``load_run_artefacts``.
    working_dir : pathlib.Path
        Run directory; used to look for conditioned-parameter files that are
        not part of the standard ``file_paths`` dict.
    configuration : dict
        Configuration object; provides parameter names as a fallback when the
        parameter dataframe is unavailable.

    Returns
    -------
    simulation_nodes : object
        Chaospy ``SimulationNodes`` object containing quadrature nodes/samples
        and the joint distribution.
    df_index_parameter : pd.DataFrame or None
        One row per model run, columns are parameter names.
    df_gof : pd.DataFrame or None
        Same rows as ``df_index_parameter`` plus columns for each GoF metric
        (MAE, MSE, RMSE, NSE, LogNSE, KGE, …).
    params_list : list[str]
        Names of uncertain parameters, derived from ``df_gof`` or the
        configuration object.
    gof_list : list[str]
        Names of GoF metric columns present in ``df_gof``.
    df_gof_conditioned : pd.DataFrame or None
        GoF dataframe filtered to runs that pass a behavioural threshold (e.g.
        NSE > 0.6).  ``None`` if the file does not exist.
    """
    print("Loading parameter and GoF data...")

    with open(file_paths["nodes_file"], "rb") as f:
        simulation_nodes = pickle.load(f)

    df_index_parameter = (
        pd.read_pickle(file_paths["df_index_parameter_file"], compression="gzip")
        if file_paths["df_index_parameter_file"].is_file() else None
    )
    params_list = (
        utility._get_parameter_columns_df_index_parameter_gof(df_index_parameter)
        if df_index_parameter is not None
        else [p["name"] for p in configuration["parameters"]]
    )

    if file_paths["df_index_parameter_gof_file"].is_file():
        df_gof = pd.read_pickle(file_paths["df_index_parameter_gof_file"], compression="gzip")
        gof_list = utility._get_gof_columns_df_index_parameter_gof(df_gof)
    else:
        print("  [warn] GoF file not found.")
        df_gof, gof_list = None, []

    # Conditioned versions (may not exist)
    cond_gof_file = working_dir / utility.DF_INDEX_PARAMETER_GOF_CONDITIONED_FILE
    df_gof_conditioned = (
        pd.read_pickle(cond_gof_file, compression="gzip") if cond_gof_file.is_file() else None
    )

    return simulation_nodes, df_index_parameter, df_gof, params_list, gof_list, df_gof_conditioned


# =============================================================================
# 3. Parameter & GoF analysis plots
# =============================================================================

def plot_parameter_analysis(df_gof, params_list, gof_list, df_gof_conditioned, save_dir):
    """Produce the full set of parameter and GoF distribution plots.

    Skips silently if ``df_gof`` is ``None``.

    Plots produced
    --------------
    prior_dist_of_params        – histogram of every parameter's prior sample
    pdf_of_gofs / cdf_of_gofs   – KDE-smoothed PDF and CDF of each GoF metric
    dist_params_{gof}_{cmp}     – parameter histograms conditioned on a GoF
                                   threshold (NSE > 0.6, RMSE < 20, KGE > 0.6)
    parallel_params_vs_{gof}    – parallel-coordinate plot coloured by RMSE /
                                   NSE / KGE

    Parameters
    ----------
    df_gof : pd.DataFrame or None
    params_list : list[str]
    gof_list : list[str]
    df_gof_conditioned : pd.DataFrame or None
    save_dir : pathlib.Path
    """
    if df_gof is None:
        print("  [skip] No GoF data — skipping parameter analysis plots.")
        return

    print("Plotting parameter analysis...")

    # Prior parameter distributions
    fig = utility.plot_subplot_params_hist_from_df(df_gof)
    fig.update_layout(title=None, margin=dict(t=10, b=10, l=50, r=10))
    _save_plotly(fig, "prior_dist_of_params", save_dir)

    # GoF PDF and CDF
    for ylabel, method in [("PDF", "pdf"), ("CDF", "cdf")]:
        fig_mpl, axs = plt.subplots(1, len(gof_list), figsize=(5 * len(gof_list), 5))
        if len(gof_list) == 1:
            axs = [axs]
        for i, gof in enumerate(gof_list):
            vals = df_gof[gof].dropna().values
            t = np.linspace(vals.min() * 0.999, vals.max() * 1.001, 500)
            kde = cp.GaussianKDE(vals, h_mat=0.005**2)
            getattr(axs[i], "hist" if method == "pdf" else "plot")(
                *([vals, 100, True, 0.5] if method == "pdf" else [t, kde.cdf(t)])
            )
            if method == "pdf":
                axs[i].plot(t, kde.pdf(t), label=f"KDE")
            axs[i].set_xlabel(gof); axs[i].grid(alpha=0.3)
        axs[0].set_ylabel(ylabel)
        plt.tight_layout()
        _save_mpl(fig_mpl, f"{method}_of_gofs", save_dir)

    # Conditional distributions
    for gof, threshold, comparison in [
        ("NSE", 0.6, "greater"), ("RMSE", 20.0, "smaller"), ("KGE", 0.6, "greater")
    ]:
        if gof not in gof_list:
            continue
        fig = utility.plot_subplot_params_hist_from_df_conditioned(
            df_gof, name_of_gof_column=gof,
            threshold_gof_value=threshold, comparison=comparison)
        fig.update_layout(title=None, margin=dict(t=10, b=10, l=50, r=10))
        _save_plotly(fig, f"dist_params_{gof.lower()}_{comparison}", save_dir)

    # Parallel coordinate plots — full ensemble
    for gof in ["RMSE", "NSE", "KGE"]:
        if gof not in gof_list:
            continue
        fig = utility.plot_parallel_params_vs_gof(
            df_gof, name_of_gof_column=gof, list_of_params=params_list)
        fig.update_layout(title=None, margin=dict(t=50, b=20, l=50, r=20))
        _save_plotly(fig, f"parallel_params_vs_{gof.lower()}", save_dir)

    # Hmmm, remove this, too specific...
    # Parallel coordinate plot — conditioned subset only
    if df_gof_conditioned is not None and "RMSE" in gof_list:
        fig = utility.plot_parallel_params_vs_gof(
            df_gof_conditioned, name_of_gof_column="RMSE", list_of_params=params_list)
        fig.update_layout(title=None, margin=dict(t=50, b=20, l=50, r=20))
        _save_plotly(fig, "parallel_params_vs_rmse_conditioned", save_dir)

    # Print summary
    print("\n  GoF summary:")
    print(df_gof[gof_list].describe().to_string())
    for metric, fn in [("NSE", "idxmax"), ("RMSE", "idxmin"), ("KGE", "idxmax")]:
        if metric in df_gof.columns:
            idx = getattr(df_gof[metric], fn)()
            print(f"  Best {metric}: {df_gof.loc[idx, metric]:.4f}")


# =============================================================================
# 4. Build statistics object and merged dataframe
# =============================================================================

def build_statistics(configuration, uqsim_args_dict, working_dir,
                      hbv_model_data_path, basin, df_simulation_result=None):
    """Reconstruct the UQEF-Dynamic statistics object and build the merged dataframe.

    Reads the per-timestep (or single-file) statistics dictionaries from
    ``working_dir``, extends the statistics object with them, attaches
    measured streamflow and forcing data, and returns a flat DataFrame
    suitable for plotting and error analysis.

    Note: the two lines setting ``stat_obj.inputModelDir_basin`` and calling
    ``stat_obj.get_measured_data`` are the only HBV-SASK-specific parts of
    this function.  For a different model, replace those two lines with the
    appropriate call to load observed QoI data.

    Parameters
    ----------
    configuration : dict
        As returned by ``load_run_artefacts``.
    uqsim_args_dict : dict
        As returned by ``load_run_artefacts``.
    working_dir : pathlib.Path
        Run directory containing the statistics output files.
    hbv_model_data_path : pathlib.Path
        Root of the HBV-SASK data directory (contains per-basin subdirs).
        Used to locate the measured streamflow CSV.
    basin : str
        HBV-SASK basin name, e.g. ``"Oldman_Basin"`` or ``"Banff_Basin"``.
    df_simulation_result : pd.DataFrame or None
        Full simulation result matrix (N_runs × T).  Pass ``None`` (default)
        to skip — the statistics object is still fully populated from the
        pre-computed statistics files.

    Returns
    -------
    stat_obj : HBVSASKStatistics
        Fully populated statistics object (has ``df_statistics``,
        ``df_measured``, ``forcing_df``, etc. as attributes).
    df_stats : pd.DataFrame
        Merged DataFrame with columns: TimeStamp, qoi, E, Var, StdDev,
        P10, P90, Skew, Kurt (availability depends on what was computed),
        measured, precipitation, temperature, and derived std-band columns
        E_minus_std, E_plus_std, E_minus_2std, E_plus_2std.
    """
    print("Building statistics object...")

    stat_obj = create_stat_object.create_statistics_object(
        configuration, uqsim_args_dict, working_dir)

    statistics_dict = uqef_dynamic_utils.read_all_saved_statistics_dict(
        workingDir=working_dir,
        list_qoi_column=stat_obj.list_qoi_column,
        single_timestamp_single_file=uqsim_args_dict.get(
            "instantly_save_results_for_each_time_step", False),
        throw_error=True)

    uqef_dynamic_utils.extend_statistics_object(
        statisticsObject=stat_obj,
        statistics_dictionary=statistics_dict,
        df_simulation_result=df_simulation_result,
        get_measured_data=False,
        get_unaltered_data=False)

    # TODO: Only this is HBV model specific
    stat_obj.inputModelDir_basin = hbv_model_data_path / basin

    # TODO: This is also hardcoded - qoi_column_name="Q_cms"
    stat_obj.get_measured_data(
        timestepRange=(stat_obj.timesteps_min, stat_obj.timesteps_max),
        qoi_column_name="Q_cms",
        transforme_mesured_data_as_original_model="False")

    stat_obj.create_df_from_statistics_data()
    stat_obj.get_forcing_data(time_column_name=utility.TIME_COLUMN_NAME)

    df = pd.merge(
        stat_obj.df_statistics, stat_obj.forcing_df,
        left_on=stat_obj.time_column_name, right_index=True)
    df[utility.TIME_COLUMN_NAME] = pd.to_datetime(df[utility.TIME_COLUMN_NAME])
    df = df.sort_values(utility.TIME_COLUMN_NAME).reset_index(drop=True)

    # Derive standard deviation bands
    if "StdDev" not in df.columns and "Var" in df.columns:
        df["StdDev"] = np.sqrt(df["Var"])
    if "StdDev" in df.columns:
        df["E_minus_std"]  = df["E"] - df["StdDev"]
        df["E_plus_std"]   = df["E"] + df["StdDev"]
        df["E_minus_2std"] = df["E"] - 2 * df["StdDev"]
        df["E_plus_2std"]  = df["E"] + 2 * df["StdDev"]
    for col in ["E", "E_minus_std", "E_minus_2std", "P10"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return stat_obj, df


# =============================================================================
# 5. UQ time-series plots
# =============================================================================

def plot_time_series(stat_obj, df, save_dir):
    """Plot UQ time-series for every QoI in the statistics object.

    Produces one combined plot per QoI showing the mean prediction (E),
    ±2σ band, P10/P90 percentile band, and observed streamflow,
    plus a separate forcing overview figure.

    Which quantities are drawn is controlled by ``dict_what_to_plot``
    inside this function.  Edit that dict to toggle individual bands
    (e.g. set ``"Sobol_m": True`` if Sobol indices were computed).

    Parameters
    ----------
    stat_obj : HBVSASKStatistics
        Populated statistics object.
    df : pd.DataFrame
        As returned by ``build_statistics``.
    save_dir : pathlib.Path
    """
    print("Plotting UQ time series...")

    dict_what_to_plot = {
        "E_minus_std": False, "E_plus_std": False,
        "E_minus_2std": True,  "E_plus_2std": True,
        "P10": True,  "P90": True,
        "StdDev": True, "Skew": False, "Kurt": False,
        "Sobol_m": False, "Sobol_m2": False, "Sobol_t": False,
    }

    for qoi in stat_obj.list_qoi_column:
        subset = df[df["qoi"] == qoi]
        fig = uqef_dynamic_utils.plotting_function_single_qoi(
            subset, single_qoi=qoi,
            dict_what_to_plot=dict_what_to_plot,
            directory=save_dir,
            fileName=f"simulation_big_plot_{qoi}_2std.html",
            dtick="M12")
        fig.update_layout(title=None, margin=dict(t=10, b=10, l=20, r=20))
        try:
            fig.write_image(str(save_dir / f"simulation_big_plot_{qoi}_2std.pdf"),
                            format="pdf", width=1200)
        except Exception as e:
            print(f"  [warn] PDF export failed for {qoi} time series: {e}")

    fig, _ = uqef_dynamic_utils.plot_forcing_mean_predicted_and_observed_all_qoi(
        stat_obj, directory=save_dir, fileName="forcing_measured_and_mean_data.html")
    fig.update_layout(title=None, margin=dict(t=10, b=10, l=20, r=20))
    try:
        fig.write_image(str(save_dir / "forcing_measured_and_mean_data.pdf"),
                        format="pdf", height=1000, width=1100)
    except Exception as e:
        print(f"  [warn] PDF export failed for forcing plot: {e}")


# =============================================================================
# 6. GoF metrics and P/R factors
# =============================================================================

def compute_gof_and_pr(stat_obj, df, qoi_column):
    """Compute and print GoF metrics, P-factor, and R-factor for the UQ run.

    The P-factor is the fraction of observed timesteps bracketed by the
    P10–P90 uncertainty band.  The R-factor is the mean width of that band
    divided by the standard deviation of the observations; values close to 1
    indicate a well-calibrated uncertainty estimate.

    Parameters
    ----------
    stat_obj : HBVSASKStatistics
        Populated statistics object.
    df : pd.DataFrame
        As returned by ``build_statistics``.
    qoi_column : str
        Name of the quantity of interest column, e.g. ``"Q_cms"``.

    Returns
    -------
    p_factor : float
    r_factor : float
    """
    print("\nGoF metrics:")
    stat_obj.compute_gof_over_different_time_series_single_qoi(
        objective_function=stat_obj.objective_function,
        qoi_column=qoi_column)

    p = stat_obj.calculate_p_factor_single_qoi(
        qoi_column=qoi_column, df_statistics=df,
        column_lower_uncertainty_bound="P10",
        column_upper_uncertainty_bound="P90",
        observed_column="measured")

    _, _, _, std_obs = stat_obj.compute_stat_of_uncertainty_band(
        qoi_column=qoi_column, df_statistics=df,
        column_lower_uncertainty_bound="P10",
        column_upper_uncertainty_bound="P90",
        observed_column="measured")

    mean_band = (df[df["qoi"] == qoi_column]["P90"] - df[df["qoi"] == qoi_column]["P10"]).mean()
    r_factor = mean_band / std_obs if std_obs and std_obs > 0 else float("nan")

    print(f"  P-factor: {p:.4f}")
    print(f"  R-factor: {r_factor:.4f}")
    return p, r_factor


# =============================================================================
# 7. Error signal construction
# =============================================================================

def build_error_signal(df, qoi_column):
    """Compute the error signal (mean prediction E − observed) for a single QoI.

    Adds derived columns to the returned dataframe so that UQ spread
    can be overlaid on the error plots without re-joining.

    Parameters
    ----------
    df : pd.DataFrame
        As returned by ``build_statistics``.
    qoi_column : str
        Name of the quantity of interest, e.g. ``"Q_cms"``.

    Returns
    -------
    df_err : pd.DataFrame
        Subset of ``df`` for ``qoi_column`` with additional columns:
        ``mean_error`` (E − measured),
        ``error_e_minus_std`` / ``error_e_plus_std`` (if StdDev available),
        ``error_p10`` / ``error_p90`` (if percentiles available),
        ``month`` (integer 1–12 for seasonal analysis).
    error_ts : pd.Series
        ``mean_error`` indexed by TimeStamp, NaN rows dropped.  Ready for
        time-series analysis (ACF, PSD, stationarity tests, …).
    """
    df_err = (df[df["qoi"] == qoi_column]
              .copy()
              .sort_values(utility.TIME_COLUMN_NAME)
              .reset_index(drop=True))

    df_err["mean_error"] = df_err["E"] - df_err["measured"]
    if "StdDev" in df_err.columns:
        df_err["error_e_minus_std"] = df_err["mean_error"] - df_err["StdDev"]
        df_err["error_e_plus_std"]  = df_err["mean_error"] + df_err["StdDev"]
    if "P10" in df_err.columns:
        df_err["error_p10"] = df_err["P10"] - df_err["measured"]
        df_err["error_p90"] = df_err["P90"] - df_err["measured"]
    df_err["month"] = pd.to_datetime(df_err[utility.TIME_COLUMN_NAME]).dt.month

    error_ts = df_err.set_index(utility.TIME_COLUMN_NAME)["mean_error"].dropna()
    return df_err, error_ts


def print_error_summary(error_ts, qoi_column=""):
    """Print descriptive statistics and autocorrelations of the error signal.

    Parameters
    ----------
    error_ts : pd.Series
        As returned by ``build_error_signal``.
    qoi_column : str, optional
        Name of the quantity of interest (used as label in output lines).
    """
    label = f" [{qoi_column}]" if qoi_column else ""
    print("\n=== Error Signal Summary ===")
    print(error_ts.describe().to_string())
    print(f"\n  Bias (mean error) : {error_ts.mean():.4f}{label}")
    print(f"  MAE               : {error_ts.abs().mean():.4f}{label}")
    print(f"  RMSE              : {np.sqrt((error_ts**2).mean()):.4f}{label}")
    print(f"  Lag-1  autocorr   : {error_ts.autocorr(1):.3f}")
    print(f"  Lag-7  autocorr   : {error_ts.autocorr(7):.3f}")
    print(f"  Lag-30 autocorr   : {error_ts.autocorr(30):.3f}")


# =============================================================================
# 8. Error time-series analysis plots
# =============================================================================

def plot_error_time_series(df_err, error_ts, save_dir, qoi_column=""):
    """Plot the error signal over time together with a rolling mean.

    Answers: *When does the model over/underpredict? Is there a temporal trend?*

    Top panel shows the raw mean error (E − observed) with optional UQ spread
    bands (±StdDev and P10/P90) to indicate how much of the bias is covered
    by parameter uncertainty.  Bottom panel shows a 30-day centred
    rolling mean ± rolling std, which smooths out day-to-day noise and makes
    seasonal or inter-annual drift clearly visible.

    Parameters
    ----------
    df_err : pd.DataFrame
        As returned by ``build_error_signal``.
    error_ts : pd.Series
        Mean error indexed by TimeStamp (also from ``build_error_signal``).
    save_dir : pathlib.Path
    qoi_column : str, optional
        Name of the quantity of interest, used as the y-axis label.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax = axes[0]
    ax.axhline(0, color="black", lw=0.8, ls="--")
    if "error_e_minus_std" in df_err.columns:
        ax.fill_between(df_err[utility.TIME_COLUMN_NAME],
                        df_err["error_e_minus_std"], df_err["error_e_plus_std"],
                        alpha=0.3, color="steelblue", label="Mean ± StdDev")
    if "error_p10" in df_err.columns:
        ax.fill_between(df_err[utility.TIME_COLUMN_NAME],
                        df_err["error_p10"], df_err["error_p90"],
                        alpha=0.2, color="orange", label="P10–P90 band")
    ax.plot(df_err[utility.TIME_COLUMN_NAME], df_err["mean_error"],
            color="steelblue", lw=1.2, label="Mean error (E − obs)")
    ylabel = f"Error {qoi_column}" if qoi_column else "Error"
    ax.set_ylabel(ylabel)
    ax.set_title("Error signal over time")
    ax.legend(); ax.grid(alpha=0.3)

    window = 30
    roll_mean = error_ts.rolling(window, center=True).mean()
    roll_std  = error_ts.rolling(window, center=True).std()
    ax2 = axes[1]
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.fill_between(roll_mean.index, roll_mean - roll_std, roll_mean + roll_std,
                     alpha=0.3, color="tomato")
    ax2.plot(roll_mean.index, roll_mean, color="tomato", lw=1.5,
             label=f"{window}-day rolling mean ± std")
    ax2.set_ylabel(ylabel); ax2.set_xlabel("Date")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    _save_mpl(fig, "error_time_series", save_dir)


def plot_error_distribution(df_err, error_ts, save_dir):
    """Plot the marginal distribution of the error signal in three complementary views.

    Answers: *Is the error Gaussian? Is there a seasonal bias (e.g., worse
    during spring snowmelt)?*

    Three panels:
      - Overall histogram with KDE overlay, mean marked — reveals whether the
        error distribution is symmetric, heavy-tailed, or skewed.
      - Monthly boxplot (J–D) — shows systematic seasonal over/under-prediction;
        e.g. a consistently positive bias in April/May suggests the model
        underestimates snowmelt-driven peak flows.
      - Empirical CDF — useful for reading off percentile bounds on the error.

    Parameters
    ----------
    df_err : pd.DataFrame
        As returned by ``build_error_signal``.  Must have a ``month`` column.
    error_ts : pd.Series
        Mean error indexed by TimeStamp.
    save_dir : pathlib.Path
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Histogram + KDE
    ax = axes[0]
    ax.hist(error_ts, bins=60, density=True, alpha=0.6, color="steelblue")
    t = np.linspace(error_ts.min(), error_ts.max(), 500)
    kde = cp.GaussianKDE(error_ts.values, h_mat=0.005**2)
    ax.plot(t, kde.pdf(t), color="navy", lw=2, label="KDE")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.axvline(error_ts.mean(), color="orange", ls="--", lw=1,
               label=f"Mean={error_ts.mean():.2f}")
    ax.set_xlabel("Error"); ax.set_ylabel("Density")
    ax.set_title("Overall Error Distribution"); ax.legend(); ax.grid(alpha=0.3)

    # Monthly boxplot
    monthly = [df_err[df_err["month"] == m]["mean_error"].dropna().values
               for m in range(1, 13)]
    axes[1].boxplot(monthly, labels=list("JFMAMJJASOND"), patch_artist=True)
    axes[1].axhline(0, color="red", ls="--", lw=1)
    axes[1].set_xlabel("Month"); axes[1].set_ylabel("Error")
    axes[1].set_title("Error by Month (seasonal bias)")
    axes[1].grid(alpha=0.3, axis="y")

    # Empirical CDF
    sorted_err = np.sort(error_ts)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    axes[2].plot(sorted_err, cdf, color="steelblue", lw=2)
    axes[2].axvline(0, color="red", ls="--", lw=1)
    axes[2].set_xlabel("Error"); axes[2].set_ylabel("Cumulative Probability")
    axes[2].set_title("Empirical CDF of Error"); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    _save_mpl(fig, "error_distribution", save_dir)


def plot_error_acf_pacf(error_ts, save_dir):
    """Plot the autocorrelation (ACF) and partial autocorrelation (PACF) of the error.

    Answers: *Is today's error predictable from yesterday's?
    Is there a persistent AR structure in the residuals?*

    Strong positive ACF at short lags (1–7 days) indicates the model
    systematically over- or under-predicts for multi-day stretches — a sign
    of memory in the error that an AR(p) correction or data assimilation
    could exploit.  The PACF helps determine the order p: the lag at which
    PACF drops inside the confidence band is the minimum AR order needed.
    Lags are shown up to 60 days.  The shaded band marks the 95% confidence
    interval under the white-noise null hypothesis.

    Parameters
    ----------
    error_ts : pd.Series
        Mean error indexed by TimeStamp.
    save_dir : pathlib.Path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf( error_ts, lags=60, ax=axes[0], alpha=0.05, title="ACF of Mean Error")
    plot_pacf(error_ts, lags=60, ax=axes[1], alpha=0.05, title="PACF of Mean Error",
              method="ywm")
    for ax in axes:
        ax.set_xlabel("Lag [days]"); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_mpl(fig, "error_acf_pacf", save_dir)


def run_stationarity_tests(error_ts):
    """Run ADF and KPSS stationarity tests on the error signal and print results.

    Answers: *Is the error process stationary, or does its mean/variance drift
    over the simulation period?*

    Two complementary tests are used because they have opposite null hypotheses:

    - **ADF** (Augmented Dickey-Fuller): H₀ = unit root (non-stationary).
      A small p-value (< 0.05) rejects the null → evidence of stationarity.
    - **KPSS** (Kwiatkowski–Phillips–Schmidt–Shin): H₀ = trend-stationary.
      A small p-value (< 0.05) rejects the null → evidence of non-stationarity.

    If both tests agree, the conclusion is clear.  If they disagree, the series
    may be near-stationary or have structural breaks worth investigating.
    Non-stationarity matters because it invalidates the assumption underlying
    ACF/PACF and spectral analysis that the statistical properties of the error
    are constant over time.

    Parameters
    ----------
    error_ts : pd.Series
        Mean error indexed by TimeStamp.
    """
    print("\n=== Stationarity Tests ===")
    err = error_ts.values

    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(err, autolag="AIC")
    print(f"  ADF  stat={adf_stat:.4f}  p={adf_p:.4f}  "
          f"→ {'STATIONARY' if adf_p < 0.05 else 'NON-STATIONARY'} at 5%")
    print(f"       critical values: {adf_crit}")

    kpss_stat, kpss_p, _, kpss_crit = kpss(err, regression="c", nlags="auto")
    print(f"  KPSS stat={kpss_stat:.4f}  p={kpss_p:.4f}  "
          f"→ {'NON-STATIONARY' if kpss_p < 0.05 else 'STATIONARY'} at 5%")
    print(f"       critical values: {kpss_crit}")


def plot_error_psd(error_ts, save_dir):
    """Plot the power spectral density (PSD) of the error signal.

    Answers: *Are there periodic error patterns at the annual, semi-annual,
    quarterly, or monthly scale?*

    Uses Welch's method (overlapping windowed periodograms) to estimate the PSD,
    which is more stable than a raw FFT on a noisy hydrological time series.
    Results are shown in two complementary views:

    - Left panel: PSD vs frequency [cycles/day] — standard spectral axis.
    - Right panel: PSD vs period [days] — easier to read hydrological cycles.

    Reference lines are drawn at the annual (365 d), semi-annual (182 d),
    quarterly (91 d), and monthly (30 d) periods.  A dominant peak at one of
    these periods indicates a recurring systematic bias that could point to a
    missing seasonal correction or a calibration issue specific to that season.

    Parameters
    ----------
    error_ts : pd.Series
        Mean error indexed by TimeStamp (daily values assumed, ``fs=1.0``).
    save_dir : pathlib.Path
    """
    freqs, psd = sig.welch(error_ts.values, fs=1.0, nperseg=min(len(error_ts) // 4, 256))
    periods = np.where(freqs > 0, 1 / freqs, np.inf)
    markers = [(365, "Annual"), (182, "Semi-annual"), (91, "Quarterly"), (30, "Monthly")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(freqs[1:], psd[1:], color="steelblue")
    for period, label in markers:
        if 1 / period >= freqs[1]:
            axes[0].axvline(1 / period, color="red", ls="--", lw=0.8, alpha=0.7)
            axes[0].text(1 / period, psd[1:].max() * 0.3, label,
                         rotation=90, fontsize=8, color="red")
    axes[0].set_xlabel("Frequency [cycles/day]"); axes[0].set_ylabel("PSD")
    axes[0].set_title("Power Spectral Density (Welch)"); axes[0].grid(alpha=0.3)

    valid = (periods < 400) & (periods > 2)
    axes[1].semilogy(periods[valid], psd[valid], color="tomato")
    for period, label in markers:
        axes[1].axvline(period, color="steelblue", ls="--", lw=0.8, alpha=0.7)
        axes[1].text(period, psd[valid].max() * 0.3, label,
                     rotation=90, fontsize=8, color="steelblue")
    axes[1].set_xlabel("Period [days]"); axes[1].set_ylabel("PSD")
    axes[1].set_title("PSD vs Period"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save_mpl(fig, "error_psd", save_dir)


def plot_error_vs_forcing(df_err, save_dir):
    """Cross-correlation and scatter plots between the error signal and forcing variables.

    Answers: *Is the error correlated with precipitation or temperature, and at
    what lag?*

    A positive cross-correlation at lag +k (forcing leads error by k days) means
    that high precipitation (or temperature) k days ago is associated with larger
    model error today — a signal of a routing or snowmelt timing deficiency.
    A negative lag (error leads forcing) would be unusual and may indicate data
    artefacts.

    For each forcing variable present in ``df_err`` (precipitation, temperature),
    produces a two-row panel:
      - top row: cross-correlation of error vs forcing at lags −30 … +30 days
        (positive lag = forcing leads error); 95 % confidence bounds shown
      - bottom row: scatter plot of error vs forcing with a linear regression line
        and Pearson r / p-value annotation

    Parameters
    ----------
    df_err : pd.DataFrame
        As returned by ``build_error_signal``.  Must contain the column
        ``mean_error`` and at least one of ``precipitation`` or ``temperature``.
    save_dir : pathlib.Path
    """
    forcing_cols = [c for c in ["precipitation", "temperature"] if c in df_err.columns]
    if not forcing_cols:
        return

    fig, axes = plt.subplots(2, len(forcing_cols),
                             figsize=(7 * len(forcing_cols), 9))
    if len(forcing_cols) == 1:
        axes = axes.reshape(2, 1)

    for ci, forcing in enumerate(forcing_cols):
        aligned = pd.concat(
            [df_err.set_index(utility.TIME_COLUMN_NAME)["mean_error"],
             df_err.set_index(utility.TIME_COLUMN_NAME)[forcing]], axis=1
        ).dropna()
        e, f = aligned["mean_error"], aligned[forcing]
        ci95 = 1.96 / np.sqrt(len(e))
        max_lags = 30
        lags = list(range(-max_lags, max_lags + 1))
        xcorr = [e.corr(f.shift(-lag)) for lag in lags]

        axes[0, ci].stem(lags, xcorr, markerfmt="C0o", linefmt="C0-", basefmt="k-")
        axes[0, ci].axhline( ci95, color="red", ls="--", lw=0.8, label="95% CI")
        axes[0, ci].axhline(-ci95, color="red", ls="--", lw=0.8)
        axes[0, ci].set_xlabel("Lag [days]"); axes[0, ci].set_ylabel("Cross-correlation")
        axes[0, ci].set_title(f"Error × {forcing}"); axes[0, ci].legend()
        axes[0, ci].grid(alpha=0.3)

        axes[1, ci].scatter(f, e, alpha=0.3, s=8, color="steelblue")
        m, b, r, p, _ = stats.linregress(f, e)
        x_line = np.linspace(f.min(), f.max(), 100)
        axes[1, ci].plot(x_line, m * x_line + b, color="red", lw=2,
                         label=f"r={r:.3f}  p={p:.3f}")
        axes[1, ci].axhline(0, color="black", lw=0.5, ls="--")
        axes[1, ci].set_xlabel(forcing); axes[1, ci].set_ylabel("Error")
        axes[1, ci].set_title(f"Error vs {forcing}"); axes[1, ci].legend()
        axes[1, ci].grid(alpha=0.3)

    plt.tight_layout()
    _save_mpl(fig, "error_vs_forcing", save_dir)


def plot_bias_vs_spread(df_err, save_dir):
    """Compare the bias (mean error) against the UQ spread (StdDev).

    Answers: *Is the model's parameter uncertainty large enough to cover the
    systematic bias, or does structural model error dominate?*

    Two panels:
      - Top: time series of |mean error| (bias) and StdDev in the same units.
        When the red StdDev line consistently lies below the blue bias line,
        parameter uncertainty cannot explain the mismatch and a structural
        deficiency is the more likely cause.
      - Bottom: the ratio |bias| / StdDev over time.  A ratio > 1 means structural
        error dominates for that timestep; < 1 means the UQ spread is wide enough
        to cover the observations.  The fraction of timesteps where ratio > 1
        is printed to stdout as a summary statistic.

    The function silently returns without plotting if ``StdDev`` is not present in
    ``df_err`` (e.g., when only the mean prediction E was available).

    Parameters
    ----------
    df_err : pd.DataFrame
        As returned by ``build_error_signal``.  Must contain ``TimeStamp``,
        ``mean_error``, and ``StdDev``.
    save_dir : pathlib.Path
    """
    if "StdDev" not in df_err.columns:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(df_err[utility.TIME_COLUMN_NAME], df_err["mean_error"].abs(),
                 color="steelblue", lw=1, label="|Mean error| (bias)")
    axes[0].plot(df_err[utility.TIME_COLUMN_NAME], df_err["StdDev"],
                 color="tomato", lw=1, label="UQ spread / StdDev (parameter uncertainty)")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Bias vs Parameter Uncertainty over Time")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    ratio = df_err["mean_error"].abs() / df_err["StdDev"].replace(0, np.nan)
    axes[1].plot(df_err[utility.TIME_COLUMN_NAME], ratio, color="purple", lw=1, alpha=0.8)
    axes[1].axhline(1.0, color="red", ls="--", lw=1,
                    label="|bias| = spread  (ratio = 1)")
    axes[1].set_ylabel("|Mean error| / StdDev"); axes[1].set_xlabel("Date")
    axes[1].set_title("Ratio > 1: structural error dominates; < 1: spread covers observations")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save_mpl(fig, "error_vs_spread", save_dir)

    pct = (ratio > 1).mean()
    print(f"\n  Timesteps where |bias| > UQ spread: {(ratio > 1).sum()} / {len(ratio)}  ({pct:.1%})")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HBV-SASK UQ analysis")
    parser.add_argument("--working_dir", type=pathlib.Path, default=WORKING_DIR)
    parser.add_argument("--save_dir",    type=pathlib.Path, default=SAVE_DIR)
    # Only this is HBV model specific
    parser.add_argument("--hbv_data",    type=pathlib.Path, default=HBV_MODEL_DATA_PATH)
    parser.add_argument("--qoi",         type=str,          default=QOI_COLUMN)
    parser.add_argument("--read_sims",   action="store_true",
                        help="Load full simulation results (can be very large)")
    parser.add_argument("--plot_parameter_analysis",   action="store_true",
                        help="Plot parameter analysis")
    parser.add_argument("--plot_time_series",          action="store_true",
                        help="Plot UQ time series (mean prediction, uncertainty bands, observations)")
    return parser.parse_args()


def main():
    args = parse_args()
    working_dir = args.working_dir
    save_dir    = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Working dir : {working_dir}")
    print(f"Save dir    : {save_dir}")

    # 1. Load run artefacts
    uqsim_args_dict, configuration, file_paths = load_run_artefacts(working_dir)
    basin = configuration["model_settings"]["basin"]

    # 2. Parameter / GoF data
    (_, _, df_gof,
     params_list, gof_list, df_gof_conditioned) = load_parameter_data(
        file_paths, working_dir, configuration)

    # 3. Parameter analysis plots
    if args.plot_parameter_analysis:
        plot_parameter_analysis(df_gof, params_list, gof_list, df_gof_conditioned, save_dir)

    # 4. Build statistics object and merged dataframe
    # Optionally pre-load full simulation results (large file — skipped by default)
    df_simulation_result = None
    if args.read_sims and file_paths.get("df_simulations_file", pathlib.Path("")).is_file():
        print("Loading full simulation results (this may take a while)...")
        df_simulation_result = pd.read_pickle(
            file_paths["df_simulations_file"], compression="gzip")

    stat_obj, df_stats = build_statistics(
        configuration, uqsim_args_dict, working_dir,
        args.hbv_data, basin, df_simulation_result)

    # 5. UQ time-series plots
    if args.plot_time_series:
        plot_time_series(stat_obj, df_stats, save_dir)

    # 6. GoF metrics and P/R factors
    compute_gof_and_pr(stat_obj, df_stats, args.qoi)

    # 7. Error signal construction
    df_err, error_ts = build_error_signal(df_stats, args.qoi)
    print_error_summary(error_ts, args.qoi)

    # 8. Error time-series analysis
    print("\nRunning error time-series analysis...")
    plot_error_time_series(df_err, error_ts, save_dir, args.qoi)
    plot_error_distribution(df_err, error_ts, save_dir)
    plot_error_acf_pacf(error_ts, save_dir)
    run_stationarity_tests(error_ts)
    plot_error_psd(error_ts, save_dir)
    plot_error_vs_forcing(df_err, save_dir)
    plot_bias_vs_spread(df_err, save_dir)

    print(f"\nDone. All outputs saved to {save_dir}")


if __name__ == "__main__":
    main()