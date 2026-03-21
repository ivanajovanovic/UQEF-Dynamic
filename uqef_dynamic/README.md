# UQEF-Dynamic Framework

## Overview

UQEF-Dynamic is a comprehensive framework for Uncertainty Quantification (UQ) and Global Sensitivity Analysis (SA) for dynamic models, with a particular focus on hydrological models and other time-dependent systems. The framework extends the capabilities of the UQEF (Uncertainty Quantification and Ensemble Framework) to handle time-dependent processes.


## Directory Structure

The `uqef_dynamic` directory is organized into the following main components:

```
uqef_dynamic/
├── models/
├── scientific_pipelines/
└── utils/
```


### Models

The `models/` directory contains implementations of various models that can be used with the framework:

```
models/
├── hbv/                  # HBV hydrological model
├── hbv_sask/             # Saskatchewan implementation of HBV model
├── ishigami/             # Ishigami test function model
├── larsim/               # LARSIM hydrological model
├── linearDampedOscillator/ # Linear damped oscillator model
├── productFunction/      # Product function model
├── pybamm/               # PyBaMM battery model
├── simpleOscilator/      # Simple oscillator model
├── sparsespace/          # Sparse space model
└── time_dependent_baseclass/ # Base class for time-dependent models
```

Each model directory typically contains:
- Model implementation files
- Model-specific utility functions
- UQ-specific model adaptations
- Statistics classes for model outputs

The `time_dependent_baseclass/` provides a common interface for all time-dependent models, ensuring consistent handling of time series data.

### Scientific Pipelines

The `scientific_pipelines/` directory contains workflows for different types of scientific simulations:

```
scientific_pipelines/
├── compare_surrogate_model_pipeline.py
├── comparing_model_and_surrogate_mpi_threading.py
├── comparing_model_and_surrogate_mpi.py
├── comparing_model_and_surrogate_parallel_over_nodes_mpi.py
├── comparing_model_and_surrogate.py
├── KL_and_PCE_simple_model.py
├── KL_and_PCE_time_dependent_processes_pipeline.py
├── list_of_simulation_runs_var1_2.py
├── list_of_simulation_runs.py
├── particle_filtering_pipeline.py
├── uq_simulation_uqsim_debug_cluster.py
├── uq_simulation_uqsim_debug_mls_larsim.py
├── uq_simulation_uqsim_ensemble.py
├── uq_simulation_uqsim_hbv.py
├── uq_simulation_uqsim_mls.py
├── uq_simulation_uqsim.py
├── uq_with_sparsespace.py
├── uqef_dynamic_postprocessing.py
└── uqPostprocessing_pipeline_hbv_sask.py
```

Key pipelines include:
- **uq_simulation_uqsim.py**: Main simulation pipeline for UQ analysis
- **KL_and_PCE_time_dependent_processes_pipeline.py**: Pipeline for Karhunen-Loève expansion and Polynomial Chaos Expansion for time-dependent processes
- **comparing_model_and_surrogate_*.py**: Pipelines for comparing original models with surrogate models
- **uqef_dynamic_postprocessing.py**: Post-processing pipeline for trained surrogate models — see detailed description below
- **plot_sobol_indices.py**: Standalone CLI script to read saved simulation results and plot Sobol sensitivity indices — see detailed description below
- **extract_timing_info.py**: Standalone CLI script to read all timing information from a simulation output directory — see detailed description below

---

## `uqef_dynamic_postprocessing.py` — Surrogate Model Post-Processing Pipeline

### Purpose

This script is the **main post-processing entry point** to use after a UQ training run has finished and a surrogate model (gPCE or KL+PCE) has been saved to disk. It reads the saved surrogate, re-evaluates it on a large independent sample set, computes statistics and sensitivity indices, and produces comparison plots.

It does **not** re-run the full training simulation — it works entirely from the files written by `uq_simulation_uqsim.py` (or equivalent) to `workingDir`.

### Main function signature

```python
main(
    mpi, rank,
    workingDir=None,            # path to the directory written by the training run
    inputModelDir=None,         # path to the original model input files (needed if reevaluate_original_model=True)
    directory_for_saving_plots=None,  # where to write output plots/files (defaults to workingDir)
    surrogate_type="pce",       # "pce" | "kl+pce"
    surrogate_type="pce",
    printing=False,
    plotting=True,
    model=None,
    **kwargs,                   # all switches listed below
)
```

### Key `kwargs` switches

| Parameter | Type | Default | Description |
|---|---|---|---|
| `surrogate_type` | `str` | `"pce"` | Which surrogate to use: `"pce"` for standard gPCE, `"kl+pce"` for KL expansion + PCE |
| `reevaluate_surrogate` | `bool` | `False` | Re-evaluate the surrogate on a fresh random sample set |
| `reevaluate_original_model` | `bool` | `False` | Also run the **original model** on the same sample set (expensive) |
| `recompute_statistics` | `bool` | `False` | Recompute mean, variance, Sobol indices from the surrogate re-evaluations |
| `recompute_generalized_sobol_indices` | `bool` | `False` | Recompute time-aggregated generalised Sobol indices |
| `compute_generalized_sobol_indices_from_kl_expansion` | `bool` | `True` | Derive generalised Sobol indices analytically from KL+PCE coefficients (fast, no re-evaluation needed) |
| `compute_generalized_sobol_indices_over_time` | `bool` | `False` | Compute rolling-window generalised Sobol indices over time |
| `look_back_window_size` | `int/list` | `[30,60,90,365]` | Window size(s) in timesteps for rolling generalised Sobol indices |
| `analyse_pce_surrogate` | `bool` | `False` | Analyse PCE coefficient magnitudes per QoI (standard PCE only) |
| `compute_Sobol_m` | `bool` | `True` | Compute first-order Sobol indices |
| `compute_Sobol_t` | `bool` | `True` | Compute total-order Sobol indices |
| `dict_stat_to_compute` | `dict` | — | Fine-grained control over which statistics to compute (Var, StdDev, P10, P90, Skew, Kurt, …) |
| `plotting` | `bool` | `True` | Generate comparison plots |
| `replot_statistics_from_statistics_object` | `bool` | `False` | Re-plot from an already-computed statistics object rather than raw re-evaluations |
| `add_measured_data` | `bool` | `False` | Overlay measured/observed data on plots |
| `add_forcing_data` | `bool` | `False` | Overlay forcing data on plots |
| `set_lower_predictions_to_zero` | `bool` | `False` | Clip negative surrogate predictions to zero (useful for non-negative QoIs such as discharge) |
| `read_saved_simulations` | `bool` | `False` | Read previously saved simulation results instead of re-running |
| `set_up_statistics_from_scratch` | `bool` | `False` | Rebuild the statistics object from raw samples rather than reading the saved statistics dictionary |

### Typical workflow

```
Training run                     Post-processing
─────────────────────────────    ──────────────────────────────────────────────────
uq_simulation_uqsim.py           uqef_dynamic_postprocessing.py
  → saves surrogate to           → reads surrogate from workingDir
    workingDir/                  → draws new samples from the joint distribution
    (gpce_coeffs, KL comps,      → evaluates surrogate (parallel, MPI)
     df_simulations, …)          → optionally evaluates original model (expensive)
                                 → computes mean, Var, P10/P90, Sobol indices
                                 → saves plots + dict_with_time_info.txt
```

### Output files written to `directory_for_saving_plots/`

| File | Description |
|---|---|
| `dict_with_time_info.txt` | Wall-time breakdown for each stage (reading, surrogate eval, model eval, statistics, total) |
| `*.html` / `*.png` | Plotly/matplotlib comparison plots of surrogate vs original model |
| Recomputed statistics pickles | If `recompute_statistics=True`, updated statistics files |

### Example usage (bottom of the script)

```python
workingDir = pathlib.Path('/path/to/training_run_output')
inputModelDir = pathlib.Path('/path/to/model/input/files')   # only needed if reevaluate_original_model=True
directory_for_saving_plots = workingDir / 'surrogate_analysis'

main(
    mpi, rank,
    workingDir=workingDir,
    inputModelDir=inputModelDir,
    directory_for_saving_plots=directory_for_saving_plots,
    surrogate_type='kl+pce',
    reevaluate_surrogate=True,
    reevaluate_original_model=False,   # set True only if you want the expensive model comparison
    recompute_generalized_sobol_indices=True,
    compute_generalized_sobol_indices_from_kl_expansion=True,
    compute_Sobol_m=True,
    compute_Sobol_t=True,
    plotting=True,
)
```

### Notes

- The script is MPI-aware: surrogate and model re-evaluations are distributed across MPI ranks. Run with `mpiexec -n <N> python uqef_dynamic_postprocessing.py`.
- For **KL+PCE surrogates** (`surrogate_type='kl+pce'`), set `analyse_pce_surrogate=False` (it is only implemented for standard PCE).
- `inputModelDir` must point to the conda environment that contains the model's input data files (e.g. PyBaMM drive cycle files). Make sure it matches the Python version of the environment you are running in.
- All timing metrics are collected in `dict_with_time_info` and written to `dict_with_time_info.txt`, including `mean_surrogate_eval_time_per_sample` and `total_runtime`.

---

## `plot_sobol_indices.py` — Sobol Sensitivity Index Plotter

### Purpose

Standalone CLI script that reads the saved statistics dictionary from a UQEF-Dynamic simulation output directory and produces publication-ready plots of Sobol sensitivity indices. No re-simulation or postprocessing run is required — it works directly from the `statistics_dictionary_qoi.pkl` written by the training pipeline.

Two plot types are produced per QoI:
- **Time-wise Sobol indices** — first-order (`Sobol_m`) and/or total-order (`Sobol_t`) indices as line plots over the simulation time axis
- **Generalized Sobol indices** — time-aggregated scalar indices shown as a grouped bar chart (total-order vs first-order)

### Usage

```bash
python uqef_dynamic/scientific_pipelines/plot_sobol_indices.py <workingDir> [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `workingDir` | *(required)* | Path to the simulation output directory |
| `--qoi` | all found | Restrict plotting to specific QoI name(s) |
| `--top N` | all | Show only the top-N most influential parameters (by generalized index) |
| `--clip-negative` | off | Set negative Sobol values to zero before plotting |
| `--no-timewise` | off | Skip time-wise line plots |
| `--no-generalized` | off | Skip generalized Sobol bar charts |
| `--save-dir DIR` | same as `workingDir/figures/` | Directory to save output PNG files |
| `--verbose` | off | Print per-parameter tables to stdout |

### Output

Figures are written to `<workingDir>/figures/` (or `--save-dir`):

| File | Description |
|---|---|
| `sobol_timewise_<qoi>.png` | Line plot of first-order (and total-order if available) Sobol indices over time |
| `sobol_generalized_<qoi>.png` | Bar chart of generalized total-order and first-order Sobol indices |

A summary table of parameter rankings is also printed to stdout.

### Example

```bash
# Basic run — saves figures to debug_output/.../figures/
python uqef_dynamic/scientific_pipelines/plot_sobol_indices.py \
    debug_output/battery_mc_10000_kl_trunc_local_debug

# Show only top 10 parameters, clip negatives, verbose output
python uqef_dynamic/scientific_pipelines/plot_sobol_indices.py \
    debug_output/battery_mc_10000_kl_trunc_local_debug \
    --top 10 --clip-negative --verbose

# Custom save directory
python uqef_dynamic/scientific_pipelines/plot_sobol_indices.py \
    debug_output/battery_mc_10000_kl_trunc_local_debug \
    --save-dir /path/to/figures/
```

---

## `extract_timing_info.py` — Timing Information Extractor

### Purpose

Standalone CLI script that reads all timing data saved by a UQEF-Dynamic simulation run (and optionally its postprocessing run) and prints a structured report. The report is simultaneously saved to a `.txt` file.

Three layers of timing data are extracted:

| Layer | File | Written by |
|---|---|---|
| 1 | `time_info.txt` | Every pipeline script (`uq_simulation_uqsim.py`, etc.) |
| 2 | `uqsim_args.pkl` | Every pipeline script (run configuration context) |
| 3 | `dict_with_time_info.txt` | `uqef_dynamic_postprocessing.py` |

### Timing keys extracted

**Layer 1 — `time_info.txt`:**

| Key | Description |
|---|---|
| `number_full_model_runs` | Total number of samples evaluated |
| `time_model_simulations` | Wall time for `uqsim.simulate()` (MPI propagation) |
| `time_computing_statistics` | Wall time for statistics computation |
| `total_time` | End-to-end wall time of the pipeline script |
| `mean_model_eval_time` | Mean pure model call time per sample (inner timer) |
| `total_model_eval_time` | Sum of all pure model call times |

**Layer 3 — `dict_with_time_info.txt`:**

| Key | Description |
|---|---|
| `time_reading_all_saved_data` | Time to load pickled results from disk |
| `time_paralle_statistics_recomputation` | Time to recompute statistics |
| `time_parallel_original_model_reevaluations` | Wall time for original model re-evaluations |
| `mean_original_model_eval_time_per_sample` | Mean original model time per sample |
| `time_parallel_pce_surrogate_reevaluations` | Wall time for PCE surrogate re-evaluations |
| `time_kl_surrogate_reevaluations` | Wall time for KL+PCE surrogate re-evaluations |
| `mean_surrogate_eval_time_per_sample` | Mean surrogate evaluation time per sample |
| `time_generalized_si_recomputation` | Time for generalised Sobol index recomputation |
| `total_runtime` | Total postprocessing wall time |

A **derived metric** is also computed: `mean outer time / sample = time_model_simulations / N`, which includes MPI communication and I/O overhead on top of the pure model evaluation.

### Usage

```bash
python uqef_dynamic/scientific_pipelines/extract_timing_info.py <workingDir> [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `workingDir` | *(required)* | Path to the simulation output directory |
| `--verbose` / `-v` | off | Also print keys that were not recorded in this run |
| `--output` / `-o` | `<workingDir>/timing_report.txt` | Path for the saved report file |

### Output

The full printed report is saved to `<workingDir>/timing_report.txt` by default (or to `--output`).

### Example

```bash
# Default: report saved to <workingDir>/timing_report.txt
python uqef_dynamic/scientific_pipelines/extract_timing_info.py \
    debug_output/battery_mc_10000_kl_trunc_local_debug

# Verbose mode + custom output file
python uqef_dynamic/scientific_pipelines/extract_timing_info.py \
    debug_output/battery_mc_10000_kl_trunc_local_debug \
    --verbose --output my_timing_report.txt
```

---

### Utilities

The `utils/` directory contains utility functions and tools that support the framework:

```
utils/
├── colors.py
├── create_stat_object.py
├── morris_sensitivity_analysis.py
├── objectivefunctions.py
├── parallel_statistics.py
├── sens_indices_sampling_based_utils.py
├── sparsespace_utils.py
├── transport_map.py
├── uqef_dynamic_utils.py
├── utility.py
└── uqPostprocessing.py
```

Key utility files include:
- **utility.py**: General utility functions for file handling, data processing, and statistical calculations
- **uqef_dynamic_utils.py**: A set of utility functions tailored to work specificly with the data structures in the UQEF-Dynamic for, e.g., model configurations, reading output files, data post-processing, and statistical calculations, etc.
- **parallel_statistics.py**: Functions for (parallel) computation of statistics
- **uqPostprocessing.py**: Post-processing utilities for UQ results
- **objectivefunctions.py**: Implementation of various goodness-of-fit metrics and objective functions



