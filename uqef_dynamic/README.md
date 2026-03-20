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



