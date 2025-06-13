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
└── uqPostprocessing_pipeline_hbv_sask.py
```

Key pipelines include:
- **uq_simulation_uqsim.py**: Main simulation pipeline for UQ analysis
- **KL_and_PCE_time_dependent_processes_pipeline.py**: Pipeline for Karhunen-Loève expansion and Polynomial Chaos Expansion for time-dependent processes
- **comparing_model_and_surrogate_*.py**: Pipelines for comparing original models with surrogate models

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

## Framework Capabilities

The UQEF-Dynamic framework provides the following key capabilities:

1. **Model Integration**: Support for various models including hydrological models (HBV, LARSIM), test functions (Ishigami), and physical models (oscillators, batteries)

2. **Uncertainty Quantification Methods**:
   - Ensamble Analysis
   - Monte Carlo (MC) sampling
   - Stochastic Collocation (SC)
   - Polynomial Chaos Expansion (PCE)
   - Karhunen-Loève (KL) expansion for time-dependent processes

3. **Sensitivity Analysis**:
   - Sobol indices (main, total, and second-order)
   - Generalized Sobol indices for time-dependent processes
   - Active subspaces
   - Gradient analysis

4. **Statistical Analysis**:
   - Computation of statistical moments (mean, variance, skewness, kurtosis)
   - Percentile calculations
   - Time-dependent statistics
   - Goodness-of-fit metrics

5. **Visualization and Post-processing**:
   - Time series visualization
   - Sensitivity indices visualization
   - Statistical moments visualization
   - Surrogate model validation

6. **Parallel Computing**:
   - MPI-based parallelization
   - Thread-based parallelization
   - Hybrid parallelization strategies

## Configuration System

The framework uses a JSON-based configuration system to define:
- Model parameters and their distributions
- Time settings for simulations
- UQ method settings
- Output and post-processing settings

Example configuration files can be found in the `data/configurations/` directory.

## Usage

The framework is typically used through the scientific pipelines, with the main entry point being `uq_simulation_uqsim.py`. The workflow generally involves:

1. Defining a configuration file
2. Selecting a model and UQ method
3. Running the simulation
4. Post-processing and analyzing the results

## Dependencies

The framework depends on several libraries:
- Chaospy for uncertainty quantification
- UQEF library
- NumPy, SciPy, Pandas for numerical computations and data handling
- Plotly and Matplotlib for visualization
- MPI libraries for parallel computing

## Authors

- Ivana Jovanovic Buha
- Florian Kuenzner
