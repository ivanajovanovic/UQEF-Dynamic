# UQEF-Dynamic Framework

Software tool for efficient forward uncertainty quantification and global sensitivity analysis of different models that produce time-varying output-of-interest (e.g., environmental models, dynamical models)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
  - [Basic Installation](#basic-installation)
  - [Installation with Optional Dependencies](#installation-with-optional-dependencies)
  - [Development Installation - install from source](#development-installation---install-from-source)
- [Dependencies and Environment Setup](#dependencies-and-environment-setup)
   - [Core Dependencies](#core-dependencies)
   - [Dependency Files](#dependency-files)
   - [Other Dependencies](#other-dependencies)
   - [Setting Up the Environment](#setting-up-the-environment)
- [Framework Capabilities](#framework-capabilities)
- [Usage - How to Run the Code/Simulation](#usage---how-to-run-the-codesimulation)
   - [Basic Usage - Command-Line Usage](#basic-usage---command-line-usage)
- [Input Arguments](#input-arguments)
- [Configuration Management System](#configuration-management-system)
- [Configuration Files](#configuration-files)
- [Available Models](#available-models)
- [Custom Model and Statistics](#custom-model-and-statistics)
- [Parallel Computing](#parallel-computing)
- [Running the Simulation on HPC](#running-the-simulation-on-hpc)
- [Paths Definitions](#paths-definitions)
- [Output Files](#output-files)
- [Author](#author)
- [Repository](#repository)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Version History](#version-history)

## Overview

UQEF-Dynamic is a comprehensive framework for Uncertainty Quantification (UQ) and Global Sensitivity Analysis (SA) for dynamic models, with a particular focus on hydrological models and other time-dependent systems. The framework extends the capabilities of the UQEF (Uncertainty Quantification Execution Framework) to handle time-dependent processes.

<!-- This code is licensed under the GNU Lesser General Public License version 3 or
later, see `COPYING` and `COPYING.LESSER`. -->

## Installation

UQEF-Dynamic can be installed in several ways depending on your needs:

### Basic Installation

For the core functionality without optional model dependencies:

```bash
pip install uqef_dynamic
```

### Installation with Optional Dependencies

#### Battery Model Support

To include support for battery models (requires PyBaMM):

```bash
pip install uqef_dynamic[battery]
```

#### LARSIM Model Support
To include support for the LARSIM hydrological model (requires Larsim_Utility_Set library):

```bash
cd Larsim_Utility_Set/
git checkout master
git pull
$(which pip) install -e .
```

#### All Optional Dependencies

To install with all optional dependencies:

```bash
pip install uqef_dynamic[all]
```

### Development Installation - install from source

For development or if you want to modify the code:

```bash
# Clone the repository
git clone https://github.com/ivanajovanovic/UQEF-Dynamic.git
cd UQEF-Dynamic

# Basic development installation
pip install -e .

# Development installation with battery support
pip install -e .[battery]

# Development installation with all optional dependencies
pip install -e .[all]
```

### Installation Notes

- **Battery Models**: The `pybamm` package is only required if you plan to use battery models. It's included as an optional dependency to avoid forcing installation for users who don't need it.
- **Python Version**: Compatible with Python 3.11
- **Dependencies**: Core dependencies will be automatically installed with any of the above methods

## Dependencies and Environment Setup

### Core Dependencies

- **Python**: Compatible with Python 3.11 (recommended)
- **UQEF**: Uncertainty Quantification Execution Framework (core library)
- **Chaospy**: Python toolbox for probabilistic modelling and forward uncertainty propagation (i.e., via the MC sampling methods or Polynomial chaos expansion-based methods)
- **NumPy/SciPy/Pandas**: For numerical computations and data handling
- **MPI Libraries**: For parallel computing (mpi4py)
- **Visualization**: Matplotlib, Plotly, Seaborn for visualization and plotting

### Dependency Files

The project includes several requirements files:

- `requirements.txt`: Basic dependencies list
- `requirements_py311_version.txt`: Dependencies with fixed versions for Python 3.11

All dependencies will be automatically installed when using `pip install`.

### Other Dependencies

- **LARSIM Model**: Require Larsim_Utility_Set library
- **Battery Models**: Require PyBaMM (Python Battery Mathematical Modelling)
- **Sparse Space Scientific Pipelines**: Require sparseSpACE toolbox

### Setting Up the Environment

The recommended way to set up the environment is using conda:

```bash
# Create a new conda environment with Python 3.11
conda create -n uqef_env python=3.11
conda install -n uqef_env --file requirements.txt --update-deps
conda activate uqef_env

## In case Chaospy was not installed properly via conda, install it via pip from the source code, 
## however, this is not recommended and you make sure that the dependencies are not broken
$(which pip) install --no-deps chaospy

# Install UQEF (assuming the repository is cloned)
# Note - this will change soon, since UQEF is now available on PyPi
cd UQEF/
git checkout parallel_statistics
$(which pip) install -e .
cd ../

# For Larsim model (if needed)
cd Larsim_Utility_Set/
git checkout master
git pull
$(which pip) install -e .
cd ../

# For sparseSpACE toolbox (if needed)
cd sparseSpACE/
$(which pip) install -e .
cd ../
```

For convenience, you can use the provided setup script:

```bash
bash set_up_conda_env.sh
```

## Framework Capabilities

The UQEF-Dynamic framework provides the following key capabilities:

1. **Model Integration**: Support for various models including hydrological models (HBV, LARSIM), test functions (Ishigami), and physical models (oscillators, batteries)

2. **Uncertainty Quantification Methods**:
   - Ensamble Analysis
   - Monte Carlo (MC) sampling
   - Polynomial Chaos Expansion (PCE) with Stochastic Collocation (SC) or Pseudo-Spectral Projection (PSP)
   - Karhunen-Loève (KL) expansion for time-dependent processes

3. **Sensitivity Analysis**:
   - Sobol indices (main, total, and second-order) computed via different methods
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
   - Thread-based parallelization or hybrid parallelization of some pipeline

## Usage - How to Run the Code/Simulation

The UQEF-Dynamic framework is primarily used through the scientific pipelines, with the main entry point being `uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py`. The `uqef_dynamic/scientific_pipelines/` subdirectory contains workflows for other different types of scientific simulations. The workflow generally involves:

1. Defining a configuration file
2. Selecting a model and UQ method
3. Running the simulation
4. Post-processing and analyzing the results


### Basic Usage - Command-Line Usage

Run the pseudo-spectral approach simulation:
```bash
python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --config_file data/configurations/configuration_hbv_5D.json \
    --model "hbvsask" \
    --inputModelDir /path/to/model/data \
    --outputResultDir /path/to/output/directory \
    --sourceDir /path/to/source/code \
    --uq_method "sc" \
    --sc_q_order 7 \
    --sc_p_order 3
```

Run the stochastic collocation simulation with MPI:
```bash
mpiexec -n 8 python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --config_file data/configurations/configuration_hbv_5D.json \
    --model "hbvsask" \
    --inputModelDir /path/to/model/data \
    --outputResultDir /path/to/output/directory \
    --sourceDir /path/to/source/code \
    --uq_method "sc" \
    --regression \
    --sc_q_order 7 \
    --sc_p_order 3 \
    --mpi
```

## Input Arguments

The framework supports numerous command-line arguments to control the simulation. Most of these arguments are inherited by the UQEF tool. Here are the most important ones:

#### General Arguments
- `--config_file`: Path to the configuration file; in case this argument is provided, it can override some of the other command-line arguments

#### Model settings
- `--model`: Name of the model (e.g., 'hbvsask', 'larsim', 'ishigami')
- `--model_variant`: Variant of the chosen model

#### UQ method and uncertain parameter settings
- `--uncertain`: Uncertain setting: can be evaluated to choose different probability distributions and their parameter values
- `--uq_method`: Define the UQ method: `sc`, `mc`, `saltelli`, or `ensemble`

#### Monte Carlo (`--uq_method mc`)
- `--mc_numevaluations`: Number of Monte Carlo samples
- `--sampling_rule`: Sampling strategy (`random`, `sobol`, `latin_hypercube`, `halton`, `hammersley`)
- `--regression`: Enable regression-based surrogate modeling (i.e., PCE-based); if this argument is enabled, the MC samples will be used to build a PCE surrogate model and other PCE-based input arguments will be used (see below)

#### Stochastic Collocation; Actually this is consider to be PCE-based simulation which can both run Stochastic Collocation or Pseudo-Spectral Projection simulation depending on the regression argument (`--uq_method sc`)
- `--sc_q_order`: Quadrature order (collocation points per dimension)
- `--sc_p_order`: Polynomial order (PCE terms)
- `--regression`: Enable stochastic collocation method
- `--sc_quadrature_rule`: Quadrature rule (default: 'G' for Gaussian); In case regression is enabled, this argument is not used
- `--sc_sparse_quadrature`: Enable sparse grid quadrature; In case regression is enabled, this argument is not used
- `--sc_poly_rule`: Polynomial rule for Stochastic Collocation
- `--sc_poly_normed`: Use normed polynomials
- `--sc_sparse_level`: Sparse grid level (if sparse grid quadrature is enabled)
- `--cross_truncation`: Cross-truncation parameter for polynomial basis

#### Saltelli (`--uq_method saltelli`)
- `--mc_numevaluations`: Number of base samples
- Take a look at other MC-related arguments for sampling rule, regression, etc.

#### Ensemble (`--uq_method ensemble`)
- `--read_nodes_from_file`: Read parameter values from file
- `--parameters_file`: File containing parameter sets

#### Conditional Analysis Options
- `--allow_conditioning_results_based_on_metric`: Enable conditioning of results based on the specified performance metric and value
- `--condition_results_based_on_metric`: Condition results based on a specific performance metric (e.g., 'NSE', 'KGE')
- `--condition_results_based_on_metric_value`: Value of the performance metric for conditioning (e.g., `0.7` for NSE > 0.7)      
- `--condition_results_based_on_metric_sign`: Sign for conditioning (`greater`, `less`, `equal`,  `greater_or_equal`, `less_or_equal`)

#### Statistics Arguments and Post-processing Options
- `--compute_Sobol_t`: Compute total Sobol indices
- `--compute_Sobol_m`: Compute main effect indices
- `--compute_Sobol_m2`: Compute second-order indices
- `--compute_sobol_indices_with_samples`: Compute Sobol indices using samples rank-based approach; only relevant for MC-based UQ methods
- `--disable_statistics`:  Disable all statistical calculations including plots (useful when restoring a saved uqsim object from file)
- `--disable_recalc_statistics`: Disable the recalculation of statistics (useful when restoring a saved uqsim object from file)
- `--disable_calc_statistics`: Disable calculation of statistics; still saves all simulation data (i.e., in one big DataFrame)
- `--parallel_statistics`: Enable parallel statistics computation over different time steps (i.e., elements of the vector-valued QoI)
- `--instantly_save_results_for_each_time_step`: Save results for each time step instantly; relevant when working with large data that cannot fit into memory
- `--compute_generalized_sobol_indices`: Compute generalized Sobol indices for time-dependent processes
- `--compute_generalized_sobol_indices_over_time`: Compute generalized Sobol indices over time for time-dependent processes; for now, this is only available for PCE-based UQ methods (not KL expansion-based methods)
- `--compute_kl_expansion_of_qoi`: Compute KL expansion of the Quantity of
   Interest (QoI)
- `--kl_expansion_order`: Order of the KL expansion
- `--compute_timewise_gpce_next_to_kl_expansion`: Compute time-wise gPCE surrogate models next to the KL expansion-based surrogate

#### Controlling saved data, mainly from statistics computation
- `--save_all_simulations`: Save complete simulation data
- `--store_qoi_data_in_stat_dict`: Store quantity of interest data
- `--store_gpce_surrogate_in_stat_dict`: Store PCE surrogate model
- `--instantly_save_results_for_each_time_step`: Save results incrementally (has to be done in custom models)
- `--save_gpce_surrogate`: Save gPCE surrogate models for each time step (if time-wise gPCE surrogates are computed)

#### Model and result directories
- `--inputModelDir`: Folder for the input files of the model
- `--outputModelDir`: Folder for the output files of the model
- `--outputResultDir`: Folder for the statistics results (plots, tables (csv), ...)
- `--sourceDir`: Source directory

#### Parallelization Arguments
- `--parallel`: Enable shared-memory parallelization with threading
- `--num_cores`: Number of cores per node to use (default: all available)
- `--mpi`: Enable MPI parallelization
- `--mpi_method`: Choose MPI solver (`MpiPoolSolver` or `MpiSolver`)
- `--mpi_combined_parallel`: Enable hybrid MPI + multiprocessing (data distribution to the nodes via MPI and parallelisation with a node via threading)
- `--chunksize`: Number of runs that are chunked into a group
- `--mpi_chunksize`: Number of runs that are sent as a package via MPI
- `--parallel_statistics`: Enable parallel statistics computation

#### Runtime Analysis and Optimization (from UQEF)
- `--analyse_runtime`: Enable runtime analysis
- `--opt_runtime`: Enable runtime optimization with load balancing
- `--opt_runtime_gpce_Dir`: Define the folder for the runtime data
- `--opt_algorithm`: Scheduling algorithm (FCFS, LPT, SPT, or MULTIFIT)
- `--opt_strategy`: Optimization strategy (FIXED_ALTERNATE, FIXED_LINEAR, or DYNAMIC)

#### UQsim State Management: Save/Restore (from UQEF)
- `--uqsim_store_to_file`: Save UQsim state for later restoration
- `--uqsim_restore_from_file`: Restore UQsim from saved state
- `--uqsim_file`: Filename for state storage (default: uqsim.saved)

## Configuration Management System

UQEF-Dynamic includes a comprehensive configuration management system that provides a structured approach to managing simulation parameters. The system supports:

- **Extended Analysis Features**: KL expansion, generalized Sobol indices, covariance matrix analysis
- **Conditional Analysis**: Results filtering based on performance metrics
- **Custom Statistics and Plotting**: Configurable computation and visualization options
- **Batch Script Compatibility**: Configuration overrides for existing workflows
- **Extended Command-Line Arguments**: Access advanced features through bash scripts

#### Quick Configuration Example

```python
from uqef_dynamic.config import ConfigurationFactory

# Create configuration with advanced features
config = ConfigurationFactory.create_configuration(
    model_type="hbvsask",
    uq_method="sc",
    sc_p_order=3,
    sc_q_order=7,
    # Advanced analysis features
    compute_kl_expansion_of_qoi=True,
    kl_expansion_order=12,
    compute_generalized_sobol_indices=True,
    compute_covariance_matrix_in_time=True,
    # Conditional analysis
    allow_conditioning_results_based_on_metric=True,
    condition_results_based_on_metric="NSE",
    condition_results_based_on_metric_value=0.7
)

# Apply to simulation
config.apply_to_uqsim(uqsim)
```

#### Extended Command-Line Arguments

All advanced features can now be accessed through command-line arguments in bash scripts:

```bash
python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --model "hbvsask" \
    --uq_method "sc" \
    --sc_p_order 3 \
    --sc_q_order 7 \
    --config_file data/configurations/configuration_hbv_10D.json \
    --compute_kl_expansion_of_qoi \
    --kl_expansion_order 10 \
    --compute_generalized_sobol_indices \
    --compute_covariance_matrix_in_time \
    --allow_conditioning_results_based_on_metric \
    --condition_results_based_on_metric NSE \
    --condition_results_based_on_metric_value 0.7
```

For detailed information about the configuration system, see `uqef_dynamic/config/README.md`.
For extended command-line arguments, see `docs/extended_arguments.md`.

## Configuration Files

The framework uses JSON configuration files to define:
- Model parameters and their distributions
- Time settings for simulations
- UQ method settings
- Output and post-processing settings

Example configuration files can be found in the `data/configurations/` directory. A typical configuration file includes:

- `time_settings`: Defines the simulation timespan, resolution, etc.
- `model_settings`: Model-specific settings
- `simulation_settings`: Settings for the simulation run
- `parameters`: List of model parameters with their distributions

#### Explanation of the relevant `simulation_settings` in the configuration file

TODO Extend this part with more settings

- `simulation_settings:mode` can be set to either `"continuous"` or `"sliding_window"`.

If you want to run the simulation in **autoregressive** mode—where the Quantity of Interest (QoI) is defined as the difference between the current model output and a scaled version of the previous (observed) model output—you must set the following options in the configuration file:

- `simulation_settings:autoregressive_model_first_order = True`
- `simulation_settings:scale_factor_autoregressive_model_first_order = <value between 0 and 1>`, e.g., `0.7` or `0.9`


## Available Models

For now, the framework includes several models:

1. **HBV/HBV-SASK**: Hydrological model
   ```bash
   --model hbvsask --config_file data/configurations/configuration_hbv_5D.json
   ```

2. **LARSIM**: Large Area Runoff Simulation Model; Note: You have to have access and install the Larsim_Utility_Set tool

   ```bash
   --model larsim --config_file data/configurations/configuration_larsim.json
   ```

3. **Battery**: PyBaMM battery model
   ```bash
   --model battery --config_file data/configurations/configuration_battery.json
   ```

4. **Oscillator Models**: Simple and linear damped oscillator models
   ```bash
   --model simple_oscillator --config_file data/configurations/configuration_simple_oscillator.json
   ```

5. **Ishigami**: Test function model
   ```bash
   --model ishigami --config_file data/configurations/configuration_ishigami.json
   ```

The `models/` directory contains implementations of various models that can be used with the framework. The `time_dependent_baseclass/` provides a common interface for all time-dependent models, ensuring consistent handling of time series data.


## Custom Model and Statistics
You can create custom models by inheriting from the `TimeDependentModelBase` class in `uqef_dynamic.models.time_dependent_baseclass`. Similarly, you can implement custom statistics by inheriting from the `TimeDependentStatisticsBase` class in `uqef_dynamic.statistics.time_dependent_statistics_baseclass`. 


## Parallel Computing

The framework supports parallel computing using MPI:

```bash
mpiexec -n <num_processes> python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --mpi \
    --mpi_method "MpiPoolSolver" \
    --num_cores <threads_per_process> \
    --parallel_statistics \
    [other options]
```

## Running the Simulation on HPC

All the simulation studies were executed on Linux Cluster HPC systems. The launch scripts are specific to that cluster and are included in the `scripts/` subdirectory for reference.

### Example HPC Script

The `scripts/` directory contains several example scripts for running simulations on HPC systems. For example, `start_hbv_cm2_sc.sh` demonstrates how to set up and run a simulation using SLURM:

```bash
#!/bin/bash

#SBATCH -J hbv_sim
#SBATCH --clusters=cm2
#SBATCH --partition=cm2_std
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=22
#SBATCH --time=2:00:00

module load python/3.6_intel
module load intel-mpi/2019-intel
source /path/to/conda/env/activate uqef_env

mpiexec -n $SLURM_NTASKS python /path/to/UQEF-Dynamic/uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --outputResultDir /path/to/results \
    --inputModelDir /path/to/model/data \
    --sourceDir /path/to/source \
    --config_file /path/to/config.json \
    --model "hbvsask" \
    --uncertain all \
    --uq_method "sc" \
    --sc_q_order 7 \
    --sc_p_order 3 \
    --mpi \
    --mpi_method "MpiPoolSolver" \
    --parallel_statistics \
    --compute_Sobol_t \
    --compute_Sobol_m
```

To adapt these scripts for other HPC environments:

1. Modify the job scheduler directives (e.g., from SLURM to PBS)
2. Update the module load commands for your specific HPC environment
3. Adjust the paths to match your installation
4. Set the appropriate number of nodes and tasks per node

## Paths Definitions

### Paths Definitions from UQEF-Dynamic

- `inputModelDir`: Path to the model input data
- `outputResultDir`: Directory to store results (saves dictionary with arguments, nodes distribution, nodes files, statistic plot results, statistics files)
- `outputModelDir`: Usually the same as outputResultDir, where model will store its output files
- `uqsim.configuration_object["Directories"]["working_dir"] = outputResultDir/model_runs`
- `sourceDir`: Path to the source code

## Output Files

The framework generates several output files:

- `uqsim_args.pkl`: Saved arguments dictionary
- `configurationObject`: Configuration object used for the simulation
- `dict_info.pkl`: Saved dictionary with different relevant data/configurations
- `nodes.simnodes.zip`: Simulation nodes
- `time_info.txt`: Timing information
- `df_simulations.pkl`: All simulated data (if configured to save all the simulation data)
- `df_index_parameter.pkl`: Parameter values for all simulation runs
- `df_index_parameter_gof.pkl`: Goodness-of-fit values over different parameter values (i.e., for all simulation runs)
- `df_state.pkl`: State results

files saved for each specific QoI:
- `statistics_dictionary_qoi_{qoi}.pkl`: Statistics for each QoI (over all the timestamp)
- `generalized_sobol_indices_{qoi}.pkl`: Generalized Sobol indices for each QoI
- `f_kl_surrogate_df_{qoi}.pkl`: Pandas DataFrame storing the learned KL (+PCE) surrogate

files saved for each specific QoI and timestamp (if configured in that way):
- `statistics_dictionary_{qoi}_{timestamp}.pkl`: Statistics for each QoI and timestamp
- `gpce_surrogate_{qoi}_{timestamp}.npy`: gPCE surrogate for each QoI and timestamp
- `gpce_coeffs_{qoi}_{timestamp}.npy`: gPCE coefficients for each QoI and timestamp
- `generalized_sobol_indices_{qoi}_{timestamp}.npy`: Generalized Sobol indices for each QoI and timestamp

Additional files for specific models and simulation set-ups:
- `df_measured.pkl`: Measured data (if available)


## Author

**Ivana Jovanovic Buha**
Technical University of Munich (TUM)
Email: ivana.jovanovic@tum.de

## Repository

GitHub: [https://github.com/ivanajovanovic/UQEF-Dynamic.git](https://github.com/ivanajovanovic/UQEF-Dynamic.git)

## Citation

If you use UQEF-Dynamic in your research, please cite:

```bibtex
@software{uqef_dynamic,
  author = {Jovanovic Buha, Ivana},
  title = {UQEF-Dynamic},
  version = {1.0},
  url = {https://github.com/ivanajovanovic/UQEF-Dynamic.git},
  institution = {Technical University of Munich}
}
```

## Acknowledgments

UQEF-Dynamic builds upon several excellent open-source projects:
- **UQEF**: As a based framework for efficient FUQ and in-parallel model execution
- **chaospy**: For probabilistic modeling, MC-based sampling and polynomial chaos expansion functionalities
- **mpi4py**: For MPI parallelization support
- **NumPy/SciPy**: For numerical computing foundations

## Version History

- **v0.1** (Current): Production-stable release with comprehensive UQ methods and parallel computing support

