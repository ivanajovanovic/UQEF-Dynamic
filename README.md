# UQEF-Dynamic Framework

Software tool for Efficient Forward Uncertainty Quantification of Dynamical Models 

## Overview

UQEF-Dynamic is a comprehensive framework for Uncertainty Quantification (UQ) and Global Sensitivity Analysis (SA) for dynamic models, with a particular focus on hydrological models and other time-dependent systems. The framework extends the capabilities of the UQEF (Uncertainty Quantification Execution Framework) to handle time-dependent processes.

<!-- This code is licensed under the GNU Lesser General Public License version 3 or
later, see `COPYING` and `COPYING.LESSER`. -->

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

## Requirements/Dependencies

### Core Dependencies

- **Python**: Compatible with Python 3.11 (recommended)
- **UQEF**: Uncertainty Quantification Execution Framework (core library)
- **Chaospy**: Python toolbox for performing uncertainty quantification
- **NumPy/SciPy/Pandas**: For numerical computations and data handling
- **MPI Libraries**: For parallel computing (mpi4py)
- **Visualization**: Matplotlib, Plotly, Seaborn for visualization and plotting

### Dependency Files

The project includes several requirements files:

- `requirements.txt`: Basic dependencies list
- `requirements/requirements_py311.txt`: Dependencies with fixed versions for Python 3.11

### Other Dependencies

- **LARSIM Model**: Require Larsim_Utility_Set library
- **Battery Models**: Require PyBaMM (Python Battery Mathematical Modelling)
- **Sparse Space Scientific Pipelines**: Require sparseSpACE toolbox

### Setting Up the Environment

The recommended way to set up the environment is using conda:

```bash
# Create a new conda environment with Python 3.11
conda create -n uqef_env python=3.11
conda install -n uqef_env --file requirements/requirements_py311.txt
conda activate uqef_env

# Install Chaospy
$(which pip) install chaospy

# Install UQEF (assuming the repository is cloned)
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

## Usage

The framework is typically used through the scientific pipelines, with the main entry point being `uq_simulation_uqsim.py`. The workflow generally involves:

1. Defining a configuration file
2. Selecting a model and UQ method
3. Running the simulation
4. Post-processing and analyzing the results


## How to Run the Code/Simulation

The UQEF-Dynamic framework is primarily used through the scientific pipelines, with the main entry point being `uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py`. The `uqef_dynamic/scientific_pipelines/` subdirectory contains workflows for other different types of scientific simulations.

### Basic Usage

```bash
python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --config_file data/configurations/configuration_hbv_5D.json \
    --model hbvsask \
    --inputModelDir /path/to/model/data \
    --outputResultDir /path/to/output/directory \
    --sourceDir /path/to/source/code \
    --uq_method sc \
    --sc_q_order 7 \
    --sc_p_order 3
```

### Configuration Files

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

### UQ Methods

The framework supports several UQ methods:

1. **Monte Carlo (MC)** - (Quasi-) Random sampling:
   ```bash
   --uq_method mc --mc_numevaluations 10000 --sampling_rule latin_hypercube
   ```

2. **Pseudo-spectral Projection (PSP) and Stochastic Collocation (SC)** - Polynomial chaos expansion:
   ```bash
   --uq_method sc --sc_q_order 7 --sc_p_order 3 --sc_quadrature_rule G
   ```

3. **Saltelli Method** - Sampling approach for computing the Sobol indices calculation:
   ```bash
   --uq_method saltelli --mc_numevaluations 10000 --sampling_rule latin_hypercube
   ```

### Available Models

The framework includes several models:

1. **HBV/HBV-SASK**: Hydrological model
   ```bash
   --model hbvsask --config_file data/configurations/configuration_hbv_5D.json
   ```

2. **LARSIM**: Large Area Runoff Simulation Model
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


### Parallel Computing

The framework supports parallel computing using MPI:

```bash
mpiexec -n <num_processes> python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --mpi \
    --mpi_method MpiPoolSolver \
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
    --model hbvsask \
    --uncertain all \
    --uq_method sc \
    --sc_q_order 7 \
    --sc_p_order 3 \
    --mpi \
    --mpi_method MpiPoolSolver \
    --parallel_statistics \
    --compute_Sobol_t \
    --compute_Sobol_m
```

To adapt these scripts for other HPC environments:

1. Modify the job scheduler directives (e.g., from SLURM to PBS)
2. Update the module load commands for your specific HPC environment
3. Adjust the paths to match your installation
4. Set the appropriate number of nodes and tasks per node

### Path Configuration

When running on HPC systems, you need to configure several paths:

- `inputModelDir`: Path to the model input data
- `outputResultDir`: Path where results will be stored
- `sourceDir`: Path to the source code
- `workingDir`: Working directory for model runs (created automatically)

## Pre-run Setup

Before each run, perform the following steps:

1. Ensure your conda environment is properly set up
2. Check that all required dependencies are installed
3. Verify that the configuration file is correctly set up
4. Create the output directory if it doesn't exist

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



## Input Arguments

The framework supports numerous command-line arguments to control the simulation. Most of these arguments are inherited by the UQEF tool. Here are the most important ones:

### Basic Arguments

- `--config_file`: Path to the configuration file
- `--model`: The model to use (e.g., hbvsask, larsim, ishigami)
- `--inputModelDir`: Directory of the input model
- `--outputResultDir`: Directory to store the output results
- `--sourceDir`: Source directory

### UQ Method Arguments

- `--uq_method`: UQ method to use (mc, sc, saltelli)
- `--mc_numevaluations`: Number of evaluations for Monte Carlo method
- `--sc_q_order`: Number of collocation points in each direction for Stochastic Collocation
- `--sc_p_order`: Number of terms in Polynomial Chaos Expansion
- `--sc_quadrature_rule`: Quadrature rule for Stochastic Collocation (G, clenshaw_curtis, etc.)
- `--sc_poly_rule`: Polynomial rule for Stochastic Collocation
- `--sampling_rule`: Sampling rule (random, sobol, latin_hypercube, etc.)

### Parallelization Arguments

- `--mpi`: Enable MPI
- `--mpi_method`: MPI method to use (MpiPoolSolver, LinearSolver)
- `--num_cores`: Number of cores to use for parallel execution
- `--parallel_statistics`: Enable parallel statistics computation
- `--chunksize`: Chunk size for parallel processing
- `--mpi_chunksize`: Chunk size for MPI

### Statistics Arguments

- `--compute_Sobol_t`: Enable computation of Sobol total indices
- `--compute_Sobol_m`: Enable computation of Sobol main indices
- `--compute_Sobol_m2`: Enable computation of Sobol second-order indices
- `--disable_statistics`: Disable statistics computation
- `--instantly_save_results_for_each_time_step`: Save results for each time step instantly

For a complete list of arguments and their explanations, refer to the `docs` subfolder or run:

```bash
python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py --help


## Authors

- Ivana Jovanovic Buha
- Florian Kuenzner