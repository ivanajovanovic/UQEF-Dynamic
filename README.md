# Forward Uncertainty Quantification of the Larsim Water Balance Model

This repository contains code for performing forward uncertainty quantification of the Larsim Model.

## Requirements

- Compatible with Python 3.6+
- Access to Larsim_Utility_Set and UQEF library, as well as necessary data to run and stimulate the Larsim model.
- Additional Software requirements:
  - chaospy
  - Requirements of the UQEF library and UQEF library itself
  - Requirements of the Larsim_Utility_Set library and Larsim_Utility_Set library itself

## How to Run the Code/Simulation

*Details on how to run the code/simulation*

## Running the Simulation on HPC

*Details on how to run the simulation on HPC*

## Pre-run Setup

Before each run, perform the following steps:

- Delete all subfolders in the `model_runs` folder.
- Optionally, delete all whm and lila files inside `../Larsim-data/WHM Regen/master_configuration`.

## Paths Definitions

### Paths Definitions from LarsimUtilityFunctions

- home_dir
- data_dir (the parent folder of the **Larsim-data** folder)
- scratch_dir (folder to save the output)
- sourceDir (folder where source code is)
- working_dir
- larsim_data_path
- sim_folder

### Paths Definitions from UQEF

- inputModelDir - larsim_data_path
- outputModelDir - outputResultDir
- outputResultDir - scratch_dir/folder_for_current_results (save dictionary with arguments, save nodes dist, save nodes files, save statistic plot results, save statistics file.)
  - uqsim.configuration_object["Directories"]["working_dir"] = outputResultDir/model_runs
- sourceDir - sourceDir

## Output Files

- outputResultDir/df_measured.pkl
- outputResultDir/df_simulated.pkl
- outputResultDir/df_unaltered_ergebnis.pkl
- outputResultDir/gof_past_sim_meas.pkl
- outputResultDir/gof_unaltered_meas.pkl
- outputResultDir/master_configuration/
- If `LarsimModel.run_and_save_simulations` is set to `True`, the following files will also be saved:
  - df_Larsim_run_i.pkl
  - parameters_Larsim_run_i.pkl
  - goodness_of_fit_i.pkl
  - df_Larsim_run_processed_i.pkl

## Input Arguments


### Brief documentation for the input arguments:

--smoketest: A flag to run a test of the script to verify the environment.
--uqsim_file: The file to load or store the UQsim state.
--uqsim_store_to_file: A flag to store the UQsim state to a file.
--uqsim_restore_from_file: A flag to restore the UQsim state from a file.
--disable_calc_statistics: A flag to disable the calculation of statistics.
--disable_recalc_statistics: A flag to disable the recalculation of statistics.
--disable_statistics: A flag to disable statistics.
-im, --inputModelDir: The directory of the input model.
-om, --outputModelDir: The directory to store the output model.
-or, --outputResultDir: The directory to store the output results.
-src, --sourceDir: The source directory.
--model: The model to use.
--model_variant: The variant of the model to use.
--simulate_wait: A flag to enable waiting during simulation.
--uncertain: The uncertain parameter settings.
--uq_method: The UQ method to use.
--regression: A flag to enable regression.
--mc_numevaluations: The number of evaluations for the Monte Carlo method.
--sc_q_order: The number of collocation points in each direction for the Stochastic Collocation method.
--sc_p_order: The number of terms in Polynomial Chaos Expansion.
--sc_sparse_quadrature: A flag to enable sparse quadrature for the Stochastic Collocation method.
--sc_quadrature_rule: The quadrature rule for the Stochastic Collocation method.
--sc_poly_normed: A flag to enable normalized polynomials for the Stochastic Collocation method.
--sc_poly_rule: The polynomial rule for the Stochastic Collocation method.
--sampling_rule: The sampling rule to use.
--transformToStandardDist: A flag to enable transformation to standard distribution.
--sampleFromStandardDist: A flag to enable sampling from standard distribution.
--config_file: The configuration file to use.
--read_nodes_from_file: A flag to enable reading nodes from a file.
--parameters_file: The parameters file to use.
--parameters_setup_file: The parameters setup file to use.
--parallel: A flag to enable parallel execution.
--num_cores: The number of cores to use for parallel execution.
--mpi: A flag to enable MPI.
--mpi_method: The MPI method to use.
--mpi_combined_parallel: A flag to enable combined parallel execution with MPI.
--instantly_save_results_for_each_time_step: A flag to enable saving results for each time step instantly.
--parallel_statistics: A flag to enable parallel statistics.
--compute_Sobol_t: A flag to enable computation of Sobol t-statistics.
--compute_Sobol_m: A flag to enable computation of Sobol m-statistics.
--compute_Sobol_m2: A flag to enable computation of Sobol m2-statistics.
--chunksize: The chunk size to use.
--mpi_chunksize: The chunk size to use with MPI.
--analyse_runtime: A flag to enable runtime analysis.
--opt_runtime: A flag to enable runtime optimization.
--opt_runtime_gpce_Dir: The directory for GPCE runtime optimization.
--opt_algorithm: The algorithm to use for optimization.
--opt_strategy: The strategy to use for optimization.

