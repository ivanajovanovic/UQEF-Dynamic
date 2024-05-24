# 

## Introduction

This repository contains supporting code for the paper...


<!-- This code is licensed under the GNU Lesser General Public License version 3 or
later, see `COPYING` and `COPYING.LESSER`. -->

All the simulations studies were executed on the Linux Cluster HPC systems. The
launch scripts are specific to that cluster and are included in the
`scripts/` subdirectory for reference. It would be necessary to modify
these job launch scripts to run these studies on another cluster. Also, one might have to change accordingly the paths specified in some scripts under `uqef_dynamic/scientific_pipelines/` subdirectory.

## Dependencies

The primary dependencies are UQEF and Chaospy tools/libraires. Secondary dependencies for pre and post-processing include various data analysis, plotting and statistical libraries. 

The code is compatible with Python 3.11 and the dependencies are specified in the
`requirements/requirements_py311.txt` file:

    conda create -n uqef_env --file requirements_py311.txt
    conda install -n uqef_env -c conda-forge nb_conda_kernels
    conda install -n uqef_env -c conda-forge pyproj
    conda activate uqef_env

    $(which pip) install chaospy

    cd UQEF/
    git checkout parallel_statistics
    $(which python) setup_new.py install
    cd ../

The file `requirements/requirements_py311-not-fixed.txt` contains the same dependencies without
fixed version requirements.

We refer the user to the followig script to help him/her set-up the conda environment `scripts/set_up_new_conda_env.sh`

## UQEF and UQEF-Dynamic explanation of the input arguments

### UQsim load/restore
- `uqsim_file`
- `uqsim_store_to_file`
- `uqsim_restore_from_file`
- `disable_calc_statistics`
- `disable_recalc_statistics`
- `disable_statistics`

### Model and result directories
- `inputModelDir`
- `outputModelDir`
- `outputResultDir`
- `sourceDir`

### Model settings
- `model`
- `model_variant`
- `simulate_wait`

### UQ method and uncertain parameter settings
- `uncertain` "all"
- `uq_method` "sc" | "saltelli" | "mc" | "ensemble"
- `regression`
- `mc_numevaluations`
- `sc_q_order`
- `sc_p_order`
- `sc_sparse_quadrature`
- `sc_quadrature_rule` "g" | "p" | "genz_keister_24" | "leja" | "clenshaw_curtis"
- `sc_poly_normed`
- `sc_poly_rule` "gram_schmidt" | "three_terms_recurrence" | "cholesky"
- `sampling_rule` "random" | "sobol" | "latin_hypercube" | "halton"  | "hammersley"
- `transformToStandardDist`
- `sampleFromStandardDist`
- `config_file`
- `read_nodes_from_file`
- `parameters_file`
- `parameters_setup_file`

### Solver settings
- `parallel`
- `num_cores`
- `mpi`
- `mpi_method`
- `mpi_combined_parallel`

### Statistics related settings
- `parallel_statistics`
- `compute_Sobol_t`
- `compute_Sobol_m`
- `compute_Sobol_m2`
- `save_all_simulations` - if set to True one file with all model forward simulations will be saved, e.g., df_all_simulations.pkl
- `store_qoi_data_in_stat_dict` - if set to True a result_dict from the Statistics obejct will have an entry 'qoi_values'
- `store_gpce_surrogate_in_stat_dict` - Only relevant for sc mode when the gPCE surrogate is produced
- `collect_and_save_state_data` - Only relevant for models which produce some state data as well
- `instantly_save_results_for_each_time_step`

#### Yet not part of the set of UQEF arguments, though possible to transfer as paramteres to the Statistics class:
- `dict_what_to_plot`
- `compute_sobol_total_indices_with_samples`: This is only relevant in the mc-saltelli's approach. Set to `True` when `uq_method`==`mc` and `compute_Sobol_t`==`True`
- `save_gpce_surrogate`: `True` or `False`; if True a gpce surrogate for each QoI for each time step is saved in a separate file
- `compute_other_stat_besides_pce_surrogate`: `True` or `False`; relevant only when ther is a gPCe computation; question weather statitics such mean, variance, pecentiles, Sobol SI, as beside gPCE surrogate computation

### Chunk parameters
- `chunksize`
- `mpi_chunksize`

### Runtime analysis and optimisation parameters
- `analyse_runtime`
- `opt_runtime`
- `opt_runtime_gpce_Dir`
- `opt_algorithm`
- `opt_strategy`

## JSON Configuration File Documentation

### Time Settings

- `start_day`, `start_month`, `start_year`, `start_hour`, `start_minute`: Define the start time of the simulation.
- `end_day`, `end_month`, `end_year`, `end_hour`, `end_minute`: Define the end time of the simulation.
- `run_full_timespan`: If set to `False`, the simulation will not run for the full time span. If set to `True` one should leave-out the `simulation_length` entry.
- `spin_up_length`: The length of the spin-up period in days.
- `simulation_length`: The length of the simulation in days.
- `resolution`: The resolution of the simulation (e.g., "daily" or "hourly").
- `cut_runs`: If set to `False`, the simulation will not cut runs. THis is relevan for some more complext models like, for example,  Larsim.
- `timestep`: The time step of the simulation in minutes.

### Model Settings

- `basis`: The basis of the model.
- `plotting`: If set to `False`, the simulation will not plot results.
- `writing_results_to_a_file`: If set to `False`, the simulation will not write results to a file.
- `corrupt_forcing_data`: If set to `False`, the simulation will not corrupt forcing data.
- `model_paths`: Contains paths related to the model.
- `hbv_model_path`: The path to the HBV model data.

### Model Paths
- Should contain key-value list of model related paths

### Simulation Settings

- `qoi`: The quantity of interest for the simulation. Maybe a single string or list of string. Maybe a string representing an output of interest or "GoF" string when some objective function should be regraded as a QoI.
- `qoi_column`: The column in the data that contains the output of interest.
- `autoregressive_model_first_order`: 
- `transform_model_output`: The method to transform model output.
- `read_measured_data`: If set to `True`, the simulation will read measured data.
- `qoi_column_measured`: The column in the data that contains the measured quantity of interest.
- `objective_function_qoi`: The objective function for the quantity of interest.
- `calculate_GoF`: If set to `True`, the simulation will calculate the goodness of fit.
- `objective_function`: The objective function for the simulation.
- `mode`: The mode of the simulation ("continuous" or "sliding_window").
- `interval`: The interval for the simulation.
- `min_periods`: The minimum number of periods for the simulation.
- `method`: The method for the simulation. Can have one of the following values:  "avrg", "min" or "max".
- `center`: The center for the simulation. Can have one of the following values: "center", "left" or "right".
- `compute_gradients`: If set to `False`, the simulation will not compute gradients.
- `eps_gradients`: The epsilon for gradients.
- `gradient_method`: The method for computing gradients.
- `gradient_analysis`: If set to `True`, the simulation will perform gradient analysis.
- `compute_active_subspaces`: If set to `False`, the simulation will not compute active subspaces.
- `save_gradient_related_runs`: If set to `True`, the simulation will save runs related to gradients.

### Parameters

Contains the parameters for the simulation. Each parameter has a name, distribution, lower and upper bounds, and a default value. It might as well have a "values_list" which can either be a single value or a list of values; if "distribution" is to "None" then aither the "default" value or "values_list" will be used, depending on other set-up from UQEF

### Unused Parameters

Contains the parameters that are not used in the simulation. Each unused parameter has a name, distribution, lower and upper bounds, and a default value.

## The (possible) output
uqsim_args.pkl
configurationObject
nodes.simnodes.zip
time_info.txt
df_all_index_parameter_values.pkl
df_state_results.pkl
[optional]
df_all_index_parameter_gof_values.pkl
df_all_simulations.pkl
[one file for each QoI and each timestamp]
statistics_dictionary_{qoi}_{timestamp}.pkl
gpce_surrogate_{qoi}_{timestamp}.pkl
gpce_coeffs_{qoi}_{timestamp}.npy