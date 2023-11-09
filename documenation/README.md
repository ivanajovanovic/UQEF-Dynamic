## JSON Configuration File Documentation

### Time Settings

- `start_day`, `start_month`, `start_year`, `start_hour`, `start_minute`: Define the start time of the simulation.
- `end_day`, `end_month`, `end_year`, `end_hour`, `end_minute`: Define the end time of the simulation.
- `run_full_timespan`: If set to `False`, the simulation will not run for the full time span.
- `spin_up_length`: The length of the spin-up period in days.
- `simulation_length`: The length of the simulation in days.
- `resolution`: The resolution of the simulation (e.g., "daily").
- `cut_runs`: If set to `False`, the simulation will not cut runs.
- `timestep`: The time step of the simulation in minutes.

### Model Settings

- `basis`: The basis of the model.
- `plotting`: If set to `False`, the simulation will not plot results.
- `writing_results_to_a_file`: If set to `False`, the simulation will not write results to a file.
- `corrupt_forcing_data`: If set to `False`, the simulation will not corrupt forcing data.
- `model_paths`: Contains paths related to the model.
- `hbv_model_path`: The path to the HBV model data.

### Simulation Settings

- `qoi`: The quantity of interest for the simulation.
- `qoi_column`: The column in the data that contains the quantity of interest.
- `transform_model_output`: The method to transform model output.
- `read_measured_data`: If set to `True`, the simulation will read measured data.
- `qoi_column_measured`: The column in the data that contains the measured quantity of interest.
- `objective_function_qoi`: The objective function for the quantity of interest.
- `calculate_GoF`: If set to `True`, the simulation will calculate the goodness of fit.
- `objective_function`: The objective function for the simulation.
- `mode`: The mode of the simulation (e.g., "continuous").
- `interval`: The interval for the simulation.
- `min_periods`: The minimum number of periods for the simulation.
- `method`: The method for the simulation (e.g., "avrg").
- `center`: The center for the simulation.
- `compute_gradients`: If set to `False`, the simulation will not compute gradients.
- `eps_gradients`: The epsilon for gradients.
- `gradient_method`: The method for computing gradients.
- `gradient_analysis`: If set to `True`, the simulation will perform gradient analysis.
- `compute_active_subspaces`: If set to `False`, the simulation will not compute active subspaces.
- `save_gradient_related_runs`: If set to `True`, the simulation will save runs related to gradients.

### Parameters

Contains the parameters for the simulation. Each parameter has a name, distribution, lower and upper bounds, and a default value.

### Unused Parameters

Contains the parameters that are not used in the simulation. Each unused parameter has a name, distribution, lower and upper bounds, and a default value.