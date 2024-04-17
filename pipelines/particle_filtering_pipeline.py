import inspect
import pathlib
import pandas as pd
import sys
import os
from collections import defaultdict
import numpy as np

# importing modules/libs for plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
import matplotlib.pyplot as plt
# pd.options.plotting.backend = "plotly"

# for parallel computing
import multiprocessing
# import concurrent.futures
import psutil
# for message passing
# from mpi4py import MPI

import chaospy as cp

# TODO - change these paths accordingly
# sys.path.insert(1, '/work/ga45met/Hydro_Models/HBV-SASK-py-tool')
sys.path.insert(1, '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic')
from common import utility
from common import transport_map
from hbv_sask import hbvsask_utility as hbv
from hbv_sask import HBVSASKModel as hbvmodel

TIME_COLUMN_NAME = 'TimeStamp'
INDEX_COLUMN_NAME = "Index_run"
PLOT_FORCING_DATA = True

def run_model_single_time_stamp_single_particle(hbvsaskModelObject, date_of_interest,
                                    parameter_value_dict, state_values_dict, unique_index_model_run = 0):
    """
    Runs the HBV-SASK model for a single time stamp and a single particle.

    Args:
        hbvsaskModelObject (object): An instance of the HBV-SASK model.
        date_of_interest (str): The date of interest in the format 'YYYY-MM-DD'.
        parameter_value_dict (dict): A dictionary containing the parameter values for the model.
        state_values_dict (dict): A dictionary containing the initial state values for the model.
        unique_index_model_run (int, optional): An optional unique index for the model run. Defaults to 0.

    Returns:
        tuple: A tuple containing the following elements:
            - unique_index_model_run (int): The unique index for the model run.
            - y_t_model (float): The model output for the given time stamp.
            - y_t_observed (float): The observed output for the given time stamp.
            - x_t_plus_1 (dict): The model states for the next time stamp.
            - parameter_value_dict (dict): The parameter values used for the model run.
    """
    assert date_of_interest >= hbvsaskModelObject.start_date_predictions, \
        "Sorry, the date of interest is before start_date_predictions"
    assert date_of_interest <= hbvsaskModelObject.end_date, "Sorry, the date of interest is after end_date"

    # print(f"The current day - {date_of_interest}")
    date_of_interest_plus_one = pd.to_datetime(date_of_interest) + pd.DateOffset(days=1)

    forcing = hbvsaskModelObject.time_series_measured_data_df.loc[[date_of_interest, ], :].copy()

    state_values_dict["WatershedArea_km2"] = 1434.73
    state_values_dict[hbvsaskModelObject.time_column_name] = date_of_interest
    
    initial_condition_df = pd.DataFrame(state_values_dict, index=[0])

    results_array_changed_param = hbvsaskModelObject.run(
        i_s=[unique_index_model_run,],
        parameters=[parameter_value_dict, ],
        createNewFolder=False,
        take_direct_value=True,
        forcing=forcing,
        initial_condition_df=initial_condition_df
    )

    # extract y_t produced by the model
    model_output_dict = results_array_changed_param[0][0]['result_time_series'].loc[date_of_interest].to_dict()
    y_t_model = model_output_dict["Q_cms"]

    # extract x_(t+1) model states for the next time-stamp
    state_dict = results_array_changed_param[0][0]['state_df'].loc[date_of_interest_plus_one].to_dict()
    x_t_plus_1 = state_dict

    # extract y_t_observed
    if hbvsaskModelObject.read_measured_streamflow:
        measured_output_date = forcing["streamflow"].values[0]
    else:
        measured_output_date = None
    y_t_observed = measured_output_date

    # return model_output_dict, state_dict, measured_output_date
    return unique_index_model_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict


def calculate_likelihood(y_t_observed, y_t_model, error_variance):
    """
    Computing Gaussian like likelihood
    """
    if y_t_observed is not None and y_t_model is not None:
        exponent = -0.5 * ((y_t_observed - y_t_model) ** 2) / error_variance
        likelihood = np.exp(exponent) / np.sqrt(2 * np.pi * error_variance)
        return likelihood
    else:
        return 0


def systematic_resample(weights):
    """
    Mapping samples i to the new samples j all with the same weights 1/N_p
    """
    N_p = len(weights)
    positions = (np.arange(N_p) + np.random.random()) / N_p  # (?)
    cumulative_sum = np.cumsum(weights)  # CDF of particles
    indices = np.zeros(N_p, dtype=int)
    i, j = 0, 0
    # while i < N_p:
    #     if cumulative_sum[j] > positions[i]:
    #         indices[i] = j
    #         i += 1
    #     else:
    #         j += 1
    while j < N_p:
        if cumulative_sum[i] > positions[j]:
            indices[j] = i
            # indices[i] = j
            j += 1
        else:
            i += 1
    return indices


def perturb_parameters(parameters, perturbation_factor=0.15):
    perturbed_parameters = {}
    for key, value in parameters.items():
        # Use the absolute value to ensure a non-negative scale
        perturbation = np.random.normal(0, perturbation_factor * abs(value))  # Todo - change this to take into the account the current std of the parameter
        perturbed_parameters[key] = value + perturbation
    return perturbed_parameters


def define_the_transport_map_parameterization(D, maxorder=5):
    # =============================================================================
    # Define the transport map parameterization
    # =============================================================================

    # Next, we define the map component functions used in the triangular transport 
    # map. The map definition requires two lists of lists: one for the monotone 
    # part (basis functions which do depend on the last argument) and one for the
    # nonmonotone part (basis functions which do not depend on the last argument).
    # Each entry in those lists is another list that defines the basis functions.
    # Polynomial basis functions are lists of integers, with a potential keyword
    # such as 'HF' appended to mark it as a Hermite function. RBFs or related basis
    # functions are defined as strings such as 'RBF 0' or 'iRBF 7'. 
    # 
    # Example: --------------------------------------------------------------------
    #
    # monotone = [
    #   [ [0] ],
    #   [ [1], [0,0,1,'HF'] ] ]
    # nonmonotone = [
    #   [ [] ],
    #   [ [], [0], [0,0], [0,0,'HF], 'RBF 0'] ]
    #
    # Explanation: ----------------------------------------------------------------
    #
    # Monotone [list]
    #   |
    #   └―― Map component 1 [list] (last argument: entry x_{0})
    #   |       |
    #   |       └―― [0] Basis function 1 (linear term for entry 0)
    #   |
    #   └―― Map component 2 [list] (last argument: entry x_{1})
    #           |
    #           └―― [1] Basis function 1 (linear term for entry 1)
    #           |
    #           └―― [0,0,1,'HF'] Basis function 2 (cross-term: quadratic Hermite function for entry 0, linear Hermite function for entry 1)
    #
    # Nonmonotone [list]
    #   |
    #   └―― Map component 1 [list] (valid arguments: constant)
    #   |       |
    #   |       └―― [] Basis function 1 (constant term)
    #   |
    #   └―― Map component 2 [list] (Valid arguments: consant, x_{0})
    #           |
    #           └―― [] Basis function 1 (constant term)
    #           |
    #           └―― [0] Basis function 2 (linear term for entry x_{0})
    #           |
    #           └―― [0,0] Basis function 3 (quadratic term for entry x_{0})
    #           |
    #           └―― [0,0,'HF'] Basis function 4 (quadratic Hermite function for entry x_{0})
    #           |
    #           └―― 'RBF 0' Basis function 5 (radial basis function for entry x_{0})

    # Create empty lists for the map component specifications
    monotone    = []
    nonmonotone = []

    # Here, we try  different form of map parameterization. Let's try using maps
    # with separable monotonicity. These are often much more efficient, but do not
    # allow for cross-terms or nonmonotone basis functions in the 'monotone' list.
    for k in range(D):
        
        # Level 1: Add an empty list entry for each map component function
        monotone.append([])
        nonmonotone.append([]) # An empty list "[]" denotes a constant
        
        # Level 2: We initiate the nonmonotone terms with a constant
        nonmonotone[-1].append([])

        # Nonmonotone part --------------------------------------------------------

        # Go through every polynomial order
        for order in range(maxorder):
            
            # We only have non-constant nonmonotone terms past the first map 
            # component, and we already added the constant term earlier, so only do
            # this for the second map component function (k > 0).
            if k > 0: 
                
                # The nonmonotone basis functions can be as nonmonotone as we want.
                # Hermite functions are generally a good choice.
                nonmonotone[-1].append([k-1]*(order+1)+['HF'])
                
        # Monotone part -----------------------------------------------------------
        
        # Let's get more fancy with the monotone part this time. If the order  we 
        # specified is one, then use a linear term. Otherwise, use a few monotone 
        # special functions: Left edge terms, integrated radial basis functions, 
        # and right edge terms
        
        # The specified order is one
        if maxorder == 1:
            
            # Then just add a linear term
            monotone[-1].append([k])
            
        # Otherweise, the order is greater than one. Let's use special terms.
        else:
            
            # Add a left edge term. The order matters for these special terms. 
            # While they are placed according to marginal quantiles, they are 
            # placed from left to right. We want the left edge term to be left.
            monotone[-1].append('LET '+str(k))
                    
            # Lets only add maxorder-1 iRBFs
            for order in range(maxorder-1):
                
                # Add an integrated radial basis function
                monotone[-1].append('iRBF '+str(k))
        
            # Then add a right edge term 
            monotone[-1].append('RET '+str(k))
    return  monotone, nonmonotone


def transform_samples_with_transport_map(parameter_samples_matrix):
    # =============================================================================
    # Use transport map to transform current parameter samples to standard Gaussian
    # =============================================================================
    # Define the transport map parameterization
    # Create empty lists for the map component specifications
    monotone    = []
    nonmonotone = []
    # require polynomial basis terms up to order 5
    maxorder    = 1
    monotone, nonmonotone = define_the_transport_map_parameterization(D=parameter_samples_matrix.shape[1], maxorder=maxorder)
    # =============================================================================
    # Create the transport map object
    # =============================================================================
    # With the map parameterization (nonmonotone, monotone) defined and the target
    # samples (X) obtained, we can start creating the transport map object.
    # To begin, delete any map object which might already exist.
    if "tm" in globals():
        del tm

    # Create the transport map object tm
    tm     = transport_map.transport_map(
        monotone                = monotone,                 # Specify the monotone parts of the map component function
        nonmonotone             = nonmonotone,              # Specify the nonmonotone parts of the map component function
        X                       = parameter_samples_matrix, # = np.random.uniform(size=(N,D)), # Dummy input A N-by-D matrix of training samples (N = ensemble size, D = variable space dimension)
        polynomial_type         = "hermite function",       # What types of polynomials did we specify? The option 'Hermite functions' here are re-scaled probabilist's Hermites, to avoid numerical overflow for higher-order terms
        monotonicity            = "separable monotonicity",   # Are we ensuring monotonicity through 'integrated rectifier' or 'separable monotonicity'?
        standardize_samples     = True,                     # Standardize the training ensemble X? Should always be True
        workers                 = 1,                        # Number of workers for the parallel optimization.
        # quadrature_input        = {                         # Keywords for the Gaussian quadrature used for integration
        #     'order'         : 25,
        #     'adaptive'      : False,
        #     'threshold'     : 1E-9,
        #     'verbose'       : False,
        #     'increment'     : 6}
        # regularization          = "l2",
        # regularization_lambda   = lmbda,
        verbose                 = False
        )

    # Optimize the transport maps. This takes a while, it's an extremeley complicated map.
    tm.optimize()

    # Store the coefficients in a dictionary
    dict_coeffs = {
        'coeffs_mon'    : tm.coeffs_mon,
        'coeffs_nonmon' : tm.coeffs_nonmon}
    
    # Save the dictionary
    # pickle.dump(dict_coeffs,open('dict_coeffs_order='+str(maxorder)+'.p','wb'))
    # =============================================================================
    # Apply the map
    # =============================================================================  
    # -----------------------------------------------------------------------------
    # forward map from the target to the reference
    # -----------------------------------------------------------------------------
    # we apply the map forward. This transforms samples from
    # the target into samples from the reference (a standard Gaussian)

    # We can evaluate the forward map with the following command:
    Z_gen   = tm.map(parameter_samples_matrix)
    # =============================================================================
    return Z_gen


def main_routine(num_processes, number_of_particles, working_dir_name="trial_single_run_hbvsaskmodel_7d_filtering"):
    # =========================================================
    # Model Related Setup
    # =========================================================
    # Defining paths
    # TODO - change these paths accordingly
    hbv_model_data_path = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
    configuration_file = pathlib.Path('/work/ga45met/Hydro_Models/HBV-SASK-py-tool/configurations/configuration_hbv_6D.json')
    inputModelDir = hbv_model_data_path
    basis = "Oldman_Basin"  # 'Banff_Basin'
    workingDir = hbv_model_data_path / basis / "model_runs" / working_dir_name
    directory_for_saving_plots = workingDir
    if not str(directory_for_saving_plots).endswith("/"):
        directory_for_saving_plots = str(directory_for_saving_plots) + "/"

    ne = number_of_particles

    # Creating Model Object
    writing_results_to_a_file = False
    plotting = False
    createNewFolder = False # create a separate folder to save results for each model run
    hbvsaskModelObject = hbvmodel.HBVSASKModel(
        configurationObject=configuration_file,
        inputModelDir=inputModelDir,
        workingDir=workingDir,
        basis=basis,
        writing_results_to_a_file=writing_results_to_a_file,
        plotting=plotting
    )

    # In case one wants to modify dates compared to those set up in the configuration object / deverge from these setting
    start_date = '2006-03-30 00:00:00'
    end_date = '2007-06-30 00:00:00'
    spin_up_length = 365  # 365*3
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # dict_with_dates_setup = {"start_date": start_date, "end_date": end_date, "spin_up_length":spin_up_length}
    run_full_timespan = False
    hbvsaskModelObject.set_run_full_timespan(run_full_timespan)
    hbvsaskModelObject.set_start_date(start_date)
    hbvsaskModelObject.set_end_date(end_date)
    hbvsaskModelObject.set_spin_up_length(spin_up_length)
    simulation_length = (hbvsaskModelObject.end_date - hbvsaskModelObject.start_date).days - hbvsaskModelObject.spin_up_length
    if simulation_length <= 0:
        simulation_length = 365
    hbvsaskModelObject.set_simulation_length(simulation_length)
    hbvsaskModelObject.set_date_ranges()
    hbvsaskModelObject.redo_input_and_measured_data_setup()

    # Get to know some of the relevant time settings, read from a json configuration file
    # print(f"start_date: {hbvsaskModelObject.start_date}")
    # print(f"start_date_predictions: {hbvsaskModelObject.start_date_predictions}")
    # print(f"end_date: {hbvsaskModelObject.end_date}")
    # print(f"full_data_range is {len(hbvsaskModelObject.full_data_range)} "
    #       f"hours including spin_up_length of {hbvsaskModelObject.spin_up_length} hours")
    # print(f"simulation_range is of length {len(hbvsaskModelObject.simulation_range)} hours")

    # Plot forcing data and observed streamflow
    hbvsaskModelObject._plot_input_data(read_measured_streamflow=True)

    list_of_dates_of_interest = list(pd.date_range(
        start=hbvsaskModelObject.start_date_predictions, end=hbvsaskModelObject.end_date, freq="1D"))
    # list_of_dates_of_interest = list(pd.date_range(
    #     start='2007-04-30 00:00:00', end='2007-05-30 00:00:00', freq="1D"))

    # =========================================================
    # Code for creating the initial parameter values and state values
    # =========================================================

    # read configuration file and save it as a configuration object
    # with open(configuration_file, 'rb') as f:
    #     configurationObject = dill.load(f)
    # simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)
    # on can as well just fetch the configurationObject from the hbvsaskModelObject
    configurationObject = hbvsaskModelObject.configurationObject

    # Sampling from the parameter space
    list_of_single_dist = []
    list_of_single_standard_dist = []
    list_of_single_min_1_1_dist = []
    param_names = []
    for param in configurationObject["parameters"]:
        # for now the Uniform distribution is only supported
        if param["distribution"] == "Uniform":
            param_names.append(param["name"])
            list_of_single_dist.append(cp.Uniform(param["lower"], param["upper"]))
            list_of_single_standard_dist.append(cp.Uniform(0, 1))
            list_of_single_min_1_1_dist.append(cp.Uniform(-1, 1))
        else:
            raise NotImplementedError(f"Sorry, the distribution {param['distribution']} is not supported yet")
    joint_params = cp.J(*list_of_single_dist)
    joint_standard_params  = cp.J(*list_of_single_standard_dist)
    joint_min_1_1_params = cp.J(*list_of_single_min_1_1_dist)

    # Sampling from the state space
    list_of_single_dist = []
    list_of_single_standard_dist = []
    list_of_single_min_1_1_dist = []
    state_names = []
    for state in configurationObject["states"]:
        # for now the Uniform distribution is only supported
        if state["distribution"] == "Uniform":
            state_names.append(state["name"])
            list_of_single_dist.append(cp.Uniform(state["lower"], state["upper"]))
            list_of_single_standard_dist.append(cp.Uniform(0, 1))
            list_of_single_min_1_1_dist.append(cp.Uniform(-1, 1))
        else:
            raise NotImplementedError(f"Sorry, the distribution {state['distribution']} is not supported yet")
    joint_states = cp.J(*list_of_single_dist)
    joint_standard_states  = cp.J(*list_of_single_standard_dist)
    joint_min_1_1_states = cp.J(*list_of_single_min_1_1_dist)

    list_parameter_value_particles = joint_params.sample(number_of_particles, rule="random").T # rule can as well be: 'sobol' | 'random' | "latin_hypercube" | "halton"
    list_state_values_particles = joint_states.sample(number_of_particles, rule="random").T
    list_unique_index_model_run_list = list(range(0, len(list_state_values_particles)))

    # Plotting inital distribution of parameter values
    fig, axs = plt.subplots(1, len(param_names), figsize=(20, 10))
    fig_plotly = make_subplots(rows=1, cols=len(param_names))
    # dict_of_distriubtions_over_parameters_initial = defaultdict(list, {key:[] for key in param_names})
    for idx in range(len(param_names)):
        parameter_name = param_names[idx]
        fig_plotly.append_trace(
                go.Histogram(
                    x=list_parameter_value_particles[:,idx],
                    name=parameter_name
                ), row=1, col=idx + 1)
        axs[idx,].hist(list_parameter_value_particles[:,idx], bins=100, density=True, alpha=0.5)
        plt.setp(axs[idx,], xlabel=f'{parameter_name}')
        axs[idx,].grid()
    plt.setp(axs[0], ylabel='Histogram')
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "initial_param_distribution.png"))
    plt.savefig(fileName)
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "initial_param_distribution.html"))
    pyo.plot(fig_plotly, filename=fileName)

    # Create list-of-dictionaries for the parameter values and state values, instead of only matrices of values
    list_of_dict_parameter_value_particles = []
    for parameter_particle in list_parameter_value_particles:
        list_of_dict_parameter_value_particles.append(dict(zip(param_names, parameter_particle)))
    list_of_dict_state_value_particles = []
    for state_particle in list_state_values_particles:
        list_of_dict_state_value_particles.append(dict(zip(state_names, state_particle)))
    list_parameter_value_particles = list_of_dict_parameter_value_particles
    list_state_values_particles = list_of_dict_state_value_particles

    uniform_particle_weights = np.ones(number_of_particles) / number_of_particles

    # Initialize the error variance with a default value
    final_predicted_streamflow = defaultdict(list, {key:[] for key in list_of_dates_of_interest})
    final_observed_streamflow = defaultdict(list, {key:[] for key in list_of_dates_of_interest})
    dates = []
    mse = 0
    error_variance = 5.0
    current_model_output_max = 0.0

    # =========================================================
    # Particle Filtering
    # =========================================================

    # Data structure to store the results
    y_t_model_per_date_dict = defaultdict(list, {key:[] for key in list_of_dates_of_interest})
    data_structure_over_dates = []

    # Outer loop that goes over all date from the configuration json
    for index_date_of_interest in range(len(list_of_dates_of_interest)):
        date_of_interest = list_of_dates_of_interest[index_date_of_interest]
        # print(f"date_of_interest - {date_of_interest}")

        new_list_parameter_value_particles = []
        new_list_state_values_particles = []
        new_list_unique_index_model_run_list = []
        y_t_models_for_date = []
        updated_weights = [] #np.zeros(len(list_parameter_value_particles))

        # This part of the code is for parallel computing of independent particles, i.e., model runs
        def process_particles_concurrently(particles_to_process):
            with (multiprocessing.Pool(processes=num_processes) as pool):
                for index_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict in \
                        pool.starmap(run_model_single_time_stamp_single_particle, \
                                     [(hbvsaskModelObject, date_of_interest, particle[0], particle[1], particle[2]) \
                                      for particle in particles_to_process]):
                    yield index_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict

        row = {}
        likelihood_over_rows = []
        weights_over_rows = []
        resampling_over_rows = []

        # Iterating over the particles and model results for each partcle
        for index_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict in process_particles_concurrently(
    zip(list_parameter_value_particles, list_state_values_particles, list_unique_index_model_run_list)
        ):
            y_t_models_for_date.append(y_t_model)
            new_list_unique_index_model_run_list.append(index_run)
            new_list_state_values_particles.append(x_t_plus_1)
            new_list_parameter_value_particles.append(parameter_value_dict)
            
            if y_t_observed is not None:
                error_variance = 0.2 * y_t_observed
            else: 
                error_variance = 5.0

            # Compute the likelikhood function for each particle and probabilities p(y_(t)|x_(t), theta_(t))
            likelihood = calculate_likelihood(y_t_observed, y_t_model, error_variance)
            updated_weights.append(likelihood)

            likelihood_over_rows.append(likelihood)

        y_t_models_for_date = np.asfarray(y_t_models_for_date)
        current_model_output_max = np.max(y_t_models_for_date) if np.max(y_t_models_for_date) > current_model_output_max else current_model_output_max
        y_t_model_per_date_dict[date_of_interest] = y_t_models_for_date
        updated_weights = np.asfarray(updated_weights)

        row[TIME_COLUMN_NAME] = date_of_interest  # date_of_interest_over_rows
        row['y_t_model'] = y_t_models_for_date  # y_t_model_over_rows
        row[INDEX_COLUMN_NAME] = new_list_unique_index_model_run_list  # index_run_over_rows
        row['likelihood'] = likelihood_over_rows

        average_predicted_streamflow = np.mean(y_t_models_for_date)  # this can as well be computed from row['y_t_model']
        if y_t_observed is not None and index_date_of_interest>0:  # disregard the zero step with allways a huge mse
            difference = average_predicted_streamflow - y_t_observed
            mse += difference ** 2
            
        # After calculating all the likelihoods and storing them in updated weights
        # Here begins likelihood/weights normalization:
        total_weight = np.sum(updated_weights)
        if abs(total_weight) < 1e-9:
            # if total weight is 0, to prevent underflow, we reset it uniformly
            normalized_weights = np.ones(len(updated_weights)) / len(updated_weights)
        else:
            normalized_weights = updated_weights / total_weight
        # Now normalized_weights contains the normalized likelihoods 
        normalized_weights = np.asfarray(normalized_weights)
        
        # Overwrite the lists storing the particles (i.e., state, parameter values and unique particle indices) for the next time-stamp
        list_unique_index_model_run_list = new_list_unique_index_model_run_list
        # list_state_values_particles  = copy.deepcopy(new_list_state_values_particles) 
        # list_parameter_value_particles = copy.deepcopy(new_list_parameter_value_particles) 

        # maybe this is not necessarry and just brings to longer execution...
        list_of_tuple_with_parameter_values = []
        for i in range(len(list_parameter_value_particles)):
            list_of_tuple_with_parameter_values.append(tuple(list_parameter_value_particles[i].values()))
        row['old_parameter_value'] = list_of_tuple_with_parameter_values

        # Resample particles based on updated weights
        resample_indices = systematic_resample(normalized_weights)
        list_parameter_value_particles = [new_list_parameter_value_particles[i] for i in resample_indices]
        list_state_values_particles = [new_list_state_values_particles[i] for i in resample_indices]
        uniform_particle_weights = [uniform_particle_weights[i] for i in resample_indices]  # TODO this is probably unnecessary

        # Perturb the parameters of resampled particles
        perturbation_factor = 0.15  # TODO Think about this
        list_of_tuple_with_parameter_values = []
        list_of_lists_with_parameter_values = []
        dict_of_distriubtions_over_parameters_for_a_date = defaultdict(list, {key:[] for key in param_names})
        for i in range(len(list_parameter_value_particles)):
            list_parameter_value_particles[i] = perturb_parameters(list_parameter_value_particles[i],
                                                                   perturbation_factor)
            list_of_tuple_with_parameter_values.append(tuple(list_parameter_value_particles[i].values()))
            list_of_lists_with_parameter_values.append(list(list_parameter_value_particles[i].values()))

            # print(f"DEBUGGING perturbed parameters values in dict {i} - {list_parameter_value_particles[i]}")
            for parameter_name in param_names:
                dict_of_distriubtions_over_parameters_for_a_date[parameter_name].append(list_parameter_value_particles[i][parameter_name])

        # Save one big matrix of particle values, might be used later one for transformation of the samples
        parameter_samples_matrix = list(zip(*list_of_lists_with_parameter_values))  # this should be a matrix of size number_of_particles x number_of_parameters
        parameter_samples_matrix = np.asfarray(parameter_samples_matrix).T

        row['weights'] = normalized_weights
        row['resample_indices'] = resample_indices
        row['new_parameter_value'] = list_of_tuple_with_parameter_values
        data_structure_over_dates.append(row)

        print(f"Date: {date_of_interest.strftime('%Y-%m-%d')}")
        print(f"Predicted Averaged Streamflow: {average_predicted_streamflow} m^3/s")
        print(f"Observed Streamflow: {y_t_observed} m^3/s")

        final_predicted_streamflow[date_of_interest] = average_predicted_streamflow
        final_observed_streamflow[date_of_interest] = y_t_observed
        dates.append(date_of_interest)

    mse_total = mse / len(dates)
    print(f"Final predicted streamflow: {final_predicted_streamflow}")
    print(f"Final observed streamflow: {final_observed_streamflow}")
    print(f"Total MSE: {mse_total}; RMSE: {np.sqrt(mse_total)}")
    
    print(f"FINISH")
        
    # =========================================================
    # Creating Data Structures whihc will be used for further analysis (saving it) and plotting
    # =========================================================

    # Create one big DataFrame storing all the simulation over time and over different particles
    unfolded_data_structure_over_dates = []
    for item in data_structure_over_dates:
        for i in range(len(item[INDEX_COLUMN_NAME])):
            single_row_dict = {key: value[i] if isinstance(value, list) else value for key, value in item.items()}
            unfolded_data_structure_over_dates.append(single_row_dict)
    df = pd.DataFrame(unfolded_data_structure_over_dates)
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "all_simulations.pkl"))
    df.to_pickle(fileName, compression="gzip")
    print(f"df - {df}")

    # Create DataFrames from dictionaries storing the final predicted and observed streamflow
    final_predicted_streamflow_df = pd.DataFrame.from_dict(final_predicted_streamflow, orient='index')
    final_predicted_streamflow_df.rename(columns={0: 'predicted_streamflow'}, inplace=True)
    final_observed_streamflow_df = pd.DataFrame.from_dict(final_observed_streamflow, orient='index')
    final_observed_streamflow_df.rename(columns={0: 'observed_streamflow'}, inplace=True)
    merged_df = final_predicted_streamflow_df.merge(final_observed_streamflow_df, left_index=True, right_index=True)
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "averaged_and_simulated.pkl"))
    merged_df.to_pickle(fileName, compression="gzip")

    # Create one big Matrix containing all model simulations over particles and over dates
    # Extract the lists from the dictionary
    lists = list(y_t_model_per_date_dict.values())
    # Use the zip function to transpose the lists into columns
    y_t_model_per_date_matrix = list(zip(*lists))
    assert len(y_t_model_per_date_matrix[0]) == len(list_of_dates_of_interest)

    # =========================================================
    # Plotting the results
    # =========================================================

    fig, axs = plt.subplots(1, len(param_names), figsize=(20, 10))
    fig_plotly = make_subplots(rows=1, cols=len(param_names))
    for idx in range(len(param_names)):
        parameter_name = param_names[idx]
        # Visualize data from the last dict_of_distriubtions_over_parameters_for_a_date 
        dict_of_distriubtions_over_parameters_for_a_date[parameter_name] = np.asfarray(dict_of_distriubtions_over_parameters_for_a_date[parameter_name])
        min_value = dict_of_distriubtions_over_parameters_for_a_date[parameter_name].min() - abs(dict_of_distriubtions_over_parameters_for_a_date[parameter_name].min())*0.001
        max_value= dict_of_distriubtions_over_parameters_for_a_date[parameter_name].max() + abs(dict_of_distriubtions_over_parameters_for_a_date[parameter_name].max())*0.001
        t = np.linspace(min_value, max_value, 1000)
        distribution = cp.GaussianKDE(dict_of_distriubtions_over_parameters_for_a_date[parameter_name], h_mat=0.005 ** 2)
        axs[idx,].hist(dict_of_distriubtions_over_parameters_for_a_date[parameter_name], bins=100, density=True, alpha=0.5)
        axs[idx,].plot(t, distribution.pdf(t), label=f"KDE {parameter_name}")
        plt.setp(axs[idx,], xlabel=f'{parameter_name}')
        axs[idx,].grid()
        fig_plotly.append_trace(
                go.Histogram(
                    x=dict_of_distriubtions_over_parameters_for_a_date[parameter_name],
                    name=parameter_name
                ), row=1, col=idx + 1)
    plt.setp(axs[0], ylabel='PDF')
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "final_param_distribution.png"))
    plt.savefig(fileName)
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "final_param_distribution.html"))
    pyo.plot(fig_plotly, filename=fileName)

    print(f"DEBUGGING - {parameter_samples_matrix.shape}")
    standar_parameter_samples_matrix = transform_samples_with_transport_map(parameter_samples_matrix)
    print(f"DEBUGGING - {standar_parameter_samples_matrix.shape}")
    # Plotting final distribution of transformed parameter values
    fig_plotly = make_subplots(rows=1, cols=len(param_names))
    for idx in range(len(param_names)):
        parameter_name = param_names[idx]
        fig_plotly.append_trace(
                go.Histogram(
                    x=standar_parameter_samples_matrix[:,idx],
                    name=parameter_name
                ), row=1, col=idx + 1)
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "final_transformed_param_distribution.html"))
    pyo.plot(fig_plotly, filename=fileName)

    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = go.Figure()

    # Add scatter plot for the model predictions
    # for date, y_t_models in y_t_model_per_date_dict.items():
    #     fig.add_trace(
    #         go.Scatter(x=[date] * len(y_t_models), y=y_t_models, mode='markers', name='Model Predictions',
    #                    marker=dict(color='blue', size=5, opacity=0.5), showlegend=False),  # Increased opacity
    #         secondary_y=False,
    #     )

    # Add all the model simulation over different particles
    # grouped = df.groupby([INDEX_COLUMN_NAME,])
    # groups = grouped.groups
    # print(f"DEBUGGING groups - {groups}")
    # lines = [
    #     go.Scatter(
    #         x=list(df.loc[val_indices, TIME_COLUMN_NAME].values),
    #         y=list(df.loc[val_indices, 'y_t_model'].values),
    #         showlegend=False,
    #         # legendgroup=colours[i],
    #         mode="lines",
    #         line=dict(
    #             color='LightSkyBlue'),
    #         opacity=0.1
    #     )
    #     for key, val_indices in groups.items()
    # ]
    lines = [
        go.Scatter(
            x=dates,
            y=single_row,
            showlegend=False,
            # legendgroup=colours[i],
            mode="lines",
            line=dict(
                color='LightSkyBlue'),
            opacity=0.1
        )
        for single_row in y_t_model_per_date_matrix
    ]
    for trace in lines:
        fig.add_trace(trace)

    # Add forcing data streamflow
    if PLOT_FORCING_DATA:
        reset_index_at_the_end = False
        if hbvsaskModelObject.time_series_measured_data_df.index.name != TIME_COLUMN_NAME:
            hbvsaskModelObject.time_series_measured_data_df.set_index(TIME_COLUMN_NAME, inplace=True)
            reset_index_at_the_end = True
        
        temp = hbvsaskModelObject.time_series_measured_data_df[hbvsaskModelObject.time_series_measured_data_df.index.isin(list_of_dates_of_interest)]  # Sample forcing data for plotting only the list_of_dates_of_interest
        N_max = temp['precipitation'].max()
        fig.add_trace(
            go.Bar(x=temp.index,
                   y=temp['precipitation'], 
                   name='Precipitation', yaxis="y2", marker_color='magenta'
                ),
        )
        if reset_index_at_the_end:
            hbvsaskModelObject.time_series_measured_data_df.reset_index(inplace=True)
            hbvsaskModelObject.time_series_measured_data_df.rename(columns={hbvsaskModelObject.time_series_measured_data_df.index.name: TIME_COLUMN_NAME}, inplace=True)

    # Add observed streamflow
    fig.add_trace(
        go.Scatter(x=merged_df.index, y=merged_df['observed_streamflow'], name='Observed Streamflow',
                   line=dict(color='orange', width=2.5)),
    )

    # Add predicted streamflow
    fig.add_trace(
        go.Scatter(x=merged_df.index, y=merged_df['predicted_streamflow'], name='Averaged Predicted Streamflow',
                   line=dict(color='blue', width=2.5)),
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date", autorange=True, range=[hbvsaskModelObject.start_date_predictions, hbvsaskModelObject.end_date], type="date")

    fig.update_yaxes(title_text="Q [cm/s]", side="left", domain=[0, 0.7], mirror=True, tickfont={"color": "#d62728"},
                     tickmode="auto", ticks="inside", titlefont={"color": "#d62728"}, range=[0, 100])

    fig.update_layout(
        # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=f'Predicted vs. Observed Streamflow with {ne} particles',
        showlegend=True,
        template="plotly_white",
    )

    if PLOT_FORCING_DATA:
        fig.update_layout(
            yaxis2=dict(
            anchor="x",
            domain=[0.7, 1],
            mirror=True,
            range=[N_max, 0],
            side="right",
            tickfont={"color": '#1f77b4'},
            nticks=3,
            tickmode="auto",
            ticks="inside",
            titlefont={"color": '#1f77b4'},
            title="N [mm/h]",
            type="linear",
            )
        )

    # Show the figure
    fig.show()
    fileName = "particle_filter_states_and_parameters.html"
    fileName = directory_for_saving_plots + fileName
    pyo.plot(fig, filename=fileName)


if __name__ == "__main__":

    # Number of parallel processes
    num_processes = multiprocessing.cpu_count()
    print(f"Number of parallel processes = {num_processes}")

    number_of_particles = ne = 2000  # 50, 100, 500

    for i in range(1):
        # working_dir_name=f"trial_single_run_hbvsaskmodel_7d_filtering/run_{i}"
        working_dir_name=f"trial_single_run_hbvsaskmodel_7d_filtering_III"
        main_routine(num_processes, number_of_particles, working_dir_name)

