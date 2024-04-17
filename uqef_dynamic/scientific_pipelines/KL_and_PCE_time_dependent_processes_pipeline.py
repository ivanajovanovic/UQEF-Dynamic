from collections import defaultdict
import pathlib
import pandas as pd
import numpy as np
import os
import time

from typing import List, Optional, Dict, Any, Union, Tuple

# plotting
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo

# for parallel computing
import multiprocessing

import chaospy as cp

from uqef_dynamic.utils import utility

# HBV-SASK Model related stuff
from uqef_dynamic.models.hbv_sask import HBVSASKModel as hbvmodel
from uqef_dynamic.models.hbv_sask import hbvsask_utility as hbv_utility

# definition of some 'static' variables
PLOT_ALL_THE_RUNS = True  # set this to False when there are a lot of samples
PLOT_FORCING_DATA = False
time_column_name = TIME_COLUMN_NAME = 'TimeStamp'
index_column_name = INDEX_COLUMN_NAME = "Index_run"
QOI_COLUMN_NAME = "model"  # "Value"
QOI_COLUM_NAME_MESURED = None
QOI_COLUMN_NAME_CENTERED = QOI_COLUMN_NAME + "_centered"
MODEL = 'hbvsask'  #'hbvsask' # 'banchmark_model' or 'simple_model' or 'hbvsask'
if MODEL == 'hbvsask':
    QOI_COLUMN_NAME = "Q_cms"
    QOI_COLUM_NAME_MESURED = "streamflow"
    QOI_COLUMN_NAME_CENTERED = QOI_COLUMN_NAME + "_centered"
    PLOT_FORCING_DATA = True
    OBSERVED_COLUM_NAME = 'streamflow'
single_qoi_column = QOI_COLUMN_NAME
single_qoi_column_centered = QOI_COLUMN_NAME + "_centered"
UQ_ALGORITHM = "pce"  # "mc" or "kl" or "pce"

# N_kl =  60 # [2, 4, 6, 8, 10]
# sgq_level = 6
# numCollocationPointsPerDim = q = 8
# poly_normed = True
# order = p = 3
# ne = number_of_particles = numSamples = numEvaluations = N = 2000

READ_FROM_FILE_SG_QUADRATURE = True
PARALLEL_COMPUTING = True
COMPUTE_GENERALIZED_SOBOL_INDICES = False #True
REEVALUATE_SURROGET_MODEL = True

# =========================================================
# Utility Functions for the Algorithm 3
# =========================================================
# TODO Add doc-strings to these functions!
# TODO Move these fucntions to a separate file

def center_outputs(N, N_quad, df_simulation_result, weights, single_qoi_column, index_column_name=INDEX_COLUMN_NAME, algorithm: str= "samples", time_column_name: str=TIME_COLUMN_NAME):
    centered_outputs = np.empty((N, N_quad))

    single_qoi_column_centered = single_qoi_column + "_centered"
    if single_qoi_column_centered not in df_simulation_result.columns:
        utility.add_centered_qoi_column_to_df_simulation_result(df_simulation_result, algorithm=algorithm, weights=weights, single_qoi_column=single_qoi_column, time_column_name=time_column_name)

    grouped = df_simulation_result.groupby([index_column_name,])
    groups = grouped.groups
    for key, val_indices in groups.items():
        centered_outputs[int(key), :] = df_simulation_result.loc[val_indices, single_qoi_column_centered].values
    print(f"DEBUGGING - centered_outputs {centered_outputs}")
    print(f"DEBUGGING - centered_outputs.shape {centered_outputs.shape}")
    return centered_outputs


def compute_covariance_matrix_in_time(N_quad, centered_outputs, weights, algorithm: str= "samples"):
    covariance_matrix = np.empty((N_quad, N_quad))
    for c in range(N_quad):
        for s in range(N_quad):
            if algorithm == "samples" or algorithm == "mc":
                N = centered_outputs.shape[0]  # len(centered_outputs[:, c])
                covariance_matrix[s, c] = 1/(N-1) * \
                np.dot(centered_outputs[:, c], centered_outputs[:, s])
            elif algorithm == "quadrature" or algorithm == "pce" or algorithm == "sc" or algorithm == "kl":
                if weights is None:
                    raise ValueError("Weights must be provided for quadrature-based algorithms")
                covariance_matrix[s, c] = np.dot(weights, centered_outputs[:, c]*centered_outputs[:,s])
            else:
                raise ValueError(f"Unknown algorithm - {algorithm}")
    return covariance_matrix


def plot_covariance_matrix(covariance_matrix, directory_for_saving_plots):
    plt.figure()
    plt.imshow(covariance_matrix, cmap='hot', interpolation='nearest')
    # Add a color bar to the side
    plt.colorbar()
    # Add title and labels if needed
    plt.title('Covariance Matrix')
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "covariance_matrix.png"))
    plt.savefig(fileName)


def solve_eigenvalue_problem(covariance_matrix, weights):
    # from scipy.linalg import eig
    # from scipy.linalg import eigh
    from numpy.linalg import eig
    from scipy.linalg import sqrtm

    # Solve Discretized (generalized) Eigenvalue Problem
    # Setting-up the system
    K = covariance_matrix
    # Check if the approximation of the covarriance matrix is symmetric
    cov_matrix_is_symmetric = np.array_equal(covariance_matrix, covariance_matrix.T)
    print(f"Check if the approximation of the covarriance matrix is symmetric - {cov_matrix_is_symmetric}")
    W = np.diag(weights)
    sqrt_W = sqrtm(W)
    LHS = sqrt_W@K@sqrt_W
    # cov_matrix_is_symmetric = np.array_equal(LHS, LHS.T)
    # print(f"Check if the approximation of the covarriance matrix is symmetric LHS - {cov_matrix_is_symmetric}")
    
    # Solving the system
    B = np.identity(LHS.shape[0])
    # Alternatively one can solve the standard eigenvalue problem with symmetric/Hermitian matrices
    # from numpy.linalg import eigh
    # eigenvalues_h, eigenvectors_h = eigh(LHS)
    eigenvalues, eigenvectors = eig(LHS)
    idx = eigenvalues.argsort()[::-1]   # Sort by descending real part of eigenvalues
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    eigenvalues = sorted_eigenvalues
    eigenvectors = sorted_eigenvectors
    # eigenvalues_real = np.asfarray([element.real for element in eigenvalues])
    # eigenvalues_real_scaled = eigenvalues_real/eigenvalues_real[0]
    final_eigenvectors = np.linalg.inv(sqrt_W)@eigenvectors
    eigenvectors = final_eigenvectors
    return eigenvalues, eigenvectors


def plot_eigenvalues(eigenvalues, directory_for_saving_plots):
    # Plotting the eigenvalues
    eigenvalues_real = np.asfarray([element.real for element in eigenvalues])
    eigenvalues_real_scaled = eigenvalues_real/eigenvalues_real[0]
    plt.figure()
    plt.yscale("log")
    plt.plot(eigenvalues, 'x')
    plt.title('Eigenvalues of the Covariance Operator')
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "eigenvalues_covariance_operator.png"))
    plt.savefig(fileName)
    plt.figure()
    plt.yscale("log")
    plt.plot(eigenvalues_real_scaled, 'x')
    plt.title('Scaled Eigenvalues of the Covariance Operator')
    fileName = os.path.abspath(os.path.join(str(directory_for_saving_plots), "scaled_eigenvalues_covariance_operator.png"))
    plt.savefig(fileName)


def setup_kl_expansion_matrix(eigenvalues, N_kl, N, N_quad, weights, centered_outputs, eigenvectors):
    # Approximating the KL Expansion - setting up the matrix
    f_kl_eval_at_params = np.empty((N_kl, N))
    # f_kl_eval_at_params_2 = np.empty((N_kl, N))

    # weights @ centered_outputs[k,:] @ eigenvectors[:,i]
    for i in range(N_kl):
        for k in range(N):
            # computing f_kl_eval_at_params[i, k]
    #         f_kl_eval_at_params_2[i, k] = np.dot(np.dot(weights, centered_outputs[k,:]), eigenvectors[:,i])
            f_kl_eval_at_params[i, k] = 0
            for m in range(N_quad):
                f_kl_eval_at_params[i, k] += weights[m]*centered_outputs.T[m,k]*eigenvectors[m,i]
    return f_kl_eval_at_params


def pce_of_kl_expansion(N_kl, polynomial_expansion, nodes, weights, f_kl_eval_at_params):
    # PCE of the KL Expansion
    f_kl_surrogate_dict = {}
    # f_kl_surrogate_coefficients = np.empty(N_kl, c_number)
    f_kl_surrogate_coefficients = []
    for i in range(N_kl):
        # TODO Change this data structure, make it that the keys are time-stampes to resamble result_dict_statistics
        f_kl_surrogate_dict[i] = {}
        # print(f"DEBUGGING - f_kl_eval_at_params[{i},:].shape - {f_kl_eval_at_params[i,:].shape}")
        f_kl_gPCE, f_kl_coeff = cp.fit_quadrature(polynomial_expansion, nodes, weights, f_kl_eval_at_params[i,:], retall=True)
        f_kl_surrogate_dict[i]["gPCE"] = f_kl_gPCE
        f_kl_surrogate_dict[i]["coeff"] = f_kl_coeff
    #     f_kl_surrogate_coefficients[i] = np.asfarray(f_kl_coeff).T
        f_kl_surrogate_coefficients.append(np.asfarray(f_kl_coeff))
    f_kl_surrogate_coefficients = np.asfarray(f_kl_surrogate_coefficients)
    return f_kl_surrogate_dict, f_kl_surrogate_coefficients


def computing_generalized_sobol_total_indices_from_kl_expan(
    f_kl_surrogate_coefficients: np.ndarray,
    polynomial_expansion: cp.polynomial,
    weights: np.ndarray,
    param_names: List[str],
    fileName: str,
    total_variance=None
):
    # TODO Important aspect here is if polynomial_expansion is normalized or not
    dic = polynomial_expansion.todict()
    alphas = []
    for idx in range(len(polynomial_expansion)):
        expons = np.array([key for key, value in dic.items() if value[idx]])
        alphas.append(tuple(expons[np.argmax(expons.sum(1))]))

    index = np.array([any(alpha) for alpha in alphas])

    dict_of_num = defaultdict(list)
    for idx in range(len(alphas[0])):
        dict_of_num[idx] = []

    # variance_over_time_array = []
    # for time_stamp in result_dict_statistics.keys():  
    #     coefficients = np.asfarray(result_dict_statistics[time_stamp]['coeff'])
    #     variance = np.sum(coefficients[index] ** 2, axis=0)
    #     variance_over_time_array.append(variance)
    #     for idx in range(len(alphas[0])):
    #         index_local = np.array([alpha[idx] > 0 for alpha in alphas])      # Compute the total Sobol indices
    #         dict_of_num[idx].append(np.sum(coefficients[index_local] ** 2, axis=0))  # scaling with norm of the polynomial corresponding to the index_local
    for i in range(f_kl_surrogate_coefficients.shape[0]):
        coefficients = np.asfarray(f_kl_surrogate_coefficients[i,:])
        variance = np.sum(coefficients[index] ** 2, axis=0)
        # variance_over_time_array.append(variance)
        for idx in range(len(alphas[0])):
            index_local = np.array([alpha[idx] > 0 for alpha in alphas])  # Compute the total Sobol indices
            dict_of_num[idx].append(np.sum(coefficients[index_local] ** 2, axis=0))

    # variance_over_time_array = np.asfarray(variance_over_time_array)
    if total_variance is None:
        # denum = np.dot(variance_over_time_array, weights)
        raise ValueError("Total variance must be provided")
    else:
        denum = total_variance
        
    for idx in range(len(alphas[0])):
        param_name = param_names[idx]
        num = np.sum(np.asfarray(dict_of_num[idx]), axis=0)
        s_tot_generalized = num/denum
        print(f"Generalized Total Sobol Index computed based on the PCE of KL expansion for {param_name} is {s_tot_generalized}")
        with open(fileName, 'a') as file:
            # Write each variable to the file followed by a newline character
            file.write(f'{param_name}: {s_tot_generalized}\n')


def computing_generalized_sobol_total_indices_from_poly_expan(
    result_dict_statistics: Dict[Any, Dict[str, Any]],
    polynomial_expansion: cp.polynomial,
    weights: np.ndarray,
    param_names: List[str],
    fileName: str,
    total_variance=None
):
    """
    Computes the generalized Sobol total indices from a polynomial expansion.
    The current implamantion of the function assumes that the polynomial expansion is normalized.
    One would have to do scaling with norms of the polynomials if they are not normalized.

    Args:
        result_dict_statistics (Dict[Any, Dict[str, Any]]): A dictionary containing the statistics of the results.
         Important assumtion is that it contains the coefficients of the polynomial expansion under 'coeff' key over time.
        polynomial_expansion (cp.polynomial): The polynomial expansion.
        weights (np.ndarray): An array of weights for time quadratures.
        param_names (List[str]): A list of parameter names.
        fileName (str): The name of the file to write the results to.

    Returns:
        None

    Raises:
        None
    """
    # TODO Important aspect here is if polynomial_expansion is normalized or not
    dic = polynomial_expansion.todict()
    alphas = []
    for idx in range(len(polynomial_expansion)):
        expons = np.array([key for key, value in dic.items() if value[idx]])
        alphas.append(tuple(expons[np.argmax(expons.sum(1))]))

    index = np.array([any(alpha) for alpha in alphas])

    dict_of_num = defaultdict(list)
    for idx in range(len(alphas[0])):
        dict_of_num[idx] = []

    variance_over_time_array = []

    for time_stamp in result_dict_statistics.keys():  
        coefficients = np.asfarray(result_dict_statistics[time_stamp]['coeff'])
        variance = np.sum(coefficients[index] ** 2, axis=0)
        variance_over_time_array.append(variance)
        for idx in range(len(alphas[0])):
            index_local = np.array([alpha[idx] > 0 for alpha in alphas])      # Compute the total Sobol indices
            dict_of_num[idx].append(np.sum(coefficients[index_local] ** 2, axis=0))  # scaling with norm of the polynomial corresponding to the index_local

    variance_over_time_array = np.asfarray(variance_over_time_array)
    if total_variance is None:
        denum = np.dot(variance_over_time_array, weights)
    else:
        denum = total_variance
    for idx in range(len(alphas[0])):
        param_name = param_names[idx]
        num = np.dot(np.asfarray(dict_of_num[idx]), weights)
        s_tot_generalized = num/denum
        print(f"Generalized Total Sobol Index for {param_name} is {s_tot_generalized}")
        with open(fileName, 'a') as file:
            # Write each variable to the file followed by a newline character
            file.write(f'{param_name}: {s_tot_generalized}\n')

# =========================================================
# Modele example
# =========================================================

# def model(t, alpha, beta, l):
#     return l*np.exp(-alpha*t)*(np.cos(beta*t)+alpha/beta*np.sin(beta*t))

class SimpleModel:
    def __init__(self, t=None, t_start=0, t_end=10, N=20):
        if t is None:
            self.t = np.linspace(t_start, t_end, N)
            self.t_start = t_start
            self.t_end = t_end
            self.N = N
        else:
            self.t = t
            self.t_start = t[0]
            self.t_end = t[-1]
            self.N = len(t)
        self.alpha = 0.5
        self.beta = 1.0
        self.l = 1.0
        self.configurationObject = {"parameters": 
        [
            {
                "name": "alpha",
                "distribution": "Uniform",
                "lower": 3/8, 
                "upper": 5/8
                }, 
            {
                "name": "beta",
                "distribution": "Uniform",
                "lower": 10/4, 
                "upper": 15/4
                }, 
            {
                "name": "l",
                "distribution": "Uniform",
                "lower": -5/4, 
                "upper": -3/4
                }
        ]
        }

    def set_configurationObject(self, configurationObject):
        self.configurationObject = configurationObject

    def set_time(self, t=None, t_start=0, t_end=10, N=20):
        if t is None:
            self.t = np.linspace(t_start, t_end, N)
            self.t_start = t_start
            self.t_end = t_end
            self.N = N
        else:
            self.t = t
            self.t_start = t[0]
            self.t_end = t[-1]
            self.N = len(t)

    def run(self, alpha=None, beta=None, l=None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if l is None:
            l = self.l
        return l*np.exp(-alpha*self.t)*(np.cos(beta*self.t)+alpha/beta*np.sin(beta*self.t))


def run_model_single_parameter_node(model, parameter_value, unique_index_model_run=0, qoi_column_name=QOI_COLUMN_NAME, qoi_column_name_measured=QOI_COLUM_NAME_MESURED, **kwargs):
    if MODEL == 'hbvsask':
        # take_direct_value should be True if parameter_value is a dict with keys being paramter name and values being parameter values;
        # if parameter_value is a list of parameter values corresponding to the order of the parameters in the configuration file, then take_direct_value should be False
        # it is assumed that model is a subclass of HydroModel from UQEF-Dynamic
        results_list = model.run(
            i_s=[unique_index_model_run, ],
            parameters=[parameter_value, ],
            createNewFolder=False,
            take_direct_value=False,
            merge_output_with_measured_data=True,
        )
        # extract y_t produced by the model
        y_t_model = results_list[0][0]['result_time_series'][qoi_column_name].to_numpy()
        # if qoi_column_name_measured is not None and qoi_column_name_measured in results_list[0][0]['result_time_series']:
        #     y_t_observed = results_list[0][0]['result_time_series'][qoi_column_name_measured].to_numpy()
        #     # y_t_observed = model.time_series_measured_data_df[qoi_column_name_measured].values
        # else:
        #     y_t_observed = None  
    elif MODEL == 'simple_model' or MODEL == 'banchmark_model':
        # t = kwargs['t']
        # y_t_model = model.run(parameter_value['alpha'], parameter_value['beta'], parameter_value['l'])
        y_t_model = model.run(*parameter_value, **kwargs)
        # y_t_observed = None
    else:
        raise NotImplementedError(f"Sorry, the model {MODEL} is not supported yet")
    # return unique_index_model_run, y_t_model, y_t_observed, parameter_value
    return unique_index_model_run, y_t_model, parameter_value


def extend_plot(fig, list_of_dates_of_interest, model, num_model_runs, plot_forcing_data=PLOT_FORCING_DATA, **kwargs):
    if MODEL == 'hbvsask':
        fig = hbv_utility.extend_hbv_plot_with_observed_and_forcing_data_and_update_layout(
            fig, list_of_dates_of_interest, model, num_model_runs, time_column_name=TIME_COLUMN_NAME, plot_forcing_data=plot_forcing_data, **kwargs)
    elif MODEL == 'simple_model' or MODEL == 'banchmark_model':
        # Set x-axis title
        fig.update_xaxes(title_text="Date", autorange=True, range=[list_of_dates_of_interest[0], list_of_dates_of_interest[-1]])

        # Update y-axis
        fig.update_yaxes(title_text="Y(t)", side="left", domain=[0, 1.0], mirror=True, tickfont={"color": "#d62728"},
                        tickmode="auto", ticks="inside", titlefont={"color": "#d62728"})

        fig.update_layout(
            # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=f'Predicted model output with {num_model_runs} parameter nodes',
            showlegend=True,
            template="plotly_white",
        )
    else:
        raise NotImplementedError(f"Sorry, the model {MODEL} is not supported yet")
    return fig

# =========================================================

def evaluate_gPCE_model_single_date_from_df_stat(single_time_stamp, df_stat, nodes):
    gPCE_model = df_stat.loc[single_time_stamp,"gPCE"]  # or df_stat.iloc?
    # print(f"DEBUGGING gPCE_model for a date {single_time_stamp} - {gPCE_model}")
    return single_time_stamp, np.array(gPCE_model(*nodes))

# =========================================================

def main_routine():
    start_time_model_simulations = time.time()

    # Number of parallel processes
    num_processes = multiprocessing.cpu_count()
    print(f"Number of parallel processes = {num_processes}")

    # Defining paths and Creating Model Object
    # TODO - change these paths accordingly
    if MODEL == 'hbvsask':
        hbv_model_data_path = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
        # configurationObject = pathlib.Path('/work/ga45met/Hydro_Models/HBV-SASK-py-tool/configurations/configuration_hbv_3D.json')
        configurationObject = pathlib.Path('/work/ga45met/Hydro_Models/HBV-SASK-py-tool/configurations/configuration_hbv_6D.json')
        # configurationObject = pathlib.Path('/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic/configurations/configuration_hbv_10D.json')
        inputModelDir = hbv_model_data_path
        basis = "Oldman_Basin"  # 'Banff_Basin'
        workingDir = hbv_model_data_path / basis / "model_runs" / "kl_and_pce_time_dependent_processes_pipeline_7d_pce_p3_sgql7_checking_pce_speed" #7d_kl40_p3_sgql7"  _mc_10d_10000
        # workingDir = hbv_model_data_path / basis / "model_runs" / "kl_and_pce_time_dependent_processes_pipeline_4d_kl_tt_beta_etf_fc" 

        # creating HBVSASK model object
        writing_results_to_a_file = False
        plotting = False
        createNewFolder = False  # create a separate folder to save results for each model run
        model = hbvmodel.HBVSASKModel(
            configurationObject=configurationObject,
            inputModelDir=inputModelDir,
            workingDir=workingDir,
            basis=basis,
            writing_results_to_a_file=writing_results_to_a_file,
            plotting=plotting
        )
    elif MODEL == 'simple_model' or MODEL == 'banchmark_model':
        # workingDir = pathlib.Path("/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic/banchmark_model_kl_and_pce_time_dependent_processes_pipeline")
        workingDir = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data/kl_and_pce_simple_model_kl_algo3_sgq_l6")
        workingDir.mkdir(parents=True, exist_ok=True)
        model = SimpleModel()
    else:
        raise NotImplementedError(f"Sorry, the model {MODEL} is not supported yet")

    directory_for_saving_plots = workingDir
    if not str(directory_for_saving_plots).endswith("/"):
        directory_for_saving_plots = str(directory_for_saving_plots) + "/"
    time_infoFileName = os.path.abspath(os.path.join(directory_for_saving_plots, f"time_info.txt"))

    # =========================================================
    # Time related set-up; more relevan for complex models
    # =========================================================
    # note: after this sub-routine one can interchangable use the following 
    # variables: t, list_of_dates_of_interest, time_quadrature

    if MODEL == 'hbvsask':
        # In case one wants to modify dates compared to those set up in the configuration object / deverge from these setting
        # if not, just comment out this whole part 
        start_date = '2006-03-30 00:00:00'
        end_date = '2007-06-30 00:00:00'
        spin_up_length = 365  # 365*3
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        # dict_with_dates_setup = {"start_date": start_date, "end_date": end_date, "spin_up_length":spin_up_length}
        run_full_timespan = False
        model.set_run_full_timespan(run_full_timespan)
        model.set_start_date(start_date)
        model.set_end_date(end_date)
        model.set_spin_up_length(spin_up_length)
        simulation_length = (model.end_date - model.start_date).days - model.spin_up_length
        if simulation_length <= 0:
            simulation_length = 365
        model.set_simulation_length(simulation_length)
        model.set_date_ranges()
        model.redo_input_and_measured_data_setup()

        # Get to know some of the relevant time settings, read from a json configuration file
        print(f"start_date: {model.start_date}")
        print(f"start_date_predictions: {model.start_date_predictions}")
        print(f"end_date: {model.end_date}")
        print(f"full_data_range is {len(model.full_data_range)} "
            f"hours including spin_up_length of {model.spin_up_length} hours")
        print(f"simulation_range is of length {len(model.simulation_range)} hours")

        # Plot forcing data and observed streamflow
        model._plot_input_data(read_measured_streamflow=True)

        # Outer loop that goes over all date from the configuration json
        list_of_dates_of_interest = list(pd.date_range(
            start=model.start_date_predictions, end=model.end_date, freq="1D"))
        time_quadrature = t = list_of_dates_of_interest
        N_quad = len(time_quadrature)
        t_starting = model.start_date_predictions
        t_final = model.end_date
    elif MODEL == 'simple_model' or MODEL == 'banchmark_model':
        t_starting = start_date = 0
        t_final = end_date = 10
        N_quad = 20
        time_quadrature = t = np.linspace(0, 10, N_quad)
        model.set_time(t=None, t_start=start_date, t_end=end_date, N=N_quad)
        list_of_dates_of_interest = model.t
    else:
        raise NotImplementedError(f"Sorry, the model {MODEL} is not supported yet")
    
    h = 1/(N_quad-1) #(t_final - t_starting)/(N_quad-1)
    weights_time = [h for i in range(N_quad)]
    assert len(t)==len(weights_time)
    weights_time[0] /= 2
    weights_time[-1] /= 2
    weights_time = np.asfarray(weights_time)
    # =========================================================
    # Code for seting-up uncertain parameters info and generating nodes in the parameter space
    # =========================================================
    # read configuration file and save it as a configuration object
    # with open(configuration_file, 'rb') as f:
    #     configurationObject = dill.load(f)
    # simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)
    # on can as well just fetch the configurationObject from the model
    configurationObject = model.configurationObject

    # Sampling from the parameter space or generating quadrature points
    # in short, generating parameter nodes in some way...
    list_of_single_dist = []
    list_of_single_standard_dist = []
    param_names = []
    for param in configurationObject["parameters"]:
        # for now the Uniform distribution is only supported
        if param["distribution"] == "Uniform":
            param_names.append(param["name"])
            list_of_single_dist.append(cp.Uniform(param["lower"], param["upper"]))
            if UQ_ALGORITHM == "mc":
                list_of_single_standard_dist.append(cp.Uniform(0, 1))
            elif UQ_ALGORITHM == "kl" or UQ_ALGORITHM == "pce":
                list_of_single_standard_dist.append(cp.Uniform(-1, 1))
            else:
                raise NotImplementedError(f"Sorry, the UQ_ALGORITHM {UQ_ALGORITHM} is not supported yet")
        else:
            raise NotImplementedError(f"Sorry, the distribution {param['distribution']} is not supported yet")
    joint_dist = cp.J(*list_of_single_dist)
    joint_dist_standard  = cp.J(*list_of_single_standard_dist)

    dim = len(param_names)

    # =========================================================
    # 1. Monte Carlo sampling - Generating nodes, i.e., uncertain parameter values
    # =========================================================

    if UQ_ALGORITHM == "mc":
        # this is actually the number of different samples in the parameter space
        ne = number_of_particles = numSamples = numEvaluations = N = 10**4 #10**3 #150
        rule = 'random'  # rule can as well be: 'sobol' | 'random' | "latin_hypercube" | "halton"
        print(f"Number of Particles: {ne}")

        # MC approach
        nodes = joint_dist_standard.sample(size=numSamples, rule=rule).round(4)  # Should be of size (dimxnumSamples)

    # =========================================================
    # 2. Quadrature approach
    # =========================================================

    elif UQ_ALGORITHM == "kl" or UQ_ALGORITHM == "pce":
        if READ_FROM_FILE_SG_QUADRATURE == True:
            sgq_level = 7  # 6, 10
            # path_to_file = pathlib.Path("/dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights") 
            path_to_file = pathlib.Path("/work/ga45met/mnt/linux_cluster_2/sparse_grid_nodes_weights")  # TODO Change this path accordingly
            parameters_file = path_to_file / f"KPU_d{dim}_l{sgq_level}.asc"
            nodes_and_weights_array = np.loadtxt(parameters_file, delimiter=',')
            nodes_read_from_file = nodes_and_weights_array[:, :dim].T  # final size should be dim x numSamples
            weights_read_from_file = nodes_and_weights_array[:, dim]
            # transform nodes and weight you have read from the file sucha that they correspond to samples from joint_dist_standard
            distsOfNodesFromFile = []
            for _ in range(dim):
                    distsOfNodesFromFile.append(cp.Uniform())
            jointDistOfNodesFromFile = cp.J(*distsOfNodesFromFile)
            nodes = utility.transformation_of_parameters(nodes_read_from_file, jointDistOfNodesFromFile, joint_dist_standard)
            weights = weights_read_from_file
            total_number_model_runs = total_number_of_nodes = nodes.shape[1]
            print(f"Total number of points in {dim}D space: {total_number_of_nodes}")
        else:
            numCollocationPointsPerDim = q = 8
            total_number_of_nodes = np.power(numCollocationPointsPerDim+1,dim)
            print(f"Number of quadrature points in 1D: {numCollocationPointsPerDim+1}")
            print(f"Total number of points in {dim}D space: {total_number_of_nodes}")
            rule = 'g'
            growth = False
            sparse = False
            nodes, weights = cp.generate_quadrature(
                numCollocationPointsPerDim, joint_dist_standard, rule=rule, growth=growth, sparse=sparse)

    else:
        raise NotImplementedError(f"Sorry, the UQ_ALGORITHM {UQ_ALGORITHM} is not supported yet")

    # =========================================================
    # Generate parameters for forcing the model
    # =========================================================

    parameters = utility.transformation_of_parameters(nodes, joint_dist_standard, joint_dist)
    # print(f"DEBUGGING parameters.shape {parameters.shape}")

    # list_unique_index_model_run_list = list(range(0, len(list_parameter_value_particles))) 
    list_unique_index_model_run_list = list(range(0, parameters.shape[1])) 
    # assert numEvaluations == len(list_unique_index_model_run_list)  # this holds for MC

    # Create list-of-dictionaries for the parameter values instead of only matrices of values stored in parameters
    list_parameter_value_particles = parameters.T  # Should be of size (numSamplesxdim)
    # print(f"DEBUGGING list_parameter_value_particles.shape {list_parameter_value_particles.shape}")
    list_of_dict_parameter_value_particles = []
    for parameter_particle in list_parameter_value_particles:
        list_of_dict_parameter_value_particles.append(dict(zip(param_names, parameter_particle)))
    list_parameter_value_particles = list_of_dict_parameter_value_particles

    # =========================================================
    # Running the model
    # =========================================================

    model_runs = np.empty((parameters.shape[1], len(t)))
    list_of_single_df = []
    def process_particles_concurrently(particles_to_process):
        with (multiprocessing.Pool(processes=num_processes) as pool):
            for index_run, y_t_model, parameter_value in \
                    pool.starmap(run_model_single_parameter_node, \
                                    [(model, particle[0], particle[1]) \
                                    for particle in particles_to_process]):
                yield index_run, y_t_model, parameter_value
    for index_run, y_t_model, parameter_value in process_particles_concurrently(\
    zip(parameters.T, list_unique_index_model_run_list)):
        model_runs[index_run] = y_t_model
        df_temp = pd.DataFrame(y_t_model, columns=[QOI_COLUMN_NAME])
        df_temp[TIME_COLUMN_NAME] = t  # list_of_dates_of_interest
        df_temp[index_column_name] = index_run
        tuple_column = [tuple(parameter_value)] * len(df_temp)
        df_temp['Parameters'] = tuple_column
        list_of_single_df.append(df_temp)
    
    df_simulation_result = pd.concat(
        list_of_single_df, ignore_index=True, sort=False, axis=0)
    
    # Shorter code
    # model_runs, df_simulation_result = utility.running_model_in_parallel_and_generating_df(
    #     model, run_model_single_parameter_node, t, parameters, list_unique_index_model_run_list, num_processes, TIME_COLUMN_NAME, INDEX_COLUMN_NAME)
    
    print(f"DEBUGGING - model_runs - {model_runs}")
    print(f"DEBUGGING - df_simulation_result - {df_simulation_result}")
    print(f"DEBUGGING - df_simulation_result.columns - {df_simulation_result.columns}")

    N = total_number_model_runs = model_runs.shape[0]
    N_quad = model_runs.shape[1]

    # =========================================================
    # Collecting all the model runs, conducting postprocessing step and producing different statistics
    # =========================================================

    # =========================================================
    # 1.2 MC Evaluation of the model
    # =========================================================
    result_dict_statistics  = None
    if UQ_ALGORITHM == "mc":
        compute_Sobol_t=True
        if PARALLEL_COMPUTING:
            result_dict_statistics = utility.computing_mc_statistics_parallel_in_time(
                num_processes, df_simulation_result, numEvaluations, single_qoi_column=QOI_COLUMN_NAME,
                compute_Sobol_t=compute_Sobol_t,samples=parameters.T)
        else:
            result_dict_statistics = utility.computing_mc_statistics(
                df_simulation_result, numEvaluations, single_qoi_column=QOI_COLUMN_NAME,
                compute_Sobol_t=compute_Sobol_t,samples=parameters.T)

    # =========================================================
    # 2.2 Computing polynomial basis and building gPCE surrogate over time
    # =========================================================

    elif UQ_ALGORITHM == "kl" or UQ_ALGORITHM == "pce":
        import scipy.special
        order = p = 3 #3 # 3
        c_number = scipy.special.binom(dim+order, dim)
        print(f"Max order of polynomial: {order}")
        print(f"Total number of expansion coefficients in {dim}D space: {int(c_number)}")
        print(f"Total number of time-stamps: {len(t)}")
        # computing polynomial basis - normalized...
        poly_rule = 'three_terms_recurrence'  # 'three_terms_recurrence', 'gram_schmidt', 'cholesky'
        poly_normed = True
        polynomial_expansion, norms = cp.generate_expansion(
            order, joint_dist_standard, rule=poly_rule, normed=poly_normed, retall=True)
        # polynomial_expansion, norms = utility.generate_polynomial_expansion(
        #     joint_dist_standard, order=order, rule=poly_rule, poly_normed=poly_normed)
        print(f"DEBUGGING - polynomial_expansion - {polynomial_expansion}")
        print(f"DEBUGGING - norms - {norms}")

        if UQ_ALGORITHM == "pce":
            regression = False
            compute_only_gpce = False
            compute_Sobol_t=True
            compute_Sobol_m=True
            compute_Sobol_m2=False
            if PARALLEL_COMPUTING:
                result_dict_statistics = utility.computing_gpce_statistics_parallel_in_time(
                    num_processes, df_simulation_result, polynomial_expansion, 
                    nodes, weights, joint_dist_standard, single_qoi_column=QOI_COLUMN_NAME,
                    regression=regression, store_gpce_surrogate_in_stat_dict=True, save_gpce_surrogate=False,
                    compute_only_gpce=compute_only_gpce,
                    compute_Sobol_t=compute_Sobol_t, compute_Sobol_m=compute_Sobol_m, compute_Sobol_m2=compute_Sobol_m2)
            else:
                result_dict_statistics = utility.computing_gpce_statistics(
                    df_simulation_result, polynomial_expansion, 
                    nodes, weights, joint_dist_standard, single_qoi_column=QOI_COLUMN_NAME,
                    regression=regression, store_gpce_surrogate_in_stat_dict=True, save_gpce_surrogate=False,
                    compute_only_gpce=compute_only_gpce,
                    compute_Sobol_t=compute_Sobol_t, compute_Sobol_m=compute_Sobol_m, compute_Sobol_m2=compute_Sobol_m2
                    )
        # elif UQ_ALGORITHM == "kl":
        #     result_dict_statistics = None
        #     # TODO compute mean and centered model output
        #     single_qoi_column_centered = single_qoi_column + "_centered"

        #     grouped = df_simulation_result.groupby([time_column_name,])
        #     groups = grouped.groups
        #     for key, val_indices in groups.items():
        #         qoi_values = df_simulation_result.loc[val_indices.values][single_qoi_column].values
        #         # compute mean
        #         mean = np.dot(qoi_values, weights)
        #         df_simulation_result.loc[val_indices, single_qoi_column_centered] = df_simulation_result.loc[val_indices, single_qoi_column] - mean

    else:
        raise NotImplementedError(f"Sorry, the UQ_ALGORITHM {UQ_ALGORITHM} is not supported yet")

    # =========================================================
    # Crete a DataFrame storing a Statistics Data
    # =========================================================
    if result_dict_statistics is not None:
        df_stat = utility.statistics_result_dict_to_df(result_dict_statistics)
    else:
        df_stat = None
    # initial column 'Sobol_t' would contain tuple of all the T.S.S.I over all the parameters
    
    if df_stat is not None:
        SOBOL_T_COMPUTED = False
        SOBOL_M_COMPUTED = False
        if 'Sobol_t' in df_stat.columns:
            SOBOL_T_COMPUTED = True
            df_unpacked = df_stat['Sobol_t'].apply(pd.Series)
            df_unpacked.columns = [f'Sobol_t_{param_names[i]}' for i in range(df_unpacked.shape[1])]
            df_stat = df_stat.drop('Sobol_t', axis=1).join(df_unpacked)
        if 'Sobol_m' in df_stat.columns:
            SOBOL_M_COMPUTED = True
            df_unpacked = df_stat['Sobol_m'].apply(pd.Series)
            df_unpacked.columns = [f'Sobol_m_{param_names[i]}' for i in range(df_unpacked.shape[1])]
            df_stat = df_stat.drop('Sobol_m', axis=1).join(df_unpacked)

        if 'E' in df_stat.columns and 'StdDev' in df_stat.columns and 'E_plus_std' not in df_stat.columns:
            df_stat['E_plus_std'] = df_stat['E'] + df_stat['StdDev']
        if 'E' in df_stat.columns and 'StdDev' in df_stat.columns and 'E_minus_std' not in df_stat.columns:
            df_stat['E_minus_std'] = df_stat['E'] - df_stat['StdDev']
        # df_stat['E_minus_std'] = df_stat['E_minus_std'].apply(lambda x: max(0, x))

        print(f"DEBUGGING - df_stat{df_stat}")
        print(f"DEBUGGING - df_stat.columns - {df_stat.columns}")

        # =========================================================
        # Plotting
        # =========================================================

        # if PLOT_ALL_THE_RUNS:
        #     # plotting all the realizations...
        #     fig = px.line(
        #         df_simulation_result,
        #         x=time_column_name, 
        #         y=single_qoi_column, # single_qoi_column_centered
        #         color=index_column_name,
        #         line_group=index_column_name, hover_name="Parameters",
        #         labels={time_column_name: "time t",
        #                 single_qoi_column: "displacement y(t)"
        #             },
        #         title="Mode Outputs (random)"
        #     )
        #     fig.update_layout(showlegend=False)
        #     fig.show()
        #     fileName = "all_runs.html"
        #     fileName = directory_for_saving_plots + fileName
        #     pyo.plot(fig, filename=fileName)

        # plotting mean +- STD
        fig = go.Figure()
            # Plotting MC results
        if PLOT_ALL_THE_RUNS:
            # Add forcing data streamflow
            lines = [
                go.Scatter(
                    x=list_of_dates_of_interest,
                    y=single_row,
                    showlegend=False,
                    # legendgroup=colours[i],
                    mode="lines",
                    line=dict(
                        color='LightSkyBlue'),
                    opacity=0.1
                )
                for single_row in model_runs
            ]
            for trace in lines:
                fig.add_trace(trace)
        if 'E_minus_std' in df_stat.columns:
            fig.add_trace(go.Scatter(x=df_stat.index, y=df_stat["E_minus_std"], line_color='rgba(188, 189, 34, 0.1)', opacity=0.1, showlegend=False))
        if 'E_plus_std' in df_stat.columns:
            fig.add_trace(go.Scatter(x=df_stat.index, y=df_stat["E_plus_std"], line_color='rgba(188, 189, 34, 0.1)', fillcolor='rgb(188, 189, 34)', fill='tonexty', opacity=0.1, showlegend=False))
        if 'E' in df_stat.columns:
            fig.add_trace(go.Scatter(x=df_stat.index, y=df_stat["E"], line_color='rgba(255, 0, 0, 1)', name='Mean Prediction'))
        fig.update_traces(mode='lines')
        # fig.update_layout(
        #     title=f"Simulation Results - {UQ_ALGORITHM} - {MODEL}",
        #     xaxis_title="time t",
        #     yaxis_title="displacement y(t)",)
        extend_plot(fig, list_of_dates_of_interest, model, total_number_model_runs, plot_forcing_data=PLOT_FORCING_DATA)
        fig.update_layout(
            title=f"Simulation Results - {UQ_ALGORITHM} - {MODEL} - #runs {total_number_model_runs}",
            xaxis_title="time t",
            yaxis_title="displacement y(t)",)
        fig.show()
        fileName = "statistics_plot.html"
        fileName = directory_for_saving_plots + fileName        
        pyo.plot(fig, filename=fileName)

        if SOBOL_T_COMPUTED:        
            si_columns = [f"Sobol_t_{param_name}" for param_name in param_names]
            fig = px.imshow(df_stat[si_columns].T, labels=dict(y='Parameters', x='Dates'))
            fig.show()
            fileName = "sobol_t_heatmap.html"
            fileName = directory_for_saving_plots + fileName
            pyo.plot(fig, filename=fileName)

            fig = go.Figure()
            for param_name in param_names:
                fig.add_trace(go.Scatter(x=df_stat.index, y=df_stat[f"Sobol_t_{param_name}"], name=param_name))
            fig.update_traces(mode='lines')
            fig.update_layout(
                title="Sobol Total S. I.",
                xaxis_title="time t",
                yaxis_title="sobol total indices",)
            fig.show()
            fileName = "sobol_t.html"
            fileName = directory_for_saving_plots + fileName
            pyo.plot(fig, filename=fileName)

        if SOBOL_M_COMPUTED:
            si_columns = [f"Sobol_m_{param_name}" for param_name in param_names]
            fig = px.imshow(df_stat[si_columns].T, labels=dict(y='Parameters', x='Dates'))
            fig.show()
            fileName = "sobol_m.html"
            fileName = directory_for_saving_plots + fileName
            pyo.plot(fig, filename=fileName)

            fig = go.Figure()
            for param_name in param_names:
                fig.add_trace(go.Scatter(x=df_stat.index, y=df_stat[f"Sobol_m_{param_name}"], name=param_name))
            fig.update_traces(mode='lines')
            fig.update_layout(
                title="Sobol First Order S. I.",
                xaxis_title="time t",
                yaxis_title="sobol first order indices",)
            fig.show()
            fileName = "sobol_m_heatmap.html"
            fileName = directory_for_saving_plots + fileName
            pyo.plot(fig, filename=fileName)

    # =========================================================
    # Algorithm 2 - Time-wise gPCE Surrogate, time-wise Sobol Indices and Generalized Sobol Indices
    # =========================================================
        
    if COMPUTE_GENERALIZED_SOBOL_INDICES and UQ_ALGORITHM == "pce":
        fileName = "generalized_sobol_t.txt"
        fileName = directory_for_saving_plots + fileName
        
        SOBOL_T_COMPUTED = False  # TODO remove
        if SOBOL_T_COMPUTED: 
            df_stat['Var_temp'] = df_stat['Var']*weights_time
            for param_name in param_names:       
                df_stat[f'Sobol_t_{param_name}_temp'] = df_stat[f"Sobol_t_{param_name}"]*df_stat['Var']*weights_time
                s_tot_generalized = df_stat[f'Sobol_t_{param_name}_temp'].sum()/df_stat['Var_temp'].sum()
                print(f"Generalized Total Sobol Index for {param_name} is {s_tot_generalized}")
                with open(fileName, 'a') as file:
                    # Write each variable to the file followed by a newline character
                    file.write(f'{param_name}: {s_tot_generalized}\n')

        else:
            # TODO Important aspect here is if polynomial_expansion is normalized or not
            computing_generalized_sobol_total_indices_from_poly_expan(
                result_dict_statistics, polynomial_expansion, weights_time, param_names, fileName)

    # =========================================================
    # Algorithm 3 - Note: weights are in stohastic space, and weights_time are in time-domian
    # =========================================================
    if UQ_ALGORITHM == "kl":
        # 3.1 Creating a matrix with centered_outputs storing centerd model outputs over time and parameters (quadrature points)
        centered_outputs = center_outputs(N, N_quad, df_simulation_result, weights, single_qoi_column, index_column_name, algorithm="quadrature")
        
        # 3.2 Computation of the covariance matrix
        covariance_matrix = compute_covariance_matrix_in_time(N_quad, centered_outputs, weights, algorithm="quadrature")

        # Plotting the covariance matrix
        plot_covariance_matrix(covariance_matrix, directory_for_saving_plots)

        # 3.3 Solve Discretized (generalized) Eigenvalue Problem
        eigenvalues, eigenvectors = solve_eigenvalue_problem(covariance_matrix, weights_time)
        print("Eigenvalues:\n", eigenvalues)
        print("Eigenvectors:\n", eigenvectors)
        
        print(f"DEBUGGING eigenvalues.shape - {eigenvalues.shape}")
        print(f"DEBUGGING eigenvectors.shape - {eigenvectors.shape}")
        
        # Plotting the eigenvalues
        plot_eigenvalues(eigenvalues, directory_for_saving_plots)
    
        # 3.4 Approximating the KL Expansion
        Var_kl_approx = np.sum(eigenvalues)
        N_kl =  60 # [2, 4, 6, 8, 10]
        weights_time = np.asfarray(weights_time)
        print(f"DEBUGGING - N {N}")
        f_kl_eval_at_params = setup_kl_expansion_matrix(eigenvalues, N_kl, N, N_quad, weights_time, centered_outputs, eigenvectors)
        print(f"DEBUGGING - f_kl_eval_at_params.shape {f_kl_eval_at_params.shape}")

        # 3.5 PCE of the KL Expansion
        f_kl_surrogate_dict, f_kl_surrogate_coefficients = pce_of_kl_expansion(N_kl, polynomial_expansion, nodes, weights, f_kl_eval_at_params)
        
        # 3.6 Generalized Sobol Indices
        if COMPUTE_GENERALIZED_SOBOL_INDICES:
            fileName = "generalized_sobol_t_kl_and_pce.txt"
            fileName = directory_for_saving_plots + fileName
            computing_generalized_sobol_total_indices_from_kl_expan(
                f_kl_surrogate_coefficients, polynomial_expansion, weights_time, param_names, fileName, total_variance=Var_kl_approx)
    
    end_time_model_simulations = time.time()
    time_model_simulations = end_time_model_simulations - start_time_model_simulations
    with open(time_infoFileName, 'a') as fp:
        fp.write(f'time_model_simulations: {time_model_simulations}\n')
    

    # =========================================================
    # Re-evaluationg the surrogate model
    # =========================================================

    if REEVALUATE_SURROGET_MODEL and UQ_ALGORITHM == "kl" or UQ_ALGORITHM == "pce":
        print(f"RE-evaluationg a surrogate model computed based on {UQ_ALGORITHM} just started!")
        start_time_surrogate_model_reevaluation = time.time()
        # RE-evaulate the surroget model
        numSamples = 10**3
        rule = 'random'
        nodes_to_eval_kl_surrogate = joint_dist_standard.sample(size=numSamples, rule=rule).round(4)
        surrogate_eval = np.zeros((len(t),nodes_to_eval_kl_surrogate.shape[1]))
        if UQ_ALGORITHM == "kl":
            mean_dict = utility.compute_mean_from_df_simulation_result(df_simulation_result, algorithm="quadrature", weights=weights, single_qoi_column=single_qoi_column, time_column_name=time_column_name)  ## df_stat['E'].to_numpy()
            df_stat = pd.DataFrame.from_dict(mean_dict, orient='index', columns=["E",])
            df_stat.index.name = time_column_name
            for m in range(len(t)):
                surrogate_eval[m,:] = mean_dict[t[m]] #mean_vector[m] df_stat.loc(t[m])
                for i in range(N_kl):
                    surrogate_eval[m,:] += f_kl_surrogate_dict[i]["gPCE"](*nodes_to_eval_kl_surrogate)*eigenvectors[m,i]
        elif UQ_ALGORITHM == "pce":
            def process_dates_concurrently(dates, df_stat, nodes):
                with multiprocessing.Pool(processes=num_processes) as pool:
                    for single_time_stamp, result in pool.starmap(evaluate_gPCE_model_single_date_from_df_stat, \
                                            [(single_time_stamp, df_stat, nodes) for single_time_stamp in dates]): 
                        yield single_time_stamp, result
            date_counter = 0
            for date, gPCE_result in process_dates_concurrently(t, df_stat, nodes_to_eval_kl_surrogate): # or list_of_dates_of_interest
                # surrogate_eval_dict[date]  = gPCE_result
                surrogate_eval[date_counter,:] = gPCE_result
                date_counter += 1
            # or, alternatively, without parallelization
            # for m in range(len(t)):
            #     surrogate_eval[m,:] = df_stat.iloc[t[m],"gPCE"](*nodes_to_eval_kl_surrogate)
        else:
            print(f"Sorry, the UQ_ALGORITHM {UQ_ALGORITHM} does not produce a surrogate model, therefore, REEVALUATE_SURROGET_MODEL should be set to False")
        
        end_time_surrogate_model_reevaluation = time.time()
        time_surrogate_model_reevaluation = end_time_surrogate_model_reevaluation - start_time_surrogate_model_reevaluation
        with open(time_infoFileName, 'a') as fp:
            fp.write(f'time_surrogate_model_reevaluation: {time_surrogate_model_reevaluation}\n')

        fig = go.Figure()
        lines = [
            go.Scatter(
                x=t,
                y=single_row,
                showlegend=False,
                # legendgroup=colours[i],
                mode="lines",
                line=dict(
                    color='LightSkyBlue'),
                opacity=0.1
            )
            for single_row in surrogate_eval.T
        ]
        for trace in lines:
            fig.add_trace(trace)
        if 'E' in df_stat.columns:
            fig.add_trace(go.Scatter(x=df_stat.index, y=df_stat["E"], line_color='rgba(255,0,0,1)', showlegend=True, name="Mean"))
        if UQ_ALGORITHM == "pce":
            if 'E_minus_std' in df_stat.columns:
                fig.add_trace(go.Scatter(x=df_stat.index, y=df_stat["E_minus_std"], line_color='rgba(255,255,255,0)', showlegend=False))
            if 'E_plus_std' in df_stat.columns:
                fig.add_trace(go.Scatter(x=df_stat.index, y=df_stat["E_plus_std"], line_color='rgba(255,255,255,0)', fillcolor='rgba(0,176,246,0.2)', fill='tonexty', showlegend=False))
        fig.update_traces(mode='lines')
        # fig.update_layout(
        #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        #     title="Surrogate model re-evaluation - {UQ_ALGORITHM}",
        #     xaxis_title="time t",
        #     yaxis_title="displacement y(t)",)
        extend_plot(fig, list_of_dates_of_interest, model, total_number_model_runs, plot_forcing_data=PLOT_FORCING_DATA)
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=f"Surrogate model re-evaluation - {UQ_ALGORITHM} - {MODEL} - #runs {total_number_model_runs}",
            xaxis_title="time t",
            yaxis_title="displacement y(t)",)
        fig.show()
        fileName = "surrogate_model_reevaluated.html"
        fileName = directory_for_saving_plots + fileName
        pyo.plot(fig, filename=fileName)


if __name__ == "__main__":
    main_routine()
