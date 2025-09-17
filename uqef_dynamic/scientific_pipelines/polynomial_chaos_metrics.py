# File to evaluate the surrogate model and produce comparisons / metrics

import numpy as np
import chaospy 
import transport_pce_pipeline
import pathlib
import matplotlib.pyplot as plt
import os
import time

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib.pyplot as plt


from uqef_dynamic.utils import utility
from uqef_dynamic.utils import transport_map
from uqef_dynamic.models.hbv_sask import hbvsask_utility as hbv
from uqef_dynamic.models.hbv_sask import HBVSASKModel as hbvmodel

working_dir_name=f"trial_single_run_hbvsaskmodel_7d_filtering_III"
hbv_model_data_path = pathlib.Path("/home/christoph/projects/thesis_code/HBV-SASK-data")
# change 6D to 10D
configuration_file = pathlib.Path('/home/christoph/projects/thesis_code/HBV-SASK-py-tool/configurations/configuration_hbv_6D.json')
# configuration_file = pathlib.Path('/home/christoph/projects/thesis_code/UQEF-Dynamic/data/configurations/configuration_hbv_10D.json')
inputModelDir = hbv_model_data_path
basin = "Oldman_Basin"  # 'Banff_Basin'
workingDir = hbv_model_data_path / basin / "model_runs" / working_dir_name
directory_for_saving_plots = workingDir




def setup_HBV():
    working_dir_name=f"trial_single_run_hbvsaskmodel_7d_filtering_III"
    hbv_model_data_path = pathlib.Path("/home/christoph/projects/thesis_code/HBV-SASK-data")
    # change 6D to 10D
    configuration_file = pathlib.Path('/home/christoph/projects/thesis_code/HBV-SASK-py-tool/configurations/configuration_hbv_6D.json')
    # configuration_file = pathlib.Path('/home/christoph/projects/thesis_code/UQEF-Dynamic/data/configurations/configuration_hbv_10D.json')
    inputModelDir = hbv_model_data_path
    basin = "Oldman_Basin"  # 'Banff_Basin'
    workingDir = hbv_model_data_path / basin / "model_runs" / working_dir_name
    directory_for_saving_plots = workingDir
    if not str(directory_for_saving_plots).endswith("/"):
        directory_for_saving_plots = str(directory_for_saving_plots) + "/"

    # Creating Model Object
    writing_results_to_a_file = False
    plotting = False
    createNewFolder = False # create a separate folder to save results for each model run
    hbvsaskModelObject = hbvmodel.HBVSASKModel(
        configurationObject=configuration_file,
        inputModelDir=inputModelDir,
        workingDir=workingDir,
        basin=basin,
        writing_results_to_a_file=writing_results_to_a_file,
        plotting=plotting
    )
    
    return hbvsaskModelObject



hbvsaskModelObject = setup_HBV()

def evaluate(inverted_parameters, mean_state_values):
        """Evaluates the hydrological model for given parameters in original (target / exponential) form"""
        
        param_names = ['TT', 'C0', 'beta', 'ETF', 'FC', 'FRAC', 'K2']
        param_dict = {name: float(val) for name, val in zip(param_names, inverted_parameters)}

        
        # unique_index_model_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict
        _, y_t_model, _, _, _ = transport_pce_pipeline.run_model_single_time_stamp_single_particle(
            hbvsaskModelObject=hbvsaskModelObject,
            date_of_interest=hbvsaskModelObject.end_date,
            parameter_value_dict=param_dict, # ! dictionary
            state_values_dict=mean_state_values
            )

        return y_t_model
    
    
def inverse(standard_parameters, transport_map, scaler):
        """Inverse parameter distribution from reference (SNV) back to target (original / exponential)"""
        
        
        # Inverse transport map approximation
        X_reconstruct = transport_map.Inverse(np.empty((0, standard_parameters.shape[1])), standard_parameters)
    
        # Inverse scaling
        X_reconstruct = scaler.inverse_transform(X_reconstruct.T).T

        # Inverse logarithm
        for i in range(X_reconstruct.shape[0]):
            X_reconstruct[i, :] = np.exp(X_reconstruct[i, :])
        
        
        return X_reconstruct



def metrics(surrogate, standar_parameter_samples_matrix, parameter_samples_matrix, mean_state_values, transport_map, scaler):
   
    start_time = time.time()
    surrogate_outputs = chaospy.call(surrogate, standar_parameter_samples_matrix)
    end_time = time.time()
    surrogate_total_time = end_time - start_time
    surrogate_avg_time = surrogate_total_time / standar_parameter_samples_matrix.shape[1]

    print(f"SURROGATE PERFORMANCE: {surrogate_total_time:.4f} seconds (based on {standar_parameter_samples_matrix.shape[1]} samples) "
          f"(~{surrogate_avg_time*1000:.6f} ms per sample)")



    model_outputs = []
    start_time = time.time()
    # for i in range(standar_parameter_samples_matrix.shape[0]):
    print("PARAMETER SAMPLE MATRIX SHAPE")
    param_HBV_matrix = parameter_samples_matrix.T
    print(parameter_samples_matrix.shape)
    
    # for i in range(parameter_samples_matrix.shape[0]):
    #     param_vec = parameter_samples_matrix[i, :]
    #     model_output = evaluate(param_vec, mean_state_values)
    #     model_outputs.append(model_output)
     
    for i in range(param_HBV_matrix.shape[1]):
        params = standar_parameter_samples_matrix[:, i].reshape(-1, 1)
        params_inv = inverse(params, transport_map, scaler)
        model_output = evaluate(params_inv, mean_state_values)
        model_outputs.append(model_output)
    
      
    end_time = time.time()
    model_total_time = end_time - start_time
    model_avg_time = model_total_time / standar_parameter_samples_matrix.shape[1]

    print(f"MODEL PERFORMANCE: {model_total_time:.4f} seconds (based on {standar_parameter_samples_matrix.shape[1]} samples) "
          f"(~{model_avg_time*1000:.6f} ms per sample)")

    
    
    mean_pce_output = np.mean(surrogate_outputs)
    pce_output = mean_pce_output
    
    
    print(f"PCE-output: {pce_output}")
    mean_model = np.mean(model_outputs)
    print(f"MODEL-output: {mean_model}")
   
    
    
    return pce_output, mean_model, surrogate_total_time, surrogate_avg_time, model_total_time, model_avg_time