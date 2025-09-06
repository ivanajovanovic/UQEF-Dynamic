# File to evaluate the surrogate model and produce comparisons / metrics

import numpy as np
import chaospy 
import transport_pce_pipeline
import pathlib
import matplotlib.pyplot as plt
import os

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

def evaluate(inverted_parameters, mean_state_values_dict):
        """Evaluates the hydrological model for given parameters in original (target / exponential) form"""
        
        
        # print()
        # print("== INVERTED PARAMETERS ==")
        # print(inverted_parameters)
        # print()
        
        # TODO: check argument values
        # unique_index_model_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict
        _, y_t_model, _, _, _ = transport_pce_pipeline.run_model_single_time_stamp_single_particle(
            hbvsaskModelObject=hbvsaskModelObject,
            date_of_interest=hbvsaskModelObject.end_date,
            parameter_value_dict=inverted_parameters, # ! dictionary
            state_values_dict=mean_state_values_dict
            )
        # print("original_parameters shape:", np.shape(original_parameters))
        # print("original_parameters:", original_parameters)
    
        # result = hbvsaskModelObject.run_model_single_time_stamp(hbvsaskModelObject.end_date, parameters=original_parameters, raise_exception_on_model_break=True)
        
        return y_t_model
    
    
def inverse(standard_parameters, transport_map, scaler):
        """Inverse parameter distribution from reference (SNV) back to target (original / exponential)"""
        
        # print()
        # print("== SAMPLED PARAMETERS ==")
        # print(standard_parameters)
        # print()
        
        # Inverse transport map approximation
        X_reconstruct = transport_map.Inverse(np.empty((0, standard_parameters.shape[1])), standard_parameters)
        # print("After transport_map.Inverse: any nan?", np.any(np.isnan(X_reconstruct)), "any inf?", np.any(np.isinf(X_reconstruct)))
        # print("After inverse map:", X_reconstruct)
    
        # Inverse scaling
        X_reconstruct = scaler.inverse_transform(X_reconstruct.T).T
        # print("After scaler.inverse_transform: any nan?", np.any(np.isnan(X_reconstruct)), "any inf?", np.any(np.isinf(X_reconstruct)))
        # print("After inverse scaling:", X_reconstruct)
    
        # Inverse logarithm
        for i in range(X_reconstruct.shape[0]):
            X_reconstruct[i, :] = np.exp(X_reconstruct[i, :])
        
        # print("After exponentiation:", X_reconstruct)

        # print("After np.exp: any nan?", np.any(np.isnan(X_reconstruct)), "any inf?", np.any(np.isinf(X_reconstruct)))

        
        return X_reconstruct



def metrics(surrogate, list_of_dates_of_interest, standar_parameter_samples_matrix, final_predicted_streamflow, final_observed_streamflow, mean_state_values, transport_map, scaler):
    last_date = list_of_dates_of_interest[-1]

    # Get the mean of standard parameters
    # mean_standard_params = np.mean(standar_parameter_samples_matrix, axis=0).reshape(-1, 1)   
    
    # print(mean_standard_params)
    # print(mean_standard_params.shape)
    # pce_output = float(chaospy.call(surrogate, mean_standard_params))

    # Evaluate PCE surrogate
    surrogate_outputs = []
    for params in standar_parameter_samples_matrix:
        params_col = params.reshape(-1, 1)
        pce_output = float(chaospy.call(surrogate, params_col))
        surrogate_outputs.append(pce_output)
        
        
    
    model_outputs = []
    for i in range(standar_parameter_samples_matrix.shape[1]):
        params = standar_parameter_samples_matrix[:, i].reshape(-1, 1)
        params_inv = inverse(params, transport_map, scaler)
        params_col = params_inv.reshape(-1, 1)
        model_output = evaluate(params_col, mean_state_values)
        model_outputs.append(model_output)

    mean_pce_output = np.mean(surrogate_outputs)
    pce_output = mean_pce_output

    # Get real model output and observation for the last date
    real_model_output = final_predicted_streamflow[last_date]
    real_observation = final_observed_streamflow[last_date]

    print(f"PCE-output: {pce_output}")
    
    mean_model = np.mean(model_outputs)
    print(f"MODEL-output: {mean_model}")
    # Plot
    categories = ['Observation', 'Real Model', 'PCE Surrogate']
    values = [real_observation, real_model_output, pce_output]

    fig = go.Figure(data=[
        go.Scatter(
            x=categories,
            y=values,
            mode='lines+markers',
            marker=dict(size=12, color='black'),
            line=dict(color='black')
        )
    ])
    fig.update_layout(
        title=f"Comparison for {last_date.strftime('%Y-%m-%d')}",
        yaxis_title="Streamflow [m³/s]",
        xaxis_title="",
        template="simple_white"
    )

    save_path = os.path.join(str(directory_for_saving_plots), f"comparison_last_date_{last_date.strftime('%Y%m%d')}.html")
    fig.write_html(save_path, auto_open=True)