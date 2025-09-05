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



def metrics(surrogate, list_of_dates_of_interest, standar_parameter_samples_matrix, final_predicted_streamflow, final_observed_streamflow):
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

    mean_pce_output = np.mean(surrogate_outputs)
    pce_output = mean_pce_output

    # Get real model output and observation for the last date
    real_model_output = final_predicted_streamflow[last_date]
    real_observation = final_observed_streamflow[last_date]

    print(f"PCE-output: {pce_output}")
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