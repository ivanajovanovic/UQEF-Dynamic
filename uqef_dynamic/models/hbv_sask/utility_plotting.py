"""
Set of functions for plotting results obtained with UQEF-Dynamic; tailord for the HBV-Sask model;
The same code as in the jupyternotebook: 
@author: Ivana Jovanovic Buha
"""
import os
import dill
import numpy as np
import sys
import pathlib
import pandas as pd
import pickle
import time
from collections import defaultdict
import sys

# importing modules/libs for plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# import plotly.offline as pyo
# # Set notebook mode to work in offline
# pyo.init_notebook_mode()

import matplotlib.pyplot as plt

pd.options.plotting.backend = "plotly"

sys.path.insert(1, '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic')
from uqef_dynamic.utils import utility
from uqef_dynamic.utils import uqef_dynamic_utils
from uqef_dynamic.models.hbv_sask import hbvsask_utility as hbv
from uqef_dynamic.models.hbv_sask import HBVSASKModel as hbvmodel
from uqef_dynamic.models.hbv_sask import HBVSASKStatistics as HBVSASKStatistics
from uqef_dynamic.utils import create_stat_object


def uq_plotting(
    df_statistics_and_measured, directory_for_saving_plots=None, dict_what_to_plot=None, list_of_qois=None, time_column_name=utility.TIME_COLUMN_NAME, fileName=f"FUQ.pdf"):
    if dict_what_to_plot is None:
        dict_what_to_plot = {
            "E_minus_2std": False, "E_plus_2std": False,
            "E_minus_std": False, "E_plus_std": False, 
            "P10": True, "P90": True,
            "StdDev": True, "Skew": False, "Kurt": False, "Sobol_m": False, "Sobol_m2": False, "Sobol_t": False
        }
    if list_of_qois is None:
        list_of_qois = set(df_statistics_and_measured['qoi'].values)
    timesteps_min = df_statistics_and_measured[time_column_name].min()
    timesteps_max = df_statistics_and_measured[time_column_name].max()
    subplot_titles = ("Temperature [°C]", "Precipitation [mm/day]", )
    n_rows = 2 #3
    for single_qoi in list_of_qois:
        if single_qoi =='Q_cms':
            subplot_titles = subplot_titles + (f"FUQ: QoI - Streamflow[m^3/s] ({single_qoi})",)
        elif single_qoi =='AET':
            subplot_titles = subplot_titles + (f"FUQ: QoI - Actual Evapotranspiration ({single_qoi})",)
        else:
            raise Error()
        n_rows+=1

    print(f"{n_rows}; {subplot_titles}")

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        vertical_spacing=0.04
    )

    # just temporary when plotting forcing and measured data...
    first_qoi = list_of_qois[0]
    df = df_statistics_and_measured.loc[
            df_statistics_and_measured['qoi'] == first_qoi] 
    fig.add_trace(
        go.Scatter(
            x=df['TimeStamp'], y=df['temperature'],
            text=df['temperature'],
            name="Temperature", mode='lines+markers',
            showlegend=False
            # marker_color='blue'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=df['TimeStamp'], y=df['precipitation'],
            text=df['precipitation'],
            name="Precipitation",
            showlegend=False,
            marker_color='red'
            # mode="lines",
            #         line=dict(
            #             color='LightSkyBlue')
        ),
        row=2, col=1
    )
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    legendrank=0
    current_row = starting_row_for_predicted_data = 3 #4
    qoi_idx = 0
    colors = [ '#0072B2', '#E69F00', '#CC79A7', '#009E73', ]
    for single_qoi in list_of_qois:
        df = df_statistics_and_measured.loc[
            df_statistics_and_measured['qoi'] == single_qoi] 
        
        legendrank+=1
        fig.add_trace(
            go.Scatter(
                x=df['TimeStamp'], y=df['E'],
                text=df['E'],
                name=f"Mean predicted {single_qoi}", mode='lines',
                line=dict(color=colors[qoi_idx]),
                # legendrank=legendrank,
            ),
            row=starting_row_for_predicted_data, col=1
        )

        # Hardcoded
        if single_qoi == "Q_cms":
            fig.add_trace(
                go.Scatter(
                    x=df['TimeStamp'], y=df['measured'],
                    name="Observed Streamflow [m^3/s]", mode='lines',
                    line=dict(color='green'),
                    showlegend=True #False
                ),
                row=starting_row_for_predicted_data, col=1
            )

        if qoi_idx == 0:
            showlegend = True
        else:
            showlegend = False

        if dict_what_to_plot.get("E_minus_std", False):
            legendrank+=1
            fig.add_trace(
                go.Scatter(
                    x=df['TimeStamp'], y=df['E_minus_std'],
                    name=f'E_minus_std', #f'{single_qoi}-E_minus_std'
                    text=df['E_minus_std'], mode='lines', line_color='rgba(200, 200, 200, 0.4)', #line_color="grey",
                    # legendrank=legendrank,
                    showlegend=False,
                ),
                row=starting_row_for_predicted_data, col=1
            )
        if dict_what_to_plot.get("E_plus_std", False):
            legendrank+=1
            fig.add_trace(
                go.Scatter(
                    x=df['TimeStamp'], y=df['E_plus_std'],
                    name=f'E+-std', #f'{single_qoi}-E_plus_std'
                    text=df['E_plus_std'], line_color='rgba(200, 200, 200, 0.4)', #line_color="grey",
                    mode='lines', fill='tonexty',
                    # legendrank=legendrank,
                    showlegend=showlegend,
                ),
                row=starting_row_for_predicted_data, col=1
            )
        if dict_what_to_plot.get("E_minus_2std", False):
            legendrank+=1
            fig.add_trace(
                go.Scatter(
                    x=df['TimeStamp'], y=df['E_minus_2std'],
                    name=f'E_minus_2std', #f'{single_qoi}-E_minus_std'
                    text=df['E_minus_std'], mode='lines', line_color='rgba(200, 200, 200, 0.4)', #line_color="grey",
                    # legendrank=legendrank,
                    showlegend=False,
                ),
                row=starting_row_for_predicted_data, col=1
            )
        if dict_what_to_plot.get("E_plus_2std", False):
            legendrank+=1
            fig.add_trace(
                go.Scatter(
                    x=df['TimeStamp'], y=df['E_plus_2std'],
                    name=f'E+-2std', #f'E_plus_2std', #f'{single_qoi}-E_plus_std'
                    text=df['E_plus_std'], line_color='rgba(200, 200, 200, 0.4)', #line_color="grey",
                    mode='lines', fill='tonexty',
                    # legendrank=legendrank,
                    showlegend=showlegend,
                ),
                row=starting_row_for_predicted_data, col=1
            )
        if dict_what_to_plot.get("P10", False):
            legendrank+=1
            fig.add_trace(go.Scatter(x=df['TimeStamp'], y=df["P10"],
                                    name=f'10th Percentile',  #f'{single_qoi}-P10',
                                    line_color='rgba(128,128,128, 0.3)', mode='lines',
                                    # legendrank=legendrank,
                                    showlegend=False,
                                    ),
                        row=starting_row_for_predicted_data, col=1)
        if dict_what_to_plot.get("P90", False):
            legendrank+=1
            fig.add_trace(go.Scatter(x=df['TimeStamp'], y=df["P90"],
                                    name=f'10th-90th Percentile', #f'{single_qoi}-P90',
                                    mode='lines',
                                    line=dict(color='rgba(128,128,128, 0.3)'),fill='tonexty',  # Fill to the next y trace
                                    fillcolor='rgba(128,128,128, 0.3)', #128,128,128("grey")  240,228,66("yellow")
                                    # legendrank=legendrank,
                                    showlegend=showlegend,
                                    ),
                        row=starting_row_for_predicted_data, col=1)
            
        qoi_idx+=1
        current_row+=1
        starting_row_for_predicted_data+=1
    
    fig.update_layout(
        xaxis=dict(
            rangemode='normal',
            range=[timesteps_min, timesteps_max],
            type="date"
        ),
        yaxis=dict(
            rangemode='normal',  # Ensures the range is not padded for markers
            autorange=True       # Auto-range is enabled
        )
    )
    fig.update_layout(height=1100, width=1100)
    fig.update_layout(
            # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            title=f'HBV-SASK Model: FUQ and SA',
            showlegend=True,
            # template="plotly_white",
        )
    fig.update_xaxes(
        tickformat='%b %y',            # Format dates as "Month Day" (e.g., "Jan 01")
        dtick="M2"                     # Set tick interval to 1 day for denser ticks
    )
    if directory_for_saving_plots is not None:
        plot_filename = directory_for_saving_plots / fileName
        fig.update_layout(title=None)
        fig.update_layout(
            margin=dict(
                t=10,  # Top margin
                b=10,  # Bottom margin
                l=20,  # Left margin
                r=20   # Right margin
            )
        )
        fig.write_image(str(plot_filename), format="pdf", width=1100,)  #height=1000, width=1100,
                    

def main(list_workindDir, inputModelDir, 
    set_lower_predictions_to_zero=True,
    set_mean_prediction_to_zero=True,
    correct_sobol_indices=False,
    read_saved_simulations=False,
    read_saved_states=False,
    instantly_save_results_for_each_time_step=False,
    time_column_name=utility.TIME_COLUMN_NAME, resolution="1D"):
    
    # =======================
    # Fatching Data
    # =======================

    statisticsObject_list = []
    df_statistics_and_measured_list = []
    si_t_df_list = []
    si_m_df_list = []
    list_of_qois = []
    list_of_qois_sets = []
    for workingDir in list_workindDir:
        statisticsObject_I, df_statistics_and_measured_I,  si_t_df_I, si_m_df_I, _, _, _= \
        create_stat_object.get_df_statistics_and_df_si_from_saved_files(
            workingDir, inputModelDir, 
            set_lower_predictions_to_zero=set_lower_predictions_to_zero,
            set_mean_prediction_to_zero=set_mean_prediction_to_zero,
            correct_sobol_indices=correct_sobol_indices,
            read_saved_simulations=read_saved_simulations, 
            read_saved_states=read_saved_states, 
            instantly_save_results_for_each_time_step=instantly_save_results_for_each_time_step,
            )
        statisticsObject_list.append(statisticsObject_I)
        df_statistics_and_measured_list.append(df_statistics_and_measured_I)
        si_t_df_list.append(si_t_df_I)
        si_m_df_list.append(si_m_df_I)
        list_of_qois.append(statisticsObject_I.list_qoi_column)
        list_of_qois_sets.append(set(statisticsObject_I.list_qoi_column))
    set_of_qois = set(list_of_qois_sets[0])
    for single_set_of_qois in list_of_qois_sets[1:]:
        set_of_qois &= single_set_of_qois  #set_of_qois.intersection(single_set_of_qois)
    overlap_lost_of_qois = list(set_of_qois)

    df_statistics_and_measured = pd.concat(df_statistics_and_measured_list, ignore_index=True)
    df_statistics_and_measured = df_statistics_and_measured.sort_values(by=time_column_name)
    if si_t_df_list and si_t_df_list is not None:
        si_t_df_list_not_none = [elem for elem in si_t_df_list if elem is not None]
        if si_t_df_list_not_none:
            si_t_df = pd.concat(si_t_df_list_not_none, ignore_index=True)
            si_t_df = si_t_df.sort_values(by=time_column_name)
        else:
            si_t_df = None
    else:
        si_t_df = None
    if si_m_df_list and si_m_df_list is not None:
        si_m_df_list_not_none = [elem for elem in si_m_df_list if elem is not None]
        if si_m_df_list_not_none:
            si_m_df = pd.concat(si_m_df_list_not_none, ignore_index=True)
            si_m_df = si_m_df.sort_values(by=time_column_name)
        else:
            si_m_df = None
    else:
        si_m_df = None

    # =======================
    # Some processing
    # =======================

    timestamp_min = df_statistics_and_measured[time_column_name].min()
    timestamp_max = df_statistics_and_measured[time_column_name].max()
    print(f"timestamp_min={timestamp_min}; timestamp_max={timestamp_max}")

    data_range = pd.date_range(start=timestamp_min, end=timestamp_max, freq=resolution)
    unique_tima_stamps_in_df = df_statistics_and_measured[time_column_name].unique()
    for single_date in data_range:
        if single_date not in unique_tima_stamps_in_df:
            print(f"Missing run id {single_date}")

    if 'StdDev' in df_statistics_and_measured.columns:
        if "E_minus_std" not in df_statistics_and_measured.columns and "E_plus_std" not in df_statistics_and_measured.columns:
            df_statistics_and_measured["E_minus_std"] = df_statistics_and_measured['E'] - df_statistics_and_measured['StdDev']
            df_statistics_and_measured["E_plus_std"] = df_statistics_and_measured['E'] + df_statistics_and_measured['StdDev']
        if "E_minus_2std" not in df_statistics_and_measured.columns and "E_plus_2std" not in df_statistics_and_measured.columns:
            df_statistics_and_measured["E_minus_2std"] = df_statistics_and_measured['E'] - 2*df_statistics_and_measured['StdDev']
            df_statistics_and_measured["E_plus_2std"] = df_statistics_and_measured['E'] + 2*df_statistics_and_measured['StdDev']
    elif 'Var' in df_statistics_and_measured.columns:
        if "E_minus_std" not in df_statistics_and_measured.columns and "E_plus_std" not in df_statistics_and_measured.columns:
            df_statistics_and_measured["E_minus_std"] = df_statistics_and_measured['E'] - np.sqrt(df_statistics_and_measured['Var'])
            df_statistics_and_measured["E_plus_std"] = df_statistics_and_measured['E'] + np.sqrt(df_statistics_and_measured['Var'])
            df_statistics_and_measured['E_minus_std'] = df_statistics_and_measured['E_minus_std'].apply(lambda x: max(0, x))
        if "E_minus_2std" not in df_statistics_and_measured.columns and "E_plus_2std" not in df_statistics_and_measured.columns:
            df_statistics_and_measured["E_minus_2std"] = df_statistics_and_measured['E'] - 2*np.sqrt(df_statistics_and_measured['Var'])
            df_statistics_and_measured["E_plus_2std"] = df_statistics_and_measured['E'] + 2*np.sqrt(df_statistics_and_measured['Var'])

    if set_lower_predictions_to_zero:
        if 'E_minus_std' in df_statistics_and_measured.columns:
            df_statistics_and_measured['E_minus_std'] = df_statistics_and_measured['E_minus_std'].apply(lambda x: max(0, x))
        if 'E_minus_2std' in df_statistics_and_measured.columns:
            df_statistics_and_measured['E_minus_2std'] = df_statistics_and_measured['E_minus_2std'].apply(lambda x: max(0, x))
        if 'P10' in df_statistics_and_measured.columns:
            df_statistics_and_measured['P10'] = df_statistics_and_measured['P10'].apply(lambda x: max(0, x))

    if set_mean_prediction_to_zero:
        df_statistics_and_measured['E'] = df_statistics_and_measured['E'].apply(lambda x: max(0, x))

    # =======================
    # Plotting using uqef_dynamic_utils functions
    # =======================
    dict_what_to_plot = {
        "E_minus_std": True, "E_plus_std": True, 
        "E_minus_2std": False, "E_plus_2std": False,
        "P10": False, "P90": False,
        "StdDev": True, "Skew": False, "Kurt": False, "Sobol_m": False, "Sobol_m2": False, "Sobol_t": False
    }
    for single_qoi in overlap_lost_of_qois:
        df_statistics_and_measured_single_qoi_subset = df_statistics_and_measured.loc[
            df_statistics_and_measured['qoi'] == single_qoi]
        fig = uqef_dynamic_utils.plotting_function_single_qoi_hbv(
            df_statistics_and_measured_single_qoi_subset, 
            single_qoi=single_qoi, 
            qoi=single_qoi, # (?) statisticsObject_I.qoi
            dict_what_to_plot=dict_what_to_plot,
            directory=str(directory_for_saving_plots),
            fileName=f"simulation_big_plot_{single_qoi}_2std.html"
        )
        fig.update_layout(template="plotly_white")
        fig.update_layout(title=None)
        fig.update_layout(
            margin=dict(
                t=10,  # Top margin
                b=10,  # Bottom margin
                l=20,  # Left margin
                r=20   # Right margin
            )
        )
        plot_filename = directory_for_saving_plots / f"simulation_big_plot_{single_qoi}_2std.pdf"
        fig.write_image(str(plot_filename), format="pdf", width=1100,)

        if si_m_df is not None:
            fig = uqef_dynamic_utils.plot_heatmap_si_single_qoi(
                qoi_column=single_qoi, si_df=si_m_df, si_type="Sobol_m")
            fig.update_layout(title_text=f"Sobol First-order SI w.r.t. QoI - {single_qoi}")
            fig.update_layout(title=None)
            fig.update_layout(
                margin=dict(
                    t=10,  # Top margin
                    b=10,  # Bottom margin
                    l=20,  # Left margin
                    r=20   # Right margin
                )
            )
            plot_filename = directory_for_saving_plots / f"sobol_first_heatmap_{single_qoi}.pdf"
            fig.write_image(str(plot_filename), format="pdf", width=1100,)
        if si_t_df is not None:
            fig = uqef_dynamic_utils.plot_heatmap_si_single_qoi(
                qoi_column=single_qoi, si_df=si_t_df, si_type="Sobol_t")
            fig.update_layout(title_text=f"Sobol Total-order SI w.r.t. QoI - {single_qoi}")
            fig.update_layout(title=None)
            fig.update_layout(
                margin=dict(
                    t=10,  # Top margin
                    b=10,  # Bottom margin
                    l=20,  # Left margin
                    r=20   # Right margin
                )
            )
            plot_filename = directory_for_saving_plots / f"sobol_first_heatmap_{single_qoi}.pdf"
            fig.write_image(str(plot_filename), format="pdf", width=1100,)
    # =======================
    dict_what_to_plot = {
        "E_minus_std": True, "E_plus_std": True, 
        "E_minus_2std": False, "E_plus_2std": False,
        "P10": False, "P90": False,
        "StdDev": True, "Skew": False, "Kurt": False, "Sobol_m": False, "Sobol_m2": False, "Sobol_t": False
    }
    uq_plotting(
        df_statistics_and_measured, directory_for_saving_plots=directory_for_saving_plots, 
        dict_what_to_plot=dict_what_to_plot, list_of_qois=overlap_lost_of_qois, time_column_name=utility.TIME_COLUMN_NAME,
        fileName=f"FUQ_Paper_percent.pdf"
        )
    # =======================


if __name__ == '__main__':
    # MC 150 000 2004--2007 LHS NSE>0.2
    workingDir_I = basis_workingDir / 'hbv_uq_mpp3.0054'
    workingDir_II = basis_workingDir / 'hbv_uq_mpp3.0055'
    workingDir_III = basis_workingDir / 'hbv_uq_mpp3.0056'

    # specific for HBV-SASK model
    hbv_model_data_path = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
    inputModelDir = hbv_model_data_path

    directory_for_saving_plots = pathlib.Path('/work/ga45met/paper_uqef_dynamic_sim/hbv_sask/mc_150000_lhc_nse02_2004_2007_oldman')
    if not str(directory_for_saving_plots).endswith("/"):
        directory_for_saving_plots = str(directory_for_saving_plots) + "/"
    directory_for_saving_plots =  pathlib.Path(directory_for_saving_plots)

    list_workindDir = [workingDir_I, workingDir_II, workingDir_III]
    main(list_workindDir=list_workindDir, inputModelDir)