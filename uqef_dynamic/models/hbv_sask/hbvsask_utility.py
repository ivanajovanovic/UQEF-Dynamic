import copy
from collections import defaultdict
import json
import pathlib
import pandas as pd
import math
import matplotlib.pyplot as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import numpy as np
import sys
import time

#####################################

epsilon = sys.float_info.epsilon
DEFAULT_PAR_VALUES_DICT = {'TT': 0.0, 'C0': 0.5, 'ETF': 0.2, 'FC': 250,
                           'beta': 2.0, 'FRAC': 0.3, 'K2': 0.05,  'LP': 0.5,
                           'K1': 0.5, 'alpha': 2.0, 
                           'UBAS': 1, 'PM': 1, "M": 1.0, "VAR_M": 1e-4}

DEFAULT_PAR_VALUES_DICT_EXTEND = {'TT': 0.0, 'C0': 0.5, 'ETF': 0.2, 'LP': 0.5, 'FC': 250,
                           'beta': 2.0, 'FRAC': 0.3, 'K1': 0.5, 'alpha': 2.0, 'K2': 0.05,
                           'UBAS': 1, 'PM': 1, "M": 1.0, "VAR_M": 1e-4}

DEFAULT_PAR_INFO_DICT = {
    'TT': {"lower": -4.0, "upper": 4.0, "default": 0.0},
    'C0': {"lower": 0.0, "upper": 5.0, "default": 0.5},
    'ETF': {"lower": 0.0, "upper": 1.0, "default": 0.2},
    'LP': {"lower": 0.0, "upper": 1.0, "default": 0.5},
    'FC': {"lower": 50.0, "upper": 1000.0, "default": 250.0},
    'beta': {"lower": 1.0, "upper": 3.0, "default": 2.0},
    'FRAC': {"lower": 0.1, "upper": 0.9, "default": 0.3},
    'K1': {"lower": 0.05, "upper": 1.0, "default": 0.5},
    'alpha': {"lower": 1.0, "upper": 3.0, "default": 2.0},
    'K2': {"lower": 0.0, "upper": 0.1, "default": 0.05},
    'UBAS': {"lower": 1.0, "upper": 3.0, "default": 1.0},
    'PM':{"lower": 0.5, "upper": 2.0, "default": 1.0},
    'M':{"lower": 0.9, "upper": 1.0, "default": 1.0},
    'VAR_M':{"lower": 1e-5, "upper": 1e-3, "default": 1e-4}
}

HBV_PARAMS_LIST = ['TT', 'C0', 'ETF', 'LP', 'FC',
                   'beta', 'FRAC', 'K1', 'alpha', 'K2',
                   'UBAS', 'PM']

TIME_COLUMN_NAME="TimeStamp"


def _plot_time_series(df, column_to_plot):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name=column_to_plot,
        x=df.index,
        y=df[column_to_plot],
        mode="lines"
    ))
    fig.update_xaxes(showgrid=True, ticklabelmode="period")
    fig.show()


def _plot_output_data_and_precipitation(input_data_df, simulated_data_df=None, input_data_time_column=None,
                                        simulated_time_column=None, measured_data_column="streamflow",
                                        simulated_column="Q_cms", precipitation_columns="precipitation",
                                        additional_columns=None, plot_measured_data=False):
    reset_index_of_input_data_df = False
    if input_data_time_column is not None and input_data_time_column != "index" and input_data_time_column in input_data_df.columns:
        reset_index_of_input_data_df = True
        input_data_df = input_data_df.set_index(input_data_time_column)

    # filter input_data_df based on time steps which occure in simulated_data_df
    if simulated_time_column is None or not simulated_time_column in simulated_data_df.columns:
        input_data_df = input_data_df.loc[simulated_data_df.index.min():simulated_data_df.index.max()]
    else:
        input_data_df = input_data_df.loc[
                        simulated_data_df[simulated_time_column].min():simulated_data_df[simulated_time_column].max()]

    if precipitation_columns in input_data_df.columns:
        N_max = input_data_df[precipitation_columns].max()

    timesteps_min = input_data_df.index.min()
    timesteps_max = input_data_df.index.max()

    fig = go.Figure()

    if plot_measured_data:
        fig.add_trace(go.Scatter(x=input_data_df.index,
                                 y=input_data_df[measured_data_column],
                                 name=measured_data_column
                                 ))

    if simulated_data_df is not None:
        if simulated_time_column is None or not simulated_time_column in simulated_data_df.columns:
            fig.add_trace(go.Scatter(x=simulated_data_df.index,
                                     y=simulated_data_df[simulated_column],
                                     name=simulated_column
                                     ))
        else:
            fig.add_trace(go.Scatter(x=simulated_data_df[simulated_time_column],
                                     y=simulated_data_df[simulated_column],
                                     name=simulated_column
                                     ))

    if precipitation_columns in input_data_df.columns:
        fig.add_trace(go.Scatter(x=input_data_df.index, y=input_data_df[precipitation_columns],
                                text=input_data_df[precipitation_columns], name="Precipitation", yaxis="y2", ))

    if reset_index_of_input_data_df:
        input_data_df.reset_index(inplace=True)
        input_data_df.rename(columns={"index": input_data_time_column}, inplace=True)

    # Update axes
    fig.update_layout(
        xaxis=dict(
            autorange=True,
            range=[timesteps_min, timesteps_max],
            type="date"
        ),
        yaxis=dict(
            side="left",
            domain=[0, 0.7],
            mirror=True,
            tickfont={"color": "#d62728"},
            tickmode="auto",
            ticks="inside",
            title="Q [cm/s]",
            titlefont={"color": "#d62728"},
        )
    )
    if precipitation_columns in input_data_df.columns:
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
                title="N [mm/day]",
                type="linear",
            ),
        )
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=1.21
    ))
    return fig


def plot_streamflow_and_precipitation(input_data_df, simulated_data_df=None, input_data_time_column=None,
                                       simulated_time_column=None, observed_streamflow_column="streamflow",
                                       simulated_streamflow_column="Q_cms", precipitation_columns="precipitation",
                                       additional_columns=None):
    if input_data_time_column is not None and input_data_time_column != "index" and input_data_time_column in input_data_df.columns:
        input_data_df = input_data_df.set_index(input_data_time_column)

    if simulated_time_column is None or not simulated_time_column in simulated_data_df.columns:
        input_data_df = input_data_df.loc[simulated_data_df.index.min():simulated_data_df.index.max()]
    else:
        input_data_df = input_data_df.loc[
                        simulated_data_df[simulated_time_column].min():simulated_data_df[simulated_time_column].max()]

    N_max = input_data_df[precipitation_columns].max()
    timesteps_min = input_data_df.index.min()
    timesteps_max = input_data_df.index.max()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=input_data_df.index,
                             y=input_data_df[observed_streamflow_column],
                             name="Observed Streamflow"
                             ))

    if simulated_data_df is not None:
        if simulated_time_column is None or not simulated_time_column in simulated_data_df.columns:
            fig.add_trace(go.Scatter(x=simulated_data_df.index,
                                     y=simulated_data_df[simulated_streamflow_column],
                                     name="Simulated Streamflow"
                                     ))
        else:
            fig.add_trace(go.Scatter(x=simulated_data_df[simulated_time_column],
                                     y=simulated_data_df[simulated_streamflow_column],
                                     name="Simulated Streamflow"
                                     ))

    if precipitation_columns in simulated_data_df.columns:
        fig.add_trace(
            go.Scatter(
            x=simulated_data_df.index, y=simulated_data_df[precipitation_columns],
            text=simulated_data_df[precipitation_columns], name="Precipitation", yaxis="y2", 
            )
        )
    else:
        fig.add_trace(go.Scatter(x=input_data_df.index, y=input_data_df[precipitation_columns],
                            text=input_data_df[precipitation_columns], name="Precipitation", yaxis="y2", ))


    if input_data_time_column is not None and input_data_time_column != "index" and input_data_time_column in input_data_df.columns:
        input_data_df.reset_index(inplace=True)
        input_data_df.rename(columns={"index": input_data_time_column}, inplace=True)

    # Update axes
    fig.update_layout(
        xaxis=dict(
            autorange=True,
            range=[timesteps_min, timesteps_max],
            type="date"
        ),
        yaxis=dict(
            side="left",
            domain=[0, 0.7],
            mirror=True,
            tickfont={"color": "#d62728"},
            tickmode="auto",
            ticks="inside",
            title="Q [cm/s]",
            titlefont={"color": "#d62728"},
        ),
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
            title="N [mm/day]",
            type="linear",
        )
    )
    # fig.update_layout(legend=dict(
    #     yanchor="bottom",
    #     y=0.01,
    #     xanchor="right",
    #     x=1.21
    # ))
    fig.update_layout(
            # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=f'HBV-SASK Model: Predicted vs. Observed Streamflow',
            showlegend=True,
            # template="plotly_white",
        )
    return fig

def extend_hbv_plot_with_observed_and_forcing_data_and_update_layout(
    fig, list_of_dates_of_interest, hbv_model, num_model_runs, time_column_name, observed_streamflow_column="streamflow", plot_forcing_data=False, **kwargs):
    """
    :param fig:
    :param list_of_dates_of_interest:
    :param hbv_model: assumed to be a model instance of the HBV model (i.e., HBVSASKModel class)
    :param num_model_runs:
    :param plot_forcing_data:
    :param kwargs:
    :return: fig
    """
    reset_index_at_the_end = False
    if hbv_model.time_series_measured_data_df.index.name != time_column_name:
        hbv_model.time_series_measured_data_df.set_index(time_column_name, inplace=True)
        reset_index_at_the_end = True

    temp = hbv_model.time_series_measured_data_df[hbv_model.time_series_measured_data_df.index.isin(list_of_dates_of_interest)]  # Sample forcing data for plotting only the list_of_dates_of_interest
    fig.add_trace(go.Scatter(x=temp.index,
                             y=temp[observed_streamflow_column],
                             name="Observed Data",
                             line_color='rgba(60, 179, 113, 1)'
                             ))

    if plot_forcing_data:
        N_max = temp['precipitation'].max()
        fig.add_trace(
            go.Bar(x=temp.index,
                    y=temp['precipitation'], 
                    name='Precipitation', yaxis="y2", marker_color='magenta'
                ),
        )

    if reset_index_at_the_end:
        hbv_model.time_series_measured_data_df.reset_index(inplace=True)
        hbv_model.time_series_measured_data_df.rename(columns={hbv_model.time_series_measured_data_df.index.name: TIME_COLUMN_NAME}, inplace=True)

    # Set x-axis title
    fig.update_xaxes(title_text="Date", autorange=True, range=[hbv_model.start_date_predictions, hbv_model.end_date], type="date")

    # Update y-axis
    fig.update_yaxes(title_text="Q [cm/s]", side="left", domain=[0, 0.7], mirror=True, tickfont={"color": "#d62728"},
                    tickmode="auto", ticks="inside", titlefont={"color": "#d62728"}, range=[0, 100])

    fig.update_layout(
        # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=f'Predicted vs. Observed Streamflow with {num_model_runs} particles',
        showlegend=True,
        template="plotly_white",
    )
    
    if plot_forcing_data:
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
            title="N [mm/day]",
            type="linear",
            )
        )
    return fig


def extend_hbv_plot_with_forcing_and_update_layout(
    fig, list_of_dates_of_interest, hbv_model, num_model_runs, time_column_name, plot_forcing_data=False, **kwargs):
    """
    :param fig:
    :param list_of_dates_of_interest:
    :param hbv_model: assumed to be a model instance of the HBV model (i.e., HBVSASKModel class)
    :param num_model_runs:
    :param plot_forcing_data:
    :param kwargs:
    :return: fig
    """
    if plot_forcing_data:
        reset_index_at_the_end = False
        if hbv_model.time_series_measured_data_df.index.name != time_column_name:
            hbv_model.time_series_measured_data_df.set_index(time_column_name, inplace=True)
            reset_index_at_the_end = True
        
        temp = hbv_model.time_series_measured_data_df[hbv_model.time_series_measured_data_df.index.isin(list_of_dates_of_interest)]  # Sample forcing data for plotting only the list_of_dates_of_interest
        N_max = temp['precipitation'].max()
        fig.add_trace(
            go.Bar(x=temp.index,
                    y=temp['precipitation'], 
                    name='Precipitation', yaxis="y2", marker_color='magenta'
                ),
        )
        if reset_index_at_the_end:
            hbv_model.time_series_measured_data_df.reset_index(inplace=True)
            hbv_model.time_series_measured_data_df.rename(columns={hbv_model.time_series_measured_data_df.index.name: TIME_COLUMN_NAME}, inplace=True)

    # Set x-axis title
    fig.update_xaxes(title_text="Date", autorange=True, range=[hbv_model.start_date_predictions, hbv_model.end_date], type="date")

    # Update y-axis
    fig.update_yaxes(title_text="Q [cm/s]", side="left", domain=[0, 0.7], mirror=True, tickfont={"color": "#d62728"},
                    tickmode="auto", ticks="inside", titlefont={"color": "#d62728"}, range=[0, 100])

    fig.update_layout(
        # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=f'Predicted vs. Observed Streamflow with {num_model_runs} particles',
        showlegend=True,
        template="plotly_white",
    )
    
    if plot_forcing_data:
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
            title="N [mm/day]",
            type="linear",
            )
        )
    return fig


def plot_input_output_state(modelObject, result_df, state_df, **kwargs):
    parsed_input_data_df = modelObject.time_series_measured_data_df.loc[
        result_df.index.min():result_df.index.max()]

    fig = make_subplots(
        rows=6, cols=1,
        subplot_titles=("Temperature [°C]", "Precipitation [mm/day]", "Measured Streamflow [m^3/s]", \
        "Evapotranspiration", "Snow Storage", "Soil Storage + Reservoirs"),
        vertical_spacing=0.05
    )

    fig.add_trace(
        go.Scatter(
            x=parsed_input_data_df.index, y=parsed_input_data_df['temperature'],
            text=parsed_input_data_df['temperature'], 
            mode='lines',
            name="Temperature",
            showlegend=False
        ), 
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=parsed_input_data_df.index, y=parsed_input_data_df['precipitation'],
            text=parsed_input_data_df['precipitation'], 
            name="Precipitation",
            showlegend=False
        ), 
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=parsed_input_data_df.index, y=parsed_input_data_df['streamflow'],
            name="Observed Streamflow",
            mode='lines',
            line=dict(dash='dashdot'),
            showlegend=True,
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=result_df.index, y=result_df['Q_cms'],
            text=result_df['Q_cms'], 
            name="Predicted Streamflow",
            showlegend=True,
        ), 
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=result_df.index, y=result_df['PET'],
            text=result_df['PET'], 
            name="Potential evapot.",  # "PET",
            mode='lines',
            line=dict(dash='dashdot'),
            showlegend=True,
        ), 
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=result_df.index, y=result_df['AET'],
            text=result_df['AET'], 
            name="Actual evapot.",  # name="AET",
            showlegend=True,
        ), 
        row=4, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=state_df.index, y=state_df['SWE'],
            text=state_df['SWE'], 
            name="Snow storage",
            showlegend=False,
        ), 
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=state_df.index, y=state_df['SMS'],
            text=state_df['SMS'], 
            name="Soil storage",
            showlegend=True,
        ), 
        row=6, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=state_df.index, y=state_df['S1'],
            text=state_df['S1'], 
            name="Fast reservoir",
            mode='lines',
            line=dict(dash='dashdot'),
            showlegend=True,
        ), 
        row=6, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=state_df.index, y=state_df['S2'],
            text=state_df['S2'], 
            name="Slow reservoir",
            mode='lines',
            line=dict(dash='dot'),
            showlegend=True,
        ), 
        row=6, col=1
    )
    fig.update_layout(
        height=1000, 
        width=1100, 
        title_text="Detailed plot of most important time-series"
        )

    title = kwargs.get("title")
    if title is None:
        title = f'HBV-SASK Model: Forcing, state, measured and predicted time series'
    fig.update_layout(
        # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=title,
        showlegend=True,
        # template="plotly_white",
    )

    # fig.update_layout(
    #     # legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    #     showlegend=True,
    #     # template="plotly_white",
    # )

    return fig
#####################################
# Now comes the set of functiones copied from https://github.com/vars-tool/vars-tool/blob/master/tutorials/hbv.py
# Original authors: Saman Razavi, Hoshin Gupta, Kasra Keshavarz, and Cordell Blanchard
#####################################

# Calculate Change in storage:
def DeltaS(state, start, end):
    S = np.zeros(len(state['SWE']))
    for n in state: S = S + state[n]
    S = S[start:end]
    S = S - S[0]
    return S


def WaterBalancePlot(flux, state, forcing, start, end):
    # Do a nice water balance plot
    t = flux['PET'][start:end].index
    S = DeltaS(state, start, end)
    P = forcing['P'][start:end].cumsum()
    AET = flux['AET'][start:end].cumsum()
    Q = flux['Q_mm'][start:end].cumsum()

    pl.figure(figsize=(10, 5))
    pl.fill_between(t, Q + AET + S, 0., color='darkgreen', label='cumulative Q')
    pl.fill_between(t, S + AET, 0., color='forestgreen', label='cumulative AET')
    pl.fill_between(t, S, 0., color='lightgreen', label='$\Delta S$')
    pl.plot(forcing['P'][start:end].cumsum(), label='cumulative P', color='b')
    # pl.fill_between(flux['Q_mm'][start:end].cumsum()+flux['AET'][start:end].cumsum()+S,label='Streamflow',color='r')
    # pl.plot(flux['AET'][start:end].cumsum(),label='AET',color='darkgreen')

    pl.legend(fontsize=13)
    pl.ylabel('Water balance (mm)', fontsize=13);
    pl.grid()


def PlotEverything(flux, state, forcing, start, end, freq):
    # Do a nice plot of model outputs:
    tS = state['SWE'].resample(freq).mean()[start:end].index
    SWE = (state['SWE'].resample(freq).mean())[start:end]
    SMS = (state['SMS'].resample(freq).mean())[start:end]
    S1 = (state['S1'].resample(freq).mean())[start:end]
    S2 = (state['S2'].resample(freq).mean())[start:end]

    t = flux['PET'][start:end].resample(freq).sum().index
    P = forcing['P'][start:end].resample(freq).sum()
    AET = flux['AET'][start:end].resample(freq).sum()
    PET = flux['PET'][start:end].resample(freq).sum()
    Q = flux['Q_mm'][start:end].resample(freq).sum()

    pl.figure(figsize=(10, 7))
    pl.subplot(2, 1, 1)
    pl.fill_between(t, PET, 0., color='lightgreen', label='PET', step='pre')
    pl.step(t, P, label='P', color='b')
    pl.step(t, Q, label='Q', color='m')
    pl.step(t, AET, label='AET', color='g')
    pl.legend(fontsize=13)
    pl.ylabel('Fluxes (mm)', fontsize=13);
    pl.grid()

    pl.subplot(2, 1, 2)
    pl.step(tS, SWE, label='SWE', color='tab:cyan')
    pl.step(tS, SMS, label='SMS', color='tab:grey')
    pl.step(tS, S1, label='S1', color='tab:olive')
    pl.step(tS, S2, label='S2', color='tab:brown')
    pl.legend(fontsize=13)
    pl.ylabel('States (mm)', fontsize=13);
    pl.grid()

# #####################################
#
#
# def get_param_info_dict(configurationObject=None):
#     configurationObject = utility.check_if_configurationObject_is_in_right_format_and_return(configurationObject,
#                                                                                              raise_error=False)
#     result_dict = defaultdict(dict)
#
#     # list_of_params_names_from_configurationObject = []
#     if configurationObject is not None:
#         for param_entry_dict in configurationObject["parameters"]:
#             param_name = param_entry_dict.get("name")
#             # list_of_params_names_from_configurationObject.append(param_name)
#             distribution = param_entry_dict.get("distribution", None)
#             if "lower_limit" in param_entry_dict:
#                 lower_limit = param_entry_dict["lower_limit"]
#             elif "lower" in param_entry_dict:
#                 lower_limit = param_entry_dict["lower"]
#             else:
#                 lower_limit = None
#             if "upper_limit" in param_entry_dict:
#                 upper_limit = param_entry_dict["upper_limit"]
#             elif "upper" in param_entry_dict:
#                 upper_limit = param_entry_dict["upper"]
#             else:
#                 upper_limit = None
#             # lower_limit = param_entry_dict.get("lower_limit", None)
#             # upper_limit = param_entry_dict.get("upper_limit", None)
#             default = param_entry_dict.get("default", None)
#             # parameter_value = param_entry_dict.get("value", None)
#             result_dict[param_name] = {
#                 'distribution': distribution, 'default': default,
#                 'lower_limit': lower_limit, 'upper_limit': upper_limit
#             }
#
#     for single_param_name in DEFAULT_PAR_INFO_DICT.keys():
#         if single_param_name in result_dict:
#             continue
#         else:
#             param_entry_dict = DEFAULT_PAR_INFO_DICT[single_param_name]
#             param_name = single_param_name
#             distribution = param_entry_dict.get("distribution", None)
#             default = param_entry_dict.get("default", None)
#             if "lower_limit" in param_entry_dict:
#                 lower_limit = param_entry_dict["lower_limit"]
#             elif "lower" in param_entry_dict:
#                 lower_limit = param_entry_dict["lower"]
#             else:
#                 lower_limit = None
#             if "upper_limit" in param_entry_dict:
#                 upper_limit = param_entry_dict["upper_limit"]
#             elif "upper" in param_entry_dict:
#                 upper_limit = param_entry_dict["upper"]
#             else:
#                 upper_limit = None
#             result_dict[param_name] = {
#                 'distribution': distribution, 'default': default,
#                 'lower_limit': lower_limit, 'upper_limit': upper_limit
#             }
#
#     return result_dict
#
#
# def parameters_configuration(parameters, configurationObject, take_direct_value=False):
#     """
#     Note: If not take_direct_value and parameters!= None, parameters_dict will contain
#     some value for every single parameter in configurationObject (e.g., it might at the end have more entries that the
#     input parameters variable)
#     :param parameters:
#     :type parameters: dictionary or array storing all uncertain parameters
#        in the same order as parameters are listed in configurationObject
#     :param configurationObject:
#     :param take_direct_value:
#     :return:
#     """
#     parameters_dict = dict() #defaultdict()  # copy.deepcopy(DEFAULT_PAR_VALUES_DICT)
#
#     if parameters is None:
#         return DEFAULT_PAR_VALUES_DICT
#
#     if isinstance(parameters, dict) and take_direct_value:
#         parameters_dict = parameters
#     else:
#         uncertain_param_counter = 0
#         configurationObject = utility.check_if_configurationObject_is_in_right_format_and_return(configurationObject)
#         for single_param in configurationObject['parameters']:
#             if single_param['distribution'] != "None":
#                 # TODO Does it make sense to round the value of parameters?
#                 parameters_dict[single_param['name']] = parameters[uncertain_param_counter]
#                 uncertain_param_counter += 1
#             else:
#                 if "value" in single_param:
#                     parameters_dict[single_param['name']] = single_param["value"]
#                 elif "default" in single_param:
#                     parameters_dict[single_param['name']] = single_param["default"]
#                 else:
#                     parameters_dict[single_param['name']] = DEFAULT_PAR_VALUES_DICT[single_param['name']]
#     return parameters_dict
#
#
# # def parameters_configuration_for_gradient_approximation(
# #         parameters_dict, configurationObject, parameter_index_to_perturb, eps_val=1e-4, take_direct_value=False):
# #
# #     info_dict_on_perturbed_param = dict()
# #
# #     configurationObject = utility._check_if_configurationObject_is_in_right_format_and_return(configurationObject)
# #     uncertain_param_counter = 0
# #     for id, single_param in enumerate(configurationObject['parameters']):
# #         # TODO if uncertain_param_counter != parameter_index_to_perturb:
# #         if id != parameter_index_to_perturb:
# #             if single_param['distribution'] != "None" and parameters[uncertain_param_counter] is not None:
# #                 parameters_dict[single_param['name']] = parameters[uncertain_param_counter]
# #                 uncertain_param_counter += 1
# #             else:
# #                 if "value" in single_param:
# #                     parameters_dict[single_param['name']] = single_param["value"]
# #                 elif "default" in single_param:
# #                     parameters_dict[single_param['name']] = single_param["default"]
# #                 else:
# #                     parameters_dict[single_param['name']] = DEFAULT_PAR_VALUES_DICT[single_param['name']]
# #         else:
# #             if "lower_limit" in single_param:
# #                 parameter_lower_limit = single_param["lower_limit"]
# #             elif "lower" in single_param:
# #                 parameter_lower_limit = single_param["lower"]
# #             else:
# #                 parameter_lower_limit = None
# #
# #             if "upper_limit" in single_param:
# #                 parameter_upper_limit = single_param["upper_limit"]
# #             elif "upper" in single_param:
# #                 parameter_upper_limit = single_param["upper"]
# #             else:
# #                 parameter_upper_limit = None
# #
# #             if parameter_lower_limit is None or parameter_upper_limit is None:
# #                 raise Exception(
# #                     'ERROR in parameters_configuration: perturb_sinlge_param_around_nominal is set to True but '
# #                     'parameter_lower_limit or parameter_upper_limit are not specified!')
# #             else:
# #                 param_h = eps_val * (parameter_upper_limit - parameter_lower_limit)
# #                 parameter_lower_limit += param_h
# #                 parameter_upper_limit -= param_h
# #
# #             if single_param['distribution'] != "None" and parameters[uncertain_param_counter] is not None:
# #                 new_parameter_value = parameters[uncertain_param_counter] + param_h
# #                 parameters_dict[single_param['name']] = (new_parameter_value, param_h)
# #                 uncertain_param_counter += 1
# #             else:
# #                 if "value" in single_param:
# #                     parameters_dict[single_param['name']] = single_param["value"] + param_h
# #                 elif "default" in single_param:
# #                     parameters_dict[single_param['name']] = single_param["default"] + param_h
# #                 else:
# #                     parameters_dict[single_param['name']] = DEFAULT_PAR_VALUES_DICT[single_param['name']] + param_h
# #
# #             info_dict_on_perturbed_param = {
# #                 "uncertain_param_counter": uncertain_param_counter, "id": id,
# #                 "name": single_param['name'], "param_h": param_h}
# #
# #     return parameters_dict, info_dict_on_perturbed_param
#
# def update_parameter_dict_for_gradient_computation(parameters, configurationObject, take_direct_value=False,
#                                       perturb_single_param_around_nominal=False,
#                                       parameter_index_to_perturb=0, eps_val=1e-4
#                                       ):
#     # TODO Rewrite bigger part of the function above
#     # iterate through all the parameters
#     list_of_parameters_from_json = configurationObject["parameters"]
#
#     for id, param_entry_dict in enumerate(list_of_parameters_from_json):
#         if perturb_single_param_around_nominal and id != parameter_index_to_perturb:
#             continue
#
#     parameter_lower_limit = param_entry_dict["lower_limit"] if "lower_limit" in param_entry_dict else None
#     parameter_upper_limit = param_entry_dict["upper_limit"] if "upper_limit" in param_entry_dict else None
#     param_h = eps_val * (parameter_upper_limit - parameter_lower_limit)
#     parameter_lower_limit += param_h
#     parameter_upper_limit -= param_h
#     raise NotImplementedError

#####################################


def read_streamflow(streamflow_inp, time_column_name="TimeStamp", streamflow_column_name="streamflow"):
    streamflow_dict = dict()
    with open(streamflow_inp, "r") as file:
        for line in file.readlines():
            line = line.strip()
            date, value = line.split()
            streamflow_dict[date] = float(value)
    streamflow_df = pd.DataFrame.from_dict(streamflow_dict, orient='index', columns=[streamflow_column_name, ])
    streamflow_df.index = pd.to_datetime(streamflow_df.index)
    streamflow_df.index.name = time_column_name
    return streamflow_df


def read_precipitation_temperature(precipitation_temperature_inp,
                                   time_column_name="TimeStamp",
                                   precipitation_column_name="precipitation",
                                   temperature_column_name="temperature"):
    precipitation_temperature_inp_dict = defaultdict(list)
    precipitation_temperature_inp_dict[time_column_name] = []
    precipitation_temperature_inp_dict[precipitation_column_name] = []
    precipitation_temperature_inp_dict[temperature_column_name] = []

    with open(precipitation_temperature_inp, "r") as file:
        for line in file.readlines():
            line = line.strip()
            date, prec, temp = line.split()
            precipitation_temperature_inp_dict[time_column_name].append(date)
            precipitation_temperature_inp_dict[precipitation_column_name].append(float(prec))
            precipitation_temperature_inp_dict[temperature_column_name].append(float(temp))

    precipitation_temperature_df = pd.DataFrame(precipitation_temperature_inp_dict)
    precipitation_temperature_df[time_column_name] = pd.to_datetime(precipitation_temperature_df[time_column_name])
    precipitation_temperature_df.set_index(time_column_name, inplace=True)

    return precipitation_temperature_df


def read_initial_conditions(initial_condition_file, return_dict_or_df="dict", timestamp=None,
                            time_column_name="TimeStamp"):
    if str(initial_condition_file).endswith('.inp'):
        initial_condition_dict = defaultdict(list)
        initial_condition_dict["WatershedArea_km2"] = []
        initial_condition_dict["SWE"] = []
        initial_condition_dict["SMS"] = []
        initial_condition_dict["S1"] = []
        initial_condition_dict["S2"] = []

        with open(initial_condition_file, "r") as file:
            for line in file.readlines():
                line = line.strip()
                list_of_values_per_line = line.split()
                if len(list_of_values_per_line) == 2:
                    initial_condition_dict[list_of_values_per_line[0]].append(float(list_of_values_per_line[1]))

        if return_dict_or_df == "dict":
            return initial_condition_dict
        else:
            initial_condition_df = pd.DataFrame(initial_condition_dict)
            return initial_condition_df
    else:
        initial_condition_df = pd.read_pickle(initial_condition_file, compression="gzip")
        if timestamp is None:
            timestamp = initial_condition_df[time_column_name].min()  # initial_condition_df.loc[0].TimeStamp
        else:
            timestamp = pd.Timestamp(timestamp)
        return initial_condition_df.loc[initial_condition_df[time_column_name] == timestamp]


def read_long_term_data(monthly_data_inp, time_column_name="month", precipitation_column_name="monthly_average_PE",
                                   temperature_column_name="monthly_average_T"):
    precipitation_temperature_monthly = defaultdict(list)
    precipitation_temperature_monthly[time_column_name] = []
    precipitation_temperature_monthly[precipitation_column_name] = []
    precipitation_temperature_monthly[temperature_column_name] = []
    with open(monthly_data_inp, "r") as file:
        inx = 0
        for line in file.readlines():
            inx += 1
            line = line.strip()
            if len(line.split()) == 2:
                temp, prec = line.split()
                precipitation_temperature_monthly[time_column_name].append(int(inx))
                precipitation_temperature_monthly[precipitation_column_name].append(float(prec))
                precipitation_temperature_monthly[temperature_column_name].append(float(temp))
    precipitation_temperature_monthly_df = pd.DataFrame(precipitation_temperature_monthly)
    precipitation_temperature_monthly_df.set_index(time_column_name, inplace=True)
    return precipitation_temperature_monthly_df


def read_param_setup_dict(factorSpace_txt):
    par_values_dict = defaultdict(dict)
    number_of_parameters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    with open(factorSpace_txt, "r", encoding="ISO-8859-15") as file:
        for line in file.readlines():
            line = line.strip()
            elements_in_one_line = line.split()
            if elements_in_one_line[0] in number_of_parameters:
                par_values_dict[elements_in_one_line[3]]["lower"] = elements_in_one_line[1]
                par_values_dict[elements_in_one_line[3]]["upper"] = elements_in_one_line[2]
    return par_values_dict


#####################################


def soil_storage_routing_module(ponding, SMS, S1, S2, AET, FC, beta, FRAC, K1, alpha, K2):
    """
    The function should return SMS_new, S1_new, S2_new, Q1, Q2
    *****  ponding: at time t*****
    *****  SMS: Soil Moisture Storage at time t - model state variable *****
    *****  S1: at time t*****
    *****  S2: at time t*****
    *****  AET: Actual Evapotranspiration at time t *****
    *****  FC: Field Capacity - model parameter ---------
    *****  beta: Shape Parameter/Exponent - model parameter ---------
        This controls the relationship between soil infiltration and soil water release.
        The default value is 1. Values less than this indicate a delayed response, while higher
        values indicate that runoff will exceed infiltration.
    *****  FRAC: Fraction of soil release entering fast reservoir ---------
    *****  K1: Fast reservoir coefficient, which determines what proportion of the storage is released per day ---------
    *****  alpha: Shape parameter (exponent) for fast reservoir equation ---------
    *****  K2: Slow reservoir coefficient which determines what proportion of the storage is released per day ---------
    """

    if SMS < FC:
        # release of water from soil
        soil_release = ponding * ((SMS / FC) ** beta)
    else:
        # release of water from soil
        soil_release = ponding

    SMS_new = SMS - AET + ponding - soil_release
    #  this might happen due to very small numerical/rounding errors
    if SMS_new < 0:
        SMS_new = 0

    soil_release_to_fast_reservoir = FRAC * soil_release
    soil_release_to_slow_reservoir = (1 - FRAC) * soil_release

    # Q1 = K1 * (S1 ** alpha)  # TODO make sure that it is not (K1 * S1) ** alpha np.pow(S1, alpha)
    Q1 = K1 * (S1 ** alpha)
    if Q1 > S1:
        Q1 = S1

    S1_new = S1 + soil_release_to_fast_reservoir - Q1

    Q2 = K2 * S2

    S2_new = S2 + soil_release_to_slow_reservoir - Q2

    return SMS_new, S1_new, S2_new, Q1, Q2


def evapotranspiration_module(SMS, T, monthly_average_T, monthly_average_PE, ETF, LP):
    """
    The function should return AET - Actual Evapotranspiration at time t,
    PET - Potential Evapotranspiration at time t
    *****  SMS: Soil Moisture Storage at time t - model state variable *****
    *****  T: Temperature at time t - model forcing *****
    *****  monthly_average_T: *****
    *****  monthly_average_PE: *****
    *****  ETF - This is the temperature anomaly correction of potential evapotranspiration - model parameters *****
    *****  LP: This is the soil moisture content below which evaporation becomes supply-limited - model parameter *****
    """
    # Potential Evapotranspiration
    PET = (1 + ETF * (T - monthly_average_T)) * monthly_average_PE
    PET = max((PET, 0))

    if SMS > LP:
        AET = PET
    else:
        # if np.absolute(LP) < epsilon:
        # if math.isclose(LP, 0):
        if abs(LP) < 1e-9:  # in this case both SMS and LP are close to zero lim(x/x) x->0 is 1
            AET = PET
        else:
            AET = PET * (SMS / LP)

    # to avoid evaporating more than water available
    AET = min((AET, SMS))

    return AET, PET


def precipitation_module(SWE, Precipitation, Temperature, TT, C0):
    """
    The function should return SWE at time t+1, ponding at time t
    *****  SWE: Snow Water Equivalent at time t - model state variable *****
    *****  Precipitation: Precipitation at time t - model forcing *****
    *****  Temperature: Temperature at time t - model forcing *****
    *****  TT: Temperature Threshold or melting/freezing point - model parameter *****
    *****  C0: base melt factor - model parameter *****
    """
    if Temperature >= TT:
        rainfall = Precipitation
        potential_snow_melt = C0 * (Temperature - TT)
        snow_melt = min((potential_snow_melt, SWE))
        ponding = rainfall + snow_melt  # Liquid Water on Surface
        SWE_new = SWE - snow_melt  # Soil Water Equivalent - Solid Water on Surface
    else:
        snowfall = Precipitation
        snow_melt = 0
        ponding = 0  # Liquid Water on Surface
        SWE_new = SWE + snowfall  # Soil Water Equivalent + Solid Water on Surface

    return SWE_new, ponding


def triangle_routing(Q, UBAS):
    """
    The function should return Q_routed - list/1d-array
    *****  Q: list/1d-array *****
    *****  UBAS: Base of unit hydrograph for watershed routing in day; default is 1 for small watersheds *****
    """
    UBAS = max((UBAS, 0.1))
    length_triangle_base = int(math.ceil(UBAS))

    if UBAS == length_triangle_base:
        # x = [0, 0.5 * UBAS, length_triangle_base]
        # v = [0, 1, 0]
        x = np.array([0, 0.5 * UBAS, length_triangle_base])
        v = np.array([0, 1, 0])
    else:
        # x = [0, 0.5 * UBAS, UBAS, length_triangle_base]
        # v = [0, 1, 0, 0]
        x = np.array([0, 0.5*UBAS, UBAS, length_triangle_base])
        v = np.array([0, 1, 0, 0])

    # weight = np.empty(shape=(length_triangle_base + 1,), dtype=np.float64)
    weight=np.zeros(length_triangle_base)
    weight[0] = 0

    # np.interp(2.5, xp, fp) or f = scipy.interpolate.interp1d(x, y); f(xnew)
    for i in range(1, length_triangle_base + 1):
        if (i - 1) < (0.5 * UBAS) and i > (0.5 * UBAS):
            # weight[i] = 0.5 * (np.interp(i - 1, x, v) + np.interp(0.5 * UBAS, x, v)) * (0.5 * UBAS - i + 1) + \
            #             0.5 * (np.interp(0.5 * UBAS, x, v) + np.interp(i, x, v)) * (i - 0.5 * UBAS)
            weight[i-1] = 0.5 * (np.interp(i - 1, x, v) + np.interp(0.5 * UBAS, x, v)) * (0.5 * UBAS - i + 1) + \
                        0.5 * (np.interp(0.5 * UBAS, x, v) + np.interp(i, x, v)) * (i - 0.5 * UBAS)
        elif i > UBAS:
            # weight[i] = 0.5 * np.interp(i - 1, x, v) * (UBAS - i + 1)
            weight[i-1] = 0.5 * np.interp(i - 1, x, v) * (UBAS - i + 1)
        else:
            # weight[i] = np.interp(i - 0.5, x, v)
            weight[i-1] = np.interp(i - 0.5, x, v)

    weight = weight / np.sum(weight)

    # Q_routed = np.empty_like(Q)
    Q_routed=np.zeros(len(Q))
    # for i in range(len(Q)):
    for i in range(1, len(Q)+1):
        temp = 0
        window_len = min((i, length_triangle_base))
        for j in range(1, 1 + window_len):
            # temp += weight[j + 1] * Q[i - j - 1]  # TODO Q[i - j]
            temp += weight[j - 1] * Q[i - j]  # TODO Q[i - j]
        # Q_routed[i] = temp
        Q_routed[i-1] = temp

    return Q_routed


#####################################
def HBV_SASK(forcing, long_term, par_values_dict, initial_condition_df, printing=False, time_column_name="TimeStamp",
             precipitation_column_name="precipitation", temperature_column_name="temperature",
             long_term_precipitation_column_name="monthly_average_PE",
             long_term_temperature_column_name="monthly_average_T", corrupt_forcing_data=False, input_error_model="multiplicative"):
    """
    HBV-SASK has 12 parameters: The first 10 ones are necessary
    to run the model, and parameters 11 and 12, if not given,
    will be set at their default values.
    :param corrupt_forcing_data:
    :param initial_condition_df should be (subset of) a pd.DataFrame that just contains 
    one row with the initial conditions for the current time t, 
    for which the forcing data is available and the predictions will be produced
    : input_error_model: "multiplicative" or "additive"
    """
    if par_values_dict is None:
        par_values_dict = {
            'TT': 0.0, 'C0': 5.0, 'ETF': 0.5, 'LP': 0.5, 'FC': 100, 'beta': 2.0, 'FRAC': 0.5,
            'K1': 0.5, 'alpha': 2.0, 'K2': 0.025, 'UBAS': 1, 'PM': 1
        }
    try:
        TT = par_values_dict.get("TT", DEFAULT_PAR_VALUES_DICT["TT"])  # float(par_values_dict["TT"])
        C0 = par_values_dict.get("C0", DEFAULT_PAR_VALUES_DICT["C0"])  # float(par_values_dict["C0"])
        ETF = par_values_dict.get("ETF", DEFAULT_PAR_VALUES_DICT["ETF"])  # float(par_values_dict["ETF"])
        LP = par_values_dict.get("LP", DEFAULT_PAR_VALUES_DICT["LP"])  # float(par_values_dict["LP"])
        FC = par_values_dict.get("FC", DEFAULT_PAR_VALUES_DICT["FC"])  # float(par_values_dict["FC"])
        beta = par_values_dict.get("beta", DEFAULT_PAR_VALUES_DICT["beta"])  # float(par_values_dict["beta"])
        FRAC = par_values_dict.get("FRAC", DEFAULT_PAR_VALUES_DICT["FRAC"])  # float(par_values_dict["FRAC"])
        K1 = par_values_dict.get("K1", DEFAULT_PAR_VALUES_DICT["K1"])  # float(par_values_dict["K1"])
        alpha = par_values_dict.get("alpha", DEFAULT_PAR_VALUES_DICT["alpha"])  # float(par_values_dict["alpha"])
        K2 = par_values_dict.get("K2", DEFAULT_PAR_VALUES_DICT["K2"])  # float(par_values_dict["K2"])
    except KeyError:
        print(f"Error while reading parameter values from param dictionary!")
        raise

    UBAS = float(par_values_dict.get("UBAS", 1))
    PM = float(par_values_dict.get("PM", 1))

    LP = LP * FC

    watershed_area = initial_condition_df["WatershedArea_km2"].values[0]
    initial_SWE = float(initial_condition_df["SWE"].values[0])
    initial_SMS = float(initial_condition_df["SMS"].values[0])
    initial_S1 = float(initial_condition_df["S1"].values[0])
    initial_S2 = float(initial_condition_df["S2"].values[0])

    flux = defaultdict(dict)
    state = defaultdict(dict)

    precipitation_array = forcing[precipitation_column_name].to_numpy()
    # Optional - corrupting the precipitation, e.g., Ajami et. al. 2007
    if corrupt_forcing_data:
        period_length = len(precipitation_array)
        if input_error_model == "additive":
            pertubation = np.random.normal(loc=0, scale=0.3*precipitation_array, size=period_length)
            precipitation_array = precipitation_array + pertubation
        elif input_error_model == "multiplicative":
            M = float(par_values_dict.get("M", DEFAULT_PAR_VALUES_DICT["M"]))
            VAR_M = float(par_values_dict.get("VAR_M", DEFAULT_PAR_VALUES_DICT["VAR_M"]))
            r = np.random.normal(loc=M, scale=np.sqrt(VAR_M), size=period_length)
            precipitation_array = np.multiply(r, precipitation_array)
        else:
            raise NotImplementedError(f"HBV-SASK Model: Sorry when corrucpting input data, input_error_model should be either additive or multiplicative; {input_error_model} is not supported")
    P = PM * precipitation_array
    T = forcing[temperature_column_name].to_numpy()

    if time_column_name in forcing.columns:
        time_series = forcing[time_column_name]
    else:
        time_series = forcing.index
    #     monthly_average_T = long_term["monthly_average_T"].to_numpy()
    #     monthly_average_PE = long_term["monthly_average_PE"].to_numpy()
    # forcing['month_time_series']=forcing.index.month.values

    period_length = len(P)  # P.shape[0]

    if printing:
        print(f" watershed_area={watershed_area}")
        print(f" initial_SWE={initial_SWE} \n initial_SMS={initial_SMS} "
              f"\n initial_S1={initial_S1} \n initial_S2={initial_S2} \n")
        print(f"period_length={period_length}")

    SWE = np.zeros(shape=(period_length + 1,), dtype=np.float64)
    SMS = np.zeros(shape=(period_length + 1,), dtype=np.float64)
    S1 = np.zeros(shape=(period_length + 1,), dtype=np.float64)
    S2 = np.zeros(shape=(period_length + 1,), dtype=np.float64)
    Q1 = np.zeros(shape=period_length, dtype=np.float64)
    Q2 = np.zeros(shape=period_length, dtype=np.float64)
    AET = np.zeros(shape=period_length, dtype=np.float64)
    PET = np.zeros(shape=period_length, dtype=np.float64)
    ponding = np.zeros(shape=period_length, dtype=np.float64)

    SWE[0] = initial_SWE
    SMS[0] = initial_SMS
    S1[0] = initial_S1
    S2[0] = initial_S2

    for t in range(period_length):
        # iteratively over time steps computing the state for the next time step and the current model output
        month = time_series[t].month  # the current month number - for Jan=1, ..., Dec=12
        single_monthly_average_PE = long_term.loc[month][long_term_precipitation_column_name]
        single_monthly_average_T = long_term.loc[month][long_term_temperature_column_name]

        SWE[t + 1], ponding[t] = precipitation_module(SWE[t], P[t], T[t], TT, C0)

        AET[t], PET[t] = evapotranspiration_module(
            SMS[t], T[t], single_monthly_average_T, single_monthly_average_PE, ETF, LP
        )

        SMS[t + 1], S1[t + 1], S2[t + 1], Q1[t], Q2[t] = soil_storage_routing_module(
            ponding[t], SMS[t], S1[t], S2[t], AET[t], FC, beta, FRAC, K1, alpha, K2
        )

    Q1_routed = triangle_routing(Q1, UBAS)
    Q = Q1_routed + Q2
    Q_cms = (Q * watershed_area * 1000) / (24 * 3600)
    Q_cms[Q_cms < 1e-4] = 1e-4
    AET[AET < 1e-5] = 1e-5
    flux["Q_cms"] = Q_cms #.conjugate()
    # Make sure flows will never get negative values because of numerical errors
    # flux["Q_cms"][flux["Q_cms"] < 10 ** -5] = 10 ** -5
    # flux["Q_cms"] = flux["Q_cms"].apply(lambda x: 1e-4 if x < 1e-4 else x)

    flux["Q_mm"] = Q #.conjugate()
    flux["AET"] = AET #.conjugate()
    flux["PET"] = PET #.conjugate()
    flux["Q1"] = Q1 #.conjugate()
    flux["Q1_routed"] = Q1_routed #.conjugate()
    flux["Q2"] = Q2 #.conjugate()
    flux["ponding"] = ponding #.conjugate()
    if corrupt_forcing_data:
        flux["precipitation"] = precipitation_array

    state["SWE"] = SWE #.conjugate()
    state["SMS"] = SMS #.conjugate()
    state["S1"] = S1 #.conjugate()
    state["S2"] = S2 #.conjugate()

    return flux, state


#####################################


def _get_full_time_span(basis):
    if basis == 'Banff_Basin':
        start_date = pd.Timestamp('1950-01-01 00:00:00')
        end_date = pd.Timestamp('2011-12-31 00:00:00')
    elif basis == 'Oldman_Basin':
        start_date = pd.Timestamp('1979-01-01 00:00:00')
        end_date = pd.Timestamp('2008-12-31 00:00:00')
    else:
        start_date = None
        end_date = None
    return start_date, end_date


# TODO Change the function such that less is pre-assumed about the structure of different dfs
def run_the_model(hbv_model_path, config_file, par_values_dict, run_full_timespan=False, basis='Oldman_Basin',
                  plotting=False, writing_results_to_a_file=False, output_path=None, **kwargs):
    # Preparing paths
    path_to_input = hbv_model_path / basis
    # initial_condition_file = path_to_input / "initial_condition.inp"
    initial_condition_file = path_to_input / "state_df.pkl"
    # initial_condition_file = path_to_input / "state_const_df.pkl"
    monthly_data_inp = path_to_input / "monthly_data.inp"
    precipitation_temperature_inp = path_to_input / "Precipitation_Temperature.inp"
    streamflow_inp = path_to_input / "streamflow.inp"
    factorSpace_txt = hbv_model_path / "factorSpace.txt"

    with open(config_file) as f:
        configuration_object = json.load(f)

    if run_full_timespan:
        start_date, end_date = _get_full_time_span(basis)
    else:
        try:
            start_date = pd.Timestamp(
                year=configuration_object["time_settings"]["start_year"],
                month=configuration_object["time_settings"]["start_month"],
                day=configuration_object["time_settings"]["start_day"],
                hour=configuration_object["time_settings"]["start_hour"]
            )
            end_date = pd.Timestamp(
                year=configuration_object["time_settings"]["end_year"],
                month=configuration_object["time_settings"]["end_month"],
                day=configuration_object["time_settings"]["end_day"],
                hour=configuration_object["time_settings"]["end_hour"]
            )
        except KeyError:
            start_date, end_date = _get_full_time_span(basis)

    if "spin_up_length" in kwargs:
        spin_up_length = kwargs["spin_up_length"]
    else:
        try:
            spin_up_length = configuration_object["time_settings"]["spin_up_length"]
        except KeyError:
            spin_up_length = 0  # 365*3

    if "simulation_length" in kwargs:
        simulation_length = kwargs["simulation_length"]
    else:
        try:
            simulation_length = configuration_object["time_settings"]["simulation_length"]
        except KeyError:
            simulation_length = (end_date - start_date).days - spin_up_length
            if simulation_length <= 0:
                simulation_length = 365

    start_date_predictions = pd.to_datetime(start_date) + pd.DateOffset(days=spin_up_length)
    end_date = pd.to_datetime(start_date_predictions) + pd.DateOffset(days=simulation_length)
    full_data_range = pd.date_range(start=start_date, end=end_date, freq="1D")
    simulation_range = pd.date_range(start=start_date_predictions, end=end_date, freq="1D")

    # print(f"start_date-{start_date}; spin_up_length-{spin_up_length}; start_date_predictions-{start_date_predictions}")
    # print(
    #     f"start_date_predictions-{start_date_predictions}; simulation_length-{simulation_length}; end_date-{end_date}")
    # print(len(simulation_range), (end_date - start_date_predictions).days)
    # assert len(time_series_data_df[start_date:end_date]) == len(full_data_range)

    # Reading the input data
    time_column_name = "TimeStamp"
    streamflow_column_name = "streamflow"
    precipitation_column_name = "precipitation"
    temperature_column_name = "temperature"
    long_term_precipitation_column_name = "monthly_average_PE"
    long_term_temperature_column_name = "monthly_average_T"

    streamflow_df = read_streamflow(streamflow_inp, time_column_name=time_column_name, streamflow_column_name=streamflow_column_name)
    precipitation_temperature_df = read_precipitation_temperature(precipitation_temperature_inp,
                                                                  time_column_name=time_column_name,
                                                                  precipitation_column_name=precipitation_column_name,
                                                                  temperature_column_name=temperature_column_name)
    time_series_data_df = pd.merge(streamflow_df, precipitation_temperature_df, left_index=True, right_index=True)
    precipitation_temperature_monthly_df = read_long_term_data(monthly_data_inp,
                                                               time_column_name=time_column_name,
                                                               precipitation_column_name=long_term_precipitation_column_name,
                                                               temperature_column_name=long_term_temperature_column_name)
    param_setup_dict = read_param_setup_dict(factorSpace_txt)

    # Parse input based on some timeframe
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    if time_column_name in time_series_data_df.columns:
        time_series_data_df = time_series_data_df.loc[
            (time_series_data_df[time_column_name] >= start_date) & (time_series_data_df[time_column_name] <= end_date)]
    else:
        time_series_data_df = time_series_data_df[start_date:end_date]
    # initial_condition_df = read_initial_conditions(initial_condition_file, return_dict_or_df="df")
    initial_condition_df = read_initial_conditions(initial_condition_file, timestamp=start_date, time_column_name=time_column_name)
    # print(initial_condition_df)

    if plotting:
        fig = make_subplots(rows=3, cols=1)
        fig.add_trace(
            go.Scatter(x=time_series_data_df.index, y=time_series_data_df[precipitation_column_name], name="P"),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=time_series_data_df.index, y=time_series_data_df[temperature_column_name], name="T"),
            row=2, col=1)
        fig.add_trace(
            go.Scatter(x=time_series_data_df.index, y=time_series_data_df[streamflow_column_name], name="Q_cms"),
            row=3, col=1)
        plot_filename = output_path / f"forcing_data.html"
        plot(fig, filename=str(plot_filename), auto_open=False)

    #############################

    # Running the model
    flux, state = HBV_SASK(forcing=time_series_data_df, long_term=precipitation_temperature_monthly_df,
                           par_values_dict=par_values_dict, initial_condition_df=initial_condition_df, printing=True,
                           time_column_name=time_column_name, precipitation_column_name=precipitation_column_name,
                           temperature_column_name=temperature_column_name,
                           long_term_precipitation_column_name=long_term_precipitation_column_name,
                           long_term_temperature_column_name=long_term_temperature_column_name)

    # Create a final df - flux
    time_series_list = list(full_data_range)  # list(simulation_range)
    assert len(list(full_data_range)) == len(flux["Q_cms"])

    flux_df = pd.DataFrame(
        list(zip(time_series_list, flux["Q_cms"], flux["Q_mm"], flux["AET"], flux["PET"], flux["Q1"], flux["Q1_routed"],
                 flux["Q2"], flux["ponding"])),
        columns=[time_column_name, 'Q_cms', 'Q_mm', 'AET', 'PET', 'Q1', 'Q1_routed', 'Q2', "ponding"]
    )

    # Create a final df - state
    last_date = time_series_list[-1]
    time_series_list.append(pd.to_datetime(last_date) + pd.DateOffset(days=1))
    state_df = pd.DataFrame(
        list(zip(time_series_list, state["SWE"], state["SMS"], state["S1"], state["S2"])),
        columns=[time_column_name, 'SWE', 'SMS', 'S1', 'S2', ]
    )
    state_df['WatershedArea_km2'] = initial_condition_df["WatershedArea_km2"].values[0]

    # Parse flux_df between start_date_predictions, end_date
    flux_df.set_index(time_column_name, inplace=True)
    flux_df = flux_df.loc[simulation_range]  # flux_df[start_date_predictions:end_date]

    # Append measured streamflow to flux_df, i.e., merge flux_df and time_series_data_df[streamflow_column_name]
    # df3 = pd.merge(flux_df, time_series_data_df[[streamflow_column_name]], left_index=True, right_index=True)
    flux_df = flux_df.merge(time_series_data_df[[streamflow_column_name, ]], left_index=True, right_index=True)

    # Parse state_df between start_date_predictions, end_date + 1
    state_df.set_index(time_column_name, inplace=True)
    state_df = state_df[start_date_predictions:]

    # TODO-Ivana Compute Metrics - from my code and from VARS code

    # reset the index
    flux_df.reset_index(inplace=True)
    flux_df.rename(columns={"index": time_column_name}, inplace=True)
    state_df.reset_index(inplace=True)
    state_df.rename(columns={"index": time_column_name}, inplace=True)

    # Write to a file
    if writing_results_to_a_file and output_path is not None:
        file_path = output_path / f"flux_df.pkl"
        flux_df.to_pickle(file_path, compression="gzip")
        file_path = output_path / f"state_df.pkl"
        state_df.to_pickle(file_path, compression="gzip")

    if plotting:
        fig = plot_streamflow_and_precipitation(input_data_df=time_series_data_df, simulated_data_df=flux_df,
                                                 input_data_time_column=time_column_name,
                                                 simulated_time_column=time_column_name,
                                                 observed_streamflow_column=streamflow_column_name,
                                                 simulated_streamflow_column="Q_cms",
                                                 precipitation_columns=precipitation_column_name,
                                                 additional_columns=None)
        # fig.add_trace(go.Scatter(x=flux_df.index, y=flux_df["Q_cms"], name="Q_cms"))
        plot_filename = output_path / f"hbv_sask_{basis}.html"
        plot(fig, filename=str(plot_filename), auto_open=False)
        # fig.show()

    return flux_df, state_df


if __name__ == "__main__":
    # Path definitions - change them accordingly
    hbv_model_path = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
    # basis = 'Oldman_Basin'  # to read in data for the Oldman Basin
    basis = 'Banff_Basin'  # to read in data for the Banff Basin
    config_file = pathlib.Path("/work/ga45met/mnt/linux_cluster_2/UQEFPP/configurations/configuration_hbv.json")
    output_path = hbv_model_path / basis / "model_runs" / "temp_7_constant_ic"
    output_path.mkdir(parents=True, exist_ok=True)

    # this will overwrite configurations from the json file
    run_full_timespan = False  # True
    plotting = True
    writing_results_to_a_file = True

    ################################

    # parameter dictionaries
    par_values_dict_extreme_lower = {
        'TT': -4.0, 'C0': 0.0, 'ETF': 0.0, 'LP': 0.0, 'FC': 50, 'beta': 1.0, 'FRAC': 0.1,
        'K1': 0.05, 'alpha': 1.0, 'K2': 0.025, 'UBAS': 1, 'PM': 1
    }

    par_values_dict_extreme_upper = {
        'TT': 0.0, 'C0': 5.0, 'ETF': 0.5, 'LP': 0.5, 'FC': 100, 'beta': 2.0, 'FRAC': 0.5,
        'K1': 0.5, 'alpha': 2.0, 'K2': 0.025, 'UBAS': 1, 'PM': 1
    }

    par_values_dict_mean = {
        'TT': 0.0, 'C0': 5.0, 'ETF': 0.5, 'LP': 0.5, 'FC': 100, 'beta': 2.0, 'FRAC': 0.5,
        'K1': 0.5, 'alpha': 2.0, 'K2': 0.025, 'UBAS': 1, 'PM': 1
    }

    simulation_time_start = time.time()
    print(f"simulation_time_start-{simulation_time_start}")

    flux, state = run_the_model(
        hbv_model_path, config_file, par_values_dict_mean, run_full_timespan=run_full_timespan, basis=basis,
        plotting=plotting, writing_results_to_a_file=writing_results_to_a_file, output_path=output_path)  # spin_up_length=0

    simulation_time_end = time.time()
    simulation_time = simulation_time_end - simulation_time_start
    print("Total time (date preprocessing, simulation time, data postprocessing): "
          "{} sec; timesteps={}".format(simulation_time, len(flux["Q_cms"])))
    # For the full simulation it prints: simulation time: 1.3676586151123047 sec; timesteps=10958

    Q = flux["Q_cms"]
    ET = flux["AET"]
    SM = state["SMS"]
