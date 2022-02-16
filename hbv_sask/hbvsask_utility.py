import copy
from collections import defaultdict
import json
import pathlib
import pandas as pd
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import numpy as np
import time
#####################################

DEFAULT_PAR_VALUES_DICT = {'TT': 0.0, 'C0': 5.0, 'ETF': 0.5, 'LP': 0.5, 'FC': 100,
                                        'beta': 2.0, 'FRAC': 0.5, 'K1': 0.5, 'alpha': 2.0, 'K2': 0.025,
                                        'UBAS': 1, 'PM': 1}
HBV_PARAMS_LIST = ['TT', 'C0', 'ETF', 'LP', 'FC',
                   'beta', 'FRAC', 'K1', 'alpha', 'K2',
                   'UBAS', 'PM']

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


def _plot_streamflow_and_precipitation(time_series_data_df,
                                       simulated_flux=None,
                                       simulated_time_column=None,
                                       observed_streamflow_column="streamflow",
                                       simulated_streamflow_column="Q_cms",
                                       precipitation_columns="precipitation",
                                       additional_columns=None):
    if simulated_time_column is None:
        time_series_data_df = time_series_data_df.loc[simulated_flux.index.min():simulated_flux.index.max()]
    else:
        time_series_data_df = time_series_data_df.loc[
                              simulated_flux[simulated_time_column].min():simulated_flux[simulated_time_column].max()]

    N_max = time_series_data_df["precipitation"].max()
    timesteps_min = time_series_data_df.index.min()
    timesteps_max = time_series_data_df.index.max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series_data_df.index,
                             y=time_series_data_df[observed_streamflow_column],
                             name="Observed Streamflow"
                             ))

    if simulated_flux is not None:
        if simulated_time_column is None:
            fig.add_trace(go.Scatter(x=simulated_flux.index,
                                     y=simulated_flux[simulated_streamflow_column],
                                     name="Simulated Streamflow"
                                     ))
        else:
            fig.add_trace(go.Scatter(x=simulated_flux[simulated_time_column],
                                     y=simulated_flux[simulated_streamflow_column],
                                     name="Simulated Streamflow"
                                     ))

    fig.add_trace(go.Scatter(x=time_series_data_df.index, y=time_series_data_df[precipitation_columns],
                             text=time_series_data_df[precipitation_columns], name="Precipitation", yaxis="y2", ))
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
            title="Q []",
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
            title="N [mm/h]",
            type="linear",
        )
    )
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=1.21
    ))
    return fig
#####################################


def read_streamflow(streamflow_inp):
    streamflow_dict = dict()
    with open(streamflow_inp, "r") as file:
        for line in file.readlines():
            line = line.strip()
            date, value = line.split()
            streamflow_dict[date] = float(value)
    streamflow_df = pd.DataFrame.from_dict(streamflow_dict, orient='index', columns=["streamflow", ])
    streamflow_df.index = pd.to_datetime(streamflow_df.index)
    streamflow_df.index.name = 'date'
    return streamflow_df


def read_precipitation_temperature(precipitation_temperature_inp):
    precipitation_temperature_inp_dict = defaultdict(list)
    precipitation_temperature_inp_dict["date"] = []
    precipitation_temperature_inp_dict["precipitation"] = []
    precipitation_temperature_inp_dict["temperature"] = []

    with open(precipitation_temperature_inp, "r") as file:
        for line in file.readlines():
            line = line.strip()
            date, prec, temp = line.split()
            precipitation_temperature_inp_dict["date"].append(date)
            precipitation_temperature_inp_dict["precipitation"].append(float(prec))
            precipitation_temperature_inp_dict["temperature"].append(float(temp))

    precipitation_temperature_df = pd.DataFrame(precipitation_temperature_inp_dict)
    precipitation_temperature_df['date'] = pd.to_datetime(precipitation_temperature_df['date'])
    precipitation_temperature_df.set_index('date', inplace=True)

    return precipitation_temperature_df


def read_initial_conditions(initial_condition_file, return_dict_or_df="dict", timestamp=None, time_column="TimeStamp"):
    if str(initial_condition_file).endswith('.inp'):
        initial_condition_dict = defaultdict(list)
        initial_condition_dict["WatershedArea_km2"] = []
        initial_condition_dict["initial_SWE"] = []
        initial_condition_dict["initial_SMS"] = []
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
            timestamp = initial_condition_df[time_column].min()  # initial_condition_df.loc[0].TimeStamp
        else:
            timestamp = pd.Timestamp(timestamp)
        return initial_condition_df.loc[initial_condition_df[time_column] == timestamp]


def read_long_term_data(monthly_data_inp):
    precipitation_temperature_monthly = defaultdict(list)
    precipitation_temperature_monthly["month"] = []
    precipitation_temperature_monthly["monthly_average_PE"] = []
    precipitation_temperature_monthly["monthly_average_T"] = []
    with open(monthly_data_inp, "r") as file:
        inx = 0
        for line in file.readlines():
            inx += 1
            line = line.strip()
            if len(line.split()) == 2:
                temp, prec = line.split()
                precipitation_temperature_monthly["month"].append(int(inx))
                precipitation_temperature_monthly["monthly_average_PE"].append(float(prec))
                precipitation_temperature_monthly["monthly_average_T"].append(float(temp))
    precipitation_temperature_monthly_df = pd.DataFrame(precipitation_temperature_monthly)
    precipitation_temperature_monthly_df.set_index("month", inplace=True)
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
    *****  SMS: Soil Moisture Storage at time t- model state variable *****
    *****  S1: at time t*****
    *****  S2: at time t*****
    *****  AET: Actual EvapoTranspiration at time t- model *****
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

    Q1 = K1 * (S1 ** alpha)  # TODO make sure that it is not (K1 * S1) ** alpha
    if Q1 > S1:
        Q1 = S1

    S1_new = S1 + soil_release_to_fast_reservoir - Q1

    Q2 = K2 * S2  # TODO

    S2_new = S2 + soil_release_to_slow_reservoir - Q2  # TODO

    return SMS_new, S1_new, S2_new, Q1, Q2


def evapotranspiration_module(SMS, T, monthly_average_T, monthly_average_PE, ETF, LP):
    """
    The function should return AET - Actual EvapoTranspiration at time t- model,
    PET - Potential EvapoTranspiration at time t
    *****  SMS: Soil Moisture Storage at time t- model state variable *****
    *****  T: Temperature at time t- model forcing *****
    *****  monthly_average_T: *****
    *****  monthly_average_PE: *****
    *****  ETF - This is the temperature anomaly correction of potential evapotranspiration - model parameters *****
    *****  LP: This is the soil moisture content below which evaporation becomes supply-limited - model parameter *****
    """
    # Potential Evapotranspiration
    PET = (1 + ETF * (T - monthly_average_T)) * monthly_average_PE
    PET = max(PET, 0)

    if SMS > LP:
        AET = PET
    else:
        AET = PET * (SMS / LP)

    # to avoid evaporating more than water available
    AET = min(AET, SMS)

    return AET, PET


def precipitation_module(SWE, Precipitation, Temperature, TemperatureThreshold, C0):
    """
    The function should return SWE at time t+1, ponding at time t
    *****  SWE: Snow Water Equivalent at time t - model state variable *****
    *****  Precipitation: Precipitation at time t - model forcing *****
    *****  Temperature: Temperature at time t - model forcing *****
    *****  TemperatureThreshold: Temperature Threshold or melting/freezing point - model parameter *****
    *****  C0: base melt factor - model parameter *****
    """
    if Temperature >= TemperatureThreshold:
        rainfall = Precipitation
        potential_snow_melt = C0 * (Temperature - TemperatureThreshold)
        snow_melt = min(potential_snow_melt, SWE)
        # Liquid Water on Surface
        ponding = rainfall + snow_melt
        # Soil Water Equivalent - Solid Water on Surface
        SWE -= snow_melt
    else:
        snowfall = Precipitation
        snow_melt = 0
        # Liquid Water on Surface
        ponding = 0
        # Soil Water Equivalent - Solid Water on Surface
        SWE += snowfall

    return SWE, ponding


def triangle_routing(Q, UBAS):
    """
    The function should return Q_routed - list/1d-array
    *****  Q: list/1d-array *****
    *****  UBAS: Base of unit hydrograph for watershed routing in day; default is 1 for small watersheds *****
    """
    UBAS = max(UBAS, 0.1)
    length_triangle_base = math.ceil(UBAS)

    if UBAS == length_triangle_base:
        x = [0, 0.5 * UBAS, length_triangle_base]
        v = [0, 1, 0]
    else:
        x = [0, 0.5 * UBAS, UBAS, length_triangle_base]
        v = [0, 1, 0, 0]

    weight = np.empty(shape=(length_triangle_base + 1,), dtype=np.float64)
    weight[0] = 0

    # np.interp(2.5, xp, fp) or f = scipy.interpolate.interp1d(x, y); f(xnew)
    for i in range(1, length_triangle_base + 1):
        if (i - 1) < (0.5 * UBAS) and i > (0.5 * UBAS):
            weight[i] = 0.5 * (np.interp(i - 1, x, v) + np.interp(0.5 * UBAS, x, v)) * (0.5 * UBAS - i + 1) + \
                        0.5 * (np.interp(0.5 * UBAS, x, v) + np.interp(i, x, v)) * (i - 0.5 * UBAS)
        elif i > UBAS:
            weight[i] = 0.5 * np.interp(i - 1, x, v) * (UBAS - i + 1)
        else:
            weight[i] = np.interp(i - 0.5, x, v)

    weight = weight / np.sum(weight)

    Q_routed = np.empty_like(Q)

    for i in range(len(Q)):
        temp = 0
        window_len = min(i, length_triangle_base)
        for j in range(window_len):
            temp += weight[j + 1] * Q[i - j - 1]  # TODO Q[i - j]
        Q_routed[i] = temp

    return Q_routed


#####################################
def HBV_SASK(forcing, long_term, par_values_dict, initial_condition_df, printing=False):
    """
    HBV-SASK has 12 parameters: The first 10 ones are necessary
    to run the model, and parameters 11 and 12, if not given,
    will be set at their default values.
    """
    if par_values_dict is None:
        par_values_dict = {
            'TT': 0.0, 'C0': 5.0, 'ETF': 0.5, 'LP': 0.5, 'FC': 100, 'beta': 2.0, 'FRAC': 0.5,
            'K1': 0.5, 'alpha': 2.0, 'K2': 0.025, 'UBAS': 1, 'PM': 1
        }
    try:
        TT = float(par_values_dict["TT"])
        C0 = float(par_values_dict["C0"])
        ETF = float(par_values_dict["ETF"])
        LP = float(par_values_dict["LP"])
        FC = float(par_values_dict["FC"])
        beta = float(par_values_dict["beta"])
        FRAC = float(par_values_dict["FRAC"])
        K1 = float(par_values_dict["K1"])
        alpha = float(par_values_dict["alpha"])
        K2 = float(par_values_dict["K2"])
    except KeyError:
        print(f"Error while reading parameter values from param dictionary!")
        raise

    UBAS = float(par_values_dict.get("UBAS", 1))
    PM = float(par_values_dict.get("PM", 1))

    LP = LP * FC

    watershed_area = initial_condition_df["WatershedArea_km2"].values[0]
    initial_SWE = initial_condition_df["initial_SWE"].values[0]
    initial_SMS = initial_condition_df["initial_SMS"].values[0]
    initial_S1 = initial_condition_df["S1"].values[0]
    initial_S2 = initial_condition_df["S2"].values[0]

    flux = defaultdict(dict)
    state = defaultdict(dict)

    P = PM * forcing.precipitation.to_numpy()
    T = forcing.temperature.to_numpy()

    if "data" in forcing.columns:
        time_series = forcing.data
    else:
        time_series = forcing.index
    #     monthly_average_T = long_term.monthly_average_T.to_numpy()
    #     monthly_average_PE = long_term.monthly_average_PE.to_numpy()

    period_length = len(P)  # P.shape[0]

    if printing:
        print(f" watershed_area={watershed_area}")
        print(f" initial_SWE={initial_SWE} \n initial_SMS={initial_SMS} "
              f"\n initial_S1={initial_S1} \n initial_S2={initial_S2} \n")
        print(f"period_length={period_length}")

    SWE = np.empty(shape=(period_length + 1,), dtype=np.float64)
    SMS = np.empty(shape=(period_length + 1,), dtype=np.float64)
    S1 = np.empty(shape=(period_length + 1,), dtype=np.float64)
    S2 = np.empty(shape=(period_length + 1,), dtype=np.float64)  # np.empty_like(P)
    Q1 = np.empty_like(P)
    Q2 = np.empty_like(P)
    AET = np.empty_like(P)
    PET = np.empty_like(P)
    ponding = np.empty_like(P)  # np.empty(shape=(period_length,), dtype=np.float32)

    SWE[0] = initial_SWE
    SMS[0] = initial_SMS
    S1[0] = initial_S1
    S2[0] = initial_S2

    for t in range(period_length):
        month = time_series[t].month  # the current month number - for Jan=1, ..., Dec=12
        single_monthly_average_PE = long_term.loc[month].monthly_average_PE
        single_monthly_average_T = long_term.loc[month].monthly_average_T

        SWE[t + 1], ponding[t] = precipitation_module(SWE[t], P[t], T[t], TT, C0)

        AET[t], PET[t] = evapotranspiration_module(
            SMS[t], T[t], single_monthly_average_T, single_monthly_average_PE, ETF, LP
        )

        SMS[t + 1], S1[t + 1], S2[t + 1], Q1[t], Q2[t] = soil_storage_routing_module(
            ponding[t], SMS[t], S1[t], S2[t], AET[t], FC, beta, FRAC, K1, alpha, K2
        )

    Q1_routed = triangle_routing(Q1, UBAS)
    Q = Q1_routed + Q2
    Q_cms = (Q * watershed_area * 1000) / (24 * 3600)  # TODO

    flux["Q_cms"] = Q_cms.conjugate()
    flux["Q_mm"] = Q.conjugate()
    flux["AET"] = AET.conjugate()
    flux["PET"] = PET.conjugate()
    flux["Q1"] = Q1.conjugate()
    flux["Q1_routed"] = Q1_routed.conjugate()
    flux["Q2"] = Q2.conjugate()
    flux["ponding"] = ponding.conjugate()

    state["SWE"] = SWE.conjugate()
    state["SMS"] = SMS.conjugate()
    state["S1"] = S1.conjugate()
    state["S2"] = S2.conjugate()

    return flux, state


def parameters_configuration(parameters, configurationObject, take_direct_value=False):
    parameters_dict = copy.deepcopy(DEFAULT_PAR_VALUES_DICT)  # defaultdict()

    if isinstance(parameters, dict) and take_direct_value:
        if parameters is None:
            return DEFAULT_PAR_VALUES_DICT
        parameters_dict = parameters
    else:
        list_of_parameters_from_json = configurationObject["parameters"]
        uncertaint_param_counter = 0
        for id, param_entry_dict in enumerate(list_of_parameters_from_json):
            # this logic should have already been followed in UQsim
            if param_entry_dict["distribution"] == "None":
                if "value" in param_entry_dict:
                    parameters_dict[param_entry_dict["name"]] = param_entry_dict["value"]
                elif "default" in param_entry_dict:
                    parameters_dict[param_entry_dict["name"]] = param_entry_dict["default"]
                else:
                    parameters_dict[param_entry_dict["name"]] = DEFAULT_PAR_VALUES_DICT[param_entry_dict["name"]]
            else:
                if parameters is None:
                    return DEFAULT_PAR_VALUES_DICT
                parameters_dict[param_entry_dict["name"]] = parameters[uncertaint_param_counter]
                uncertaint_param_counter += 1

    return parameters_dict

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


def run_the_model(hbv_model_path, config_file, par_values_dict, run_full_timespan=False, basis='Oldman_Basin',
                  plotting=False, writing_results_to_a_file=False, output_path=None, **kwargs):
    # Preparing paths
    path_to_input = hbv_model_path / basis
    # initial_condition_file = path_to_input / "initial_condition.inp"
    initial_condition_file = path_to_input / "state_df.pkl"
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
    streamflow_df = read_streamflow(streamflow_inp)
    precipitation_temperature_df = read_precipitation_temperature(precipitation_temperature_inp)
    time_series_data_df = pd.merge(streamflow_df, precipitation_temperature_df, left_index=True, right_index=True)
    precipitation_temperature_monthly_df = read_long_term_data(monthly_data_inp)
    param_setup_dict = read_param_setup_dict(factorSpace_txt)

    # Parse input based on some timeframe
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    if "data" in time_series_data_df.columns:
        time_series_data_df = time_series_data_df.loc[
            (time_series_data_df['data'] >= start_date) & (time_series_data_df['data'] <= end_date)]
    else:
        time_series_data_df = time_series_data_df[start_date:end_date]
    # initial_condition_df = read_initial_conditions(initial_condition_file, return_dict_or_df="df")
    initial_condition_df = read_initial_conditions(initial_condition_file, timestamp=start_date, time_column="TimeStamp")
    # print(initial_condition_df)

    if plotting:
        fig = make_subplots(rows=3, cols=1)
        fig.add_trace(
            go.Scatter(x=precipitation_temperature_df.index, y=precipitation_temperature_df.precipitation, name="P"),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=precipitation_temperature_df.index, y=precipitation_temperature_df.temperature, name="T"),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=streamflow_df.index, y=streamflow_df.streamflow, name="Q_cms"),
            row=3, col=1
        )
        plot_filename = output_path / f"forcing_data.html"
        plot(fig, filename=str(plot_filename), auto_open=False)

    #############################

    # Running the model
    flux, state = HBV_SASK(
        forcing=time_series_data_df,
        long_term=precipitation_temperature_monthly_df,
        par_values_dict=par_values_dict,
        initial_condition_df=initial_condition_df,
        printing=True
    )

    # if "data" in time_series_data_df.columns:
    #     time_series = time_series_data_df.data
    # else:
    #     time_series = time_series_data_df.index

    # Create a final df - flux
    time_series_list = list(full_data_range)  # list(simulation_range)
    assert len(list(full_data_range)) == len(flux["Q_cms"])

    flux_df = pd.DataFrame(
        list(zip(time_series_list, flux["Q_cms"], flux["Q_mm"], flux["AET"], flux["PET"], flux["Q1"], flux["Q1_routed"],
                 flux["Q2"], flux["ponding"])),
        columns=['TimeStamp', 'Q_cms', 'Q_mm', 'AET', 'PET', 'Q1', 'Q1_routed', 'Q2', "ponding"]
    )

    # Create a final df - state
    last_date = time_series_list[-1]
    time_series_list.append(pd.to_datetime(last_date) + pd.DateOffset(days=1))
    state_df = pd.DataFrame(
        list(zip(time_series_list, state["SWE"], state["SMS"], state["S1"], state["S2"])),
        columns=['TimeStamp', 'initial_SWE', 'initial_SMS', 'S1', 'S2', ]
    )
    state_df['WatershedArea_km2'] = initial_condition_df["WatershedArea_km2"].values[0]

    # Parse flux_df between start_date_predictions, end_date
    flux_df.set_index('TimeStamp', inplace=True)
    flux_df = flux_df.loc[simulation_range]  # flux_df[start_date_predictions:end_date]

    # Append measured streamflow to flux_df, i.e., merge flux_df and time_series_data_df[streamflow]
    # df3 = pd.merge(flux_df, time_series_data_df[["streamflow"]], left_index=True, right_index=True)
    flux_df = flux_df.merge(time_series_data_df[["streamflow", ]], left_index=True, right_index=True)

    # Parse state_df between start_date_predictions, end_date + 1
    state_df.set_index('TimeStamp', inplace=True)
    state_df = state_df[start_date_predictions:]

    # TODO-Ivana Compute Metrics - from my code and from VARS code

    # reset the index
    flux_df.reset_index(inplace=True)
    flux_df.rename(columns={"index": 'TimeStamp'}, inplace=True)
    state_df.reset_index(inplace=True)
    state_df.rename(columns={"index": 'TimeStamp'}, inplace=True)

    # Write to a file
    if writing_results_to_a_file and output_path is not None:
        file_path = output_path / f"flux_df.pkl"
        flux_df.to_pickle(file_path, compression="gzip")
        file_path = output_path / f"state_df.pkl"
        state_df.to_pickle(file_path, compression="gzip")

    if plotting:
        fig = _plot_streamflow_and_precipitation(
            time_series_data_df,
            simulated_flux=flux_df,
            simulated_time_column="TimeStamp",
            observed_streamflow_column="streamflow",
            simulated_streamflow_column="Q_cms",
            precipitation_columns="precipitation",
            additional_columns=None
        )
        # fig.add_trace(go.Scatter(x=flux_df.index, y=flux_df["Q_cms"], name="Q_cms"))
        plot_filename = output_path / f"hbv_sask_{basis}.html"
        plot(fig, filename=str(plot_filename), auto_open=False)
        # fig.show()

    return flux_df, state_df


if __name__ == "__main__":
    # Path definitions - change them accordingly
    hbv_model_path = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
    basis = 'Oldman_Basin'  # to read in data for the Oldman Basin
    # basis = 'Banff_Basin'  # to read in data for the Banff Basin
    config_file = pathlib.Path("/work/ga45met/mnt/linux_cluster_2/UQEFPP/configurations/configuration_hbv.json")
    output_path = hbv_model_path / basis / "model_runs" / "temp_6"
    output_path.mkdir(parents=True, exist_ok=True)

    # this will overwrite configurations from the json file
    run_full_timespan = False
    plotting = True
    writing_results_to_a_file = True

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
    SM = state["initial_SMS"]
