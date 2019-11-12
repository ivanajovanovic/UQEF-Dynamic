"""
A set of Utility functions for working with pandas DataFrame created from Larsim Lila files - post-processing
Everything here is havily depended on the way DataFrames were initally created in larsimInputOutputUtilities

@author: Ivana Jovanovic
"""


import csv
import datetime
from decimal import Decimal
from distutils.util import strtobool
from glob import glob
import json
import pandas as pd
import pandas_profiling
import re #for regular expressions
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import time


def profile_df_from_file(df_file_path):
    df = pd.read_pickle(df_file_path)
    profile_df(df)

    #TODO Delete afterwards when you add plotting utility functions
    df.count()
    mean_per_stations = []
    column_names = []
    for column in df:
        mean_per_stations.append(df[column].mean())
        column_names.append(column)
    max_mean_station = column_names[mean_per_stations.index(max(mean_per_stations))]
    print(max_mean_station)
    df[max_mean_station]
    plt.close('all')
    plt.figure()
    df[max_mean_station].plot()
    df.plot(subplots=True, use_index=True, legend=False, sharex=True, sharey=True, kind='line', figsize=(20, 20))



def profile_df(df, rejecterTr = 0.9, write_to_file=True, path_to_write ='./'):
    # df['TimeStamp'] = df['TimeStamp'].astype('datetime64[ns]')
    df['TimeStamp'] = df['TimeStamp'].apply(lambda x: pd.Timestamp(x))
    df['Value'] = df['Value'].astype(float)
    # df.set_index("TimeStamp", drop=True, inplace=True)
    profile = pandas_profiling.ProfileReport(df)
    rejected_variables = profile.get_rejected_variables(threshold=rejecterTr)

    # Save report to file
    if write_to_file:
        file_name = str(df['Type']) + "profile.html"
        profile_file_path = osp.abspath(osp.join(path_to_write, file_name))
        profile.to_file(outputfile=profile_file_path)

def concat_2_df(df1, df2):
    return pd.concat([df1, df2], axis=1, ignore_index=False, verify_integrity=False, sort=False)

def parse_df_based_on_time(df, interval_of_interest):
    """
    This function filters out only the values of the input object
    which are were measured during the interval_of_interest

    Input: interval_of_interest - tuple of the fome (start_data, end_date)
    start_data and end_date format - '%d.%m.%Y  %H:%M'

    Return: pandas.DataFrame object
    """
    if interval_of_interest is None:
        # raise ValueError('Error - sampled timeStamp+value file does not exist')
        print("Error - you should have specify interval of interest")
        return

    start_date = datetime.datetime.strptime(interval_of_interest[0], '%d.%m.%Y  %H:%M')
    end_date = datetime.datetime.strptime(interval_of_interest[1], '%d.%m.%Y  %H:%M')

    # Parse panda object based on time values
    df['TimeStamp'] = df['TimeStamp'].apply(lambda x: pd.Timestamp(x))
    #df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], format='%d.%m.%Y %H:%M')
    df = df.loc[(df["TimeStamp"] > start_date) & (df["TimeStamp"] <= end_date)]
    #mask = (df['TimeStamp'] >= start_date) & (df['TimeStamp'] <= end_date)
    #df = df.loc[mask, :]

    return df


def filter_nan_from_pandas(df, filter_nan_percentage=0.2, write_in_file=True, file_path=None):
    # Filter out stations/columns with > filter_nan_percentage*100% of NaN values
    df = df.loc[:, df.isnull().mean() < filter_nan_percentage]

    if write_in_file:
        new_name = os.path.splitext(file_path)[0] + "_no_nan" + ".pkl"
        df.to_pickle(new_name)

    return df


def transformToDailyResolution(resultsDataFrame):
    resultsDataFrame_Daily = resultsDataFrame.copy(deep=True)
    resultsDataFrame_Daily['TimeStamp_Date'] = [entry.date() for entry in resultsDataFrame_Daily['TimeStamp']]
    resultsDataFrame_Daily['TimeStamp_Time'] = [entry.time() for entry in resultsDataFrame_Daily['TimeStamp']]
    resultsDataFrame_Daily = resultsDataFrame_Daily.groupby(['Stationskennung', 'Type', 'TimeStamp_Date', 'Index_run'])[
        'Value'].mean().reset_index()
    resultsDataFrame_Daily = resultsDataFrame_Daily.rename({'TimeStamp_Date': 'TimeStamp'}, axis='columns')
    resultsDataFrame_Daily['TimeStamp'] = resultsDataFrame_Daily['TimeStamp'].apply(lambda x: pd.Timestamp(x))
    return resultsDataFrame_Daily


def filterResultForStationAndTypeOfOutpu(resultsDataFrame, station="MARI", type_of_output='Abfluss Messung'):
    resultsDataFrame = resultsDataFrame.loc[
        (resultsDataFrame['Stationskennung'] == station) &
        (resultsDataFrame['Type'] == type_of_output)]
    return resultsDataFrame


def filterResultForStation(resultsDataFrame, station="MARI"):
    resultsDataFrame = resultsDataFrame.loc[resultsDataFrame['Stationskennung'] == station]
    return resultsDataFrame


def filterResultForTypeOfOutpu(resultsDataFrame, type_of_output='Abfluss Messung'):
    resultsDataFrame = resultsDataFrame.loc[resultsDataFrame['Type'] == type_of_output]
    return resultsDataFrame


def align_dataFrames_timewise(biggerDF, smallerDF):

    start_date, end_date = smallerDF.TimeStamp.values[0], smallerDF.TimeStamp.values[-1]
    mask = (biggerDF['TimeStamp'] >= start_date) & (biggerDF['TimeStamp'] <= end_date)
    biggerDF_aligned = biggerDF.loc[mask, :]
    #assert len(biggerDF_aligned['TimeStamp'].unique()) == len(smallerDF['TimeStamp'].unique())

    return biggerDF_aligned


def calculateRMSE(measuredDF, simulatedDF):
    squared_error = np.square(
        np.subtract(measuredDF.Value.values, simulatedDF.Value.values))
    rmse = np.sqrt(np.mean(squared_error))
    return rmse


def calculateBIAS(measuredDF, simulatedDF):
    residual = np.subtract(measuredDF.Value.values, simulatedDF.Value.values)
    bias = np.abs(np.mean(residual))
    return bias


def calculateNSE(measuredDF, simulatedDF):
    squared_error = np.square(
        np.subtract(measuredDF.Value.values, simulatedDF.Value.values))
    squared_error_to_mean = np.square(
        np.subtract(simulatedDF.Value.values, np.mean(measuredDF.Value.values)))
    nse = 1 - np.sum(squared_error) / np.sum(squared_error_to_mean)
    return nse


def calculateLogNSE(measuredDF, simulatedDF):
    squared_error = np.square(
        np.subtract(np.log(measuredDF.Value.values), np.log(simulatedDF.Value.values)))
    squared_error_to_mean = np.square(
        np.subtract(np.log(simulatedDF.Value.values), np.mean(np.log(measuredDF.Value.values))))
    logNSE = 1 - np.sum(squared_error) / np.sum(squared_error_to_mean)
    return logNSE


def calculateBraviasPearson(measuredDF, simulatedDF):
    mean_measured = np.mean(measuredDF.Value.values)
    mean_simulated = np.mean(simulatedDF.Value.values)
    term1 = np.subtract(measuredDF.Value.values, mean_measured)
    term2 = np.subtract(simulatedDF.Value.values, mean_measured)
    term3 = np.subtract(simulatedDF.Value.values, mean_simulated)
    term4 = np.prod(term1, term2)
    term5 = np.square(term1)
    term6 = np.square(term3)
    r_2 = np.square(np.sum(term4))/(np.sum(term5)*np.sum(term6))
    return r_2


# TODO remove this - seems as it is wrong formula
def calculateNSE2(measuredDF, simulatedDF):
    squared_error = np.square(
        np.subtract(measuredDF.Value.values, simulatedDF.Value.values))
    n = len(measuredDF.Value.values)
    denumerator = np.sum(np.square(simulatedDF.Value.values)) - np.square(np.sum(measuredDF.Value.values)) / n
    nse = 1 - np.sum(squared_error) / denumerator
    return nse


#TODO Choose which GOF do I want to compute
def calculateGoodnessofFit(measuredDF, predictedDF, station="MARI", type_of_output_of_Interest_measured="Ground Truth", type_of_output_of_Interest="Abfluss Messung", dailyStatisict=False):
    result_dictionary = {}
    #calulcate statistics for all the stations
    if isinstance(station, list):
        #Iterate over STATIONS
        for particulaStation in station:
            result_tuple_ForSingleStation = _calculateGoodnessofFit_ForSingleStation(measuredDF, predictedDF, station=particulaStation, type_of_output_of_Interest_measured=type_of_output_of_Interest_measured, type_of_output_of_Interest=type_of_output_of_Interest, dailyStatisict=dailyStatisict)
            result_dictionary[particulaStation] = result_tuple_ForSingleStation

    #calulcate statistics for a particular station
    else:
        result_tuple_ForSingleStation = _calculateGoodnessofFit_ForSingleStation(measuredDF, predictedDF, station=station, type_of_output_of_Interest_measured=type_of_output_of_Interest_measured, type_of_output_of_Interest=type_of_output_of_Interest, dailyStatisict=dailyStatisict)
        result_dictionary[station] = result_tuple_ForSingleStation
    return result_dictionary


def _calculateGoodnessofFit_ForSingleStation(measuredDF, predictedDF, station="MARI", type_of_output_of_Interest_measured="Ground Truth", type_of_output_of_Interest="Abfluss Messung", dailyStatisict=False):

    #filter out particular station and data types from the bigger dataFrames
    streamflow_gt_currentStation = measuredDF.loc[
        (measuredDF['Stationskennung'] == station) & (measuredDF['Type'] == type_of_output_of_Interest_measured)]

    streamflow_predicted_currentStation = predictedDF.loc[
        (predictedDF['Stationskennung'] == station) & (predictedDF['Type'] == type_of_output_of_Interest)]


    if dailyStatisict:
        # on daily basis
        streamflow_gt_currentStation_daily = transformToDailyResolution(streamflow_gt_currentStation)
        streamflow_predicted_currentStation_daily =transformToDailyResolution(streamflow_predicted_currentStation)

        #DataFrame containing measurements might be longer than the one containing model predictions - alignment is needed
        streamflow_gt_currentStation_daily_aligned = align_dataFrames_timewise(biggerDF=streamflow_gt_currentStation_daily, smallerDF=streamflow_predicted_currentStation_daily)

        #calculate mean of the observed - measured discharge
        #mean_gt_discharge = np.mean(streamflow_gt_currentStation_daily_aligned.Value.values)

        #RMSE
        rmse = calculateRMSE(measuredDF=streamflow_gt_currentStation_daily_aligned, simulatedDF=streamflow_predicted_currentStation_daily)

        #BIAS
        bias = calculateBIAS(measuredDF=streamflow_gt_currentStation_daily_aligned, simulatedDF=streamflow_predicted_currentStation_daily)

        #NSE
        nse = calculateNSE(measuredDF=streamflow_gt_currentStation_daily_aligned, simulatedDF=streamflow_predicted_currentStation_daily)

        # NSE Calculation type 2
        logNse = calculateLogNSE(measuredDF=streamflow_gt_currentStation_daily_aligned, simulatedDF=streamflow_predicted_currentStation_daily)

    else:
        # on hourly basis
        #DataFrame containing measurements might be longer than the one containing model predictions - alignment is needed
        streamflow_gt_currentStation_aligned = align_dataFrames_timewise(biggerDF=streamflow_gt_currentStation, smallerDF=streamflow_predicted_currentStation)

        #calculate mean of the observed - measured discharge
        #mean_gt_discharge = np.mean(streamflow_gt_currentStation_aligned.Value.values)

        #RMSE
        rmse = calculateRMSE(measuredDF=streamflow_gt_currentStation_aligned, simulatedDF=streamflow_predicted_currentStation)

        #BIAS
        bias = calculateBIAS(measuredDF=streamflow_gt_currentStation_aligned, simulatedDF=streamflow_predicted_currentStation)

        #NSE
        nse = calculateNSE(measuredDF=streamflow_gt_currentStation_aligned, simulatedDF=streamflow_predicted_currentStation)

        # NSE Calculation type 2
        logNse = calculateLogNSE(measuredDF=streamflow_gt_currentStation_aligned, simulatedDF=streamflow_predicted_currentStation)

    return (rmse, bias, nse, logNse)