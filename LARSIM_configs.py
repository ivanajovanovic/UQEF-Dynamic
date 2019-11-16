import csv
import datetime
from decimal import Decimal
from distutils.util import strtobool
from glob import glob
import json
import pandas as pd
import re #for regular expressions
import os
import os.path as osp
import numpy as np
import subprocess
import time

#from paths import *


##############################################################
##### Utility functions for time settings and calculation of different time variables
##############################################################

def datetime_parse(configuration):
    """
    Function which reads json configuration file and determins the start end end date of the simulation
    """

    data = configuration["Timeframe"]
    start_date = datetime.datetime(data["start_year"],data["start_month"], data["start_day"],
                                   data["start_hour"], data["start_minute"])

    end_date = datetime.datetime(data["end_year"], data["end_month"], data["end_day"],
                                 data["end_hour"], data["end_minute"])
    return [start_date, end_date]

def tape10_configurations(timeframe, master_tape10_file, new_path):

    parameter = ["EREIGNISBEGINN       ", "EREIGNISENDE       ", "VORHERSAGEBEGINN     ", "VORHERSAGEDAUER      "]

    ereignisbeginn = "EREIGNISBEGINN       " + str(timeframe[0].day).zfill(2) + " " + str(timeframe[0].month).zfill(
        2) + " " + str(timeframe[0].year) + " " + str(timeframe[0].hour).zfill(2) + " " + str(timeframe[0].minute).zfill(2) + "\n"

    ereignisende = "EREIGNISENDE         " + str(timeframe[1].day).zfill(2) + " " + str(timeframe[1].month).zfill(
        2) + " " + str(timeframe[1].year) + " " + str(timeframe[1].hour).zfill(2) + " " + str(timeframe[1].minute).zfill(2) + "\n"

    vorhersagebeginn = "VORHERSAGEBEGINN     53\n" # this is set by default, might be changed as well... #TODO Change this
    changes = [ereignisbeginn, ereignisende, vorhersagebeginn]
    #print(changes)

    def replace(master_tape10_file, new_path, parameter, changes):
        i = 0
        with open(master_tape10_file, "r", encoding="ISO-8859-1") as f_in:
            with open(new_path, "w", encoding="ISO-8859-1") as f_out:
                for line in f_in:
                    for params in parameter:
                        if params in line:
                            new_line = changes[i]
                            i += 1
                            break
                        else:
                            new_line = line
                    f_out.write(line.replace(line, new_line))

    # calculates duration of prediction based on date settings in tape10, adds it to tape10
    def calc_duration(changes, timeframe):

        temp = changes[0].split(" ")
        #date_begin = datetime.datetime(int(temp[9]), int(temp[8]), int(temp[7]), int(temp[10]), int(temp[11]))
        date_begin = timeframe[0]
        temp = changes[1].split(" ")
        #date_end = datetime.datetime(int(temp[11]), int(temp[10]), int(temp[9]), int(temp[12]), int(temp[13]))
        date_end = timeframe[1]
        total_time = date_end - date_begin
        total_time = int(total_time.total_seconds() / 3600)
        temp = changes[2].split(" ")
        beginn = int(temp[5])
        changes.append(parameter[3] + str(total_time - beginn) + "\n")

    calc_duration(changes, timeframe)
    print(changes)
    replace(master_tape10_file, new_path, parameter, changes)


#calculates number of timesteps set in tape10 configuration - one way by using VORHERSAGEDAUER
def tape10_timesteps(tape10_path):
    a = 0
    b = 0
    #interval = 0
    with open(tape10_path, "r", encoding="ISO-8859-1") as tape:
        for lines in tape:
            line = lines.split(" ")
            if line[0] == "INTERVALLAENGE":
                interval = float(line[7])
            if line[0] == "VORHERSAGEBEGINN":
                a = float(line[5])
            if line[0] == "VORHERSAGEDAUER":
                b = float(line[6])

    grid_size = int((a+b) / interval) + 1
    t = [i * interval for i in range(grid_size)]

    return t

#calculates number of timesteps set in tape10 configuration - the second way by using difference in start and end time from tape10
def timeArray_of_tape10_timesteps(timeframe, interval=1):
    start_timestep = timeframe[0]
    end_timestep = timeframe[1]
    dateTimeDifference = end_timestep - start_timestep
    dateTimeDifferenceInHours = int(dateTimeDifference.total_seconds() / 3600)

    t = [i * interval for i in range(dateTimeDifferenceInHours)]
    return t

##############################################################
##### Utility functions for finding & filtering & parsing input and initial state files
##############################################################

# Filter out whm files
def copy_whm_files(timeframe, all_whms_path, new_path):
    # list all the whm files
    all_whm_fils = []
    for root, dirs, files in os.walk(all_whms_path):
        all_whm_fils.extend(glob(os.path.join(root, '*.whm')))
    all_whm_fils.sort()
    all_whm_basename = [osp.basename(one_whm_path) for one_whm_path in all_whm_fils]
    #print(all_whm_basename)

    # filter only whms between two dates
    new_whms_files = []
    start_date_min_3 = timeframe[0] - datetime.timedelta(days=3)
    day_count = (timeframe[1] - start_date_min_3).days + 1
    for single_date in (start_date_min_3 + datetime.timedelta(n) for n in range(day_count)):
        single_whm = str(single_date.year) + str(single_date.month).zfill(2) + str(single_date.day).zfill(2) + "00.whm"
        if osp.exists(osp.abspath(os.path.join(all_whms_path, single_whm))):
            new_whms_files.append(single_whm)
    print(new_whms_files)

    # copy
    #for idx, one_whm_file in enumerate(all_past_whm_fils):
    #    call(["cp " + one_whm_file + " " + new_path], shell=True)
    #    subprocess.run(["mkdir", outputResultDir])
    for idx, one_whm_file in enumerate(new_whms_files):
        #call(["cp " + os.path.abspath(os.path.join(all_whms_path, one_whm_file)) + " " + os.path.abspath(os.path.join(new_path, one_whm_file))], shell=True)
        subprocess.run(["cp", os.path.abspath(os.path.join(all_whms_path, one_whm_file)),os.path.abspath(os.path.join(new_path, one_whm_file))])


# Parse big lila files and create small ones
# TODO Do this in more elegant way - using pandas
def master_lila_parser_on_time_crete_new(timeframe, master_lila_paths, new_lila_paths):
    """
    This function samples out ALL big lila files based on the time interval

    Args:
    timeframe - is tuple with begin and end of interval
    form of the date+time = dd.mm.yyyy hh:mm;

    """
    #configured_lila_paths = [os.path.abspath(os.path.join(new_path, i)) for i in new_lila_files]
    start_date_min_3 = timeframe[0] - datetime.timedelta(days=3)
    interval_of_interest = (str(start_date_min_3.day).zfill(2) + "." + str(start_date_min_3.month).zfill(2) + "." +
                            str(start_date_min_3.year) + " " + str(start_date_min_3.hour).zfill(2) + ":" + str(start_date_min_3.minute).zfill(2),

                str(timeframe[1].day).zfill(2) + "." + str(timeframe[1].month).zfill(2) + "." + str(timeframe[1].year) + " " +
                            str(timeframe[1].hour).zfill(2) + ":" + str(timeframe[1].minute).zfill(2))

    i = 0
    for idx, file_path in enumerate(master_lila_paths):
        with open(master_lila_paths[idx], "r", encoding="ISO-8859-1") as f:
            with open(new_lila_paths[idx], "w", encoding="ISO-8859-1") as out:
                while 1:
                    for lines in f:
                        line = lines.split(";")
                        out.writelines(lines)
                        if line[0] == "Kommentar":
                            break
                    break
                for lines in f:
                    line = lines.split(";")
                    if line[0] == interval_of_interest[0]: i = 1
                    if line[0] == interval_of_interest[1]:
                        out.writelines(lines)
                        i = 0
                    if i == 1: out.writelines(lines)
    print('You have successfully parsed master lila files based on input timespan')


def var_limits(var, limits):
    if var < limits[0]: var = limits[0]
    elif var > limits[1]: var = limits[1]
    return var


# tape 35 changes
def tape35_configurations(parameters, curr_working_dir, configurationObject, TGB=None, addSampledValue=True):
    #with open("configurations.json") as f:
    #    data = json.load(f)


    variable_names = []
    limits = []
    for i in configurationObject["Variables"]:
        variable_names.append(i["name"])
        limits.append((i["lower_limit"], i["upper_limit"]))

    # Check if local tape35 file exists
    #while not os.path.exists(curr_working_dir+"/tape35"):
    #    time.sleep(1)
    if not os.path.exists(curr_working_dir+"/tape35"):
        raise IOError('File does not exist: %s. %s' % (curr_working_dir+"/tape35", IOError.strerror))

    tape = pd.read_csv(curr_working_dir+"/tape35", index_col=False, delimiter=";")
    tape.loc[:, "Unnamed: 32"] = ""
    tape.rename(columns={"Unnamed: 32": ""}, inplace=True)
    # Skip strip cause it might confuse Larsim afterwards
    #tape.rename(columns=lambda x: x.strip(), inplace=True)


    #TODO Thnik if this should go outside - in a LarsimModel.run()
    try:
        TGB = int(configurationObject["GaugeControlRegion"]["TGB"])
        addSampledValue = strtobool(configurationObject["GaugeControlRegion"]["addSampledValue"])
    except KeyError:
        TGB = None
        addSampledValue = True


    # changes tape35 entries: original entry + added value
    if TGB is not None:
        if addSampledValue:
            for j in range(0, len(variable_names)):
                tape.loc[tape['   TGB'] == TGB, variable_names[j]] = round(Decimal(var_limits((tape.loc[tape['   TGB'] == TGB, variable_names[j]] + parameters[j]), limits[j])), 2)
        else:
            for j in range(0, len(variable_names)):
                tape.loc[tape['   TGB'] == TGB, variable_names[j]] = round(Decimal(var_limits(parameters[j], limits[j])), 2)
    else:
        if addSampledValue:
            for j in range(0, len(variable_names)):
                for i in range(0, len(tape[variable_names[0]])):
                    tape.loc[i, variable_names[j]] = round(Decimal(var_limits((tape.loc[i, variable_names[j]] + parameters[j]), limits[j])), 2)
        else:
            for j in range(0, len(variable_names)):
                for i in range(0, len(tape[variable_names[0]])):
                    tape.loc[i, variable_names[j]] = round(Decimal(var_limits(parameters[j], limits[j])), 2)



    tape.to_csv(curr_working_dir+"/tape35", index=False, sep=";")

##############################################################
##### Utility functions for parsing ergebnis files and storing the output timeseries into pandas.DataFrame object
##############################################################

def result_parser_toPandas(file_path, index_run = 0):
    """
    Function parses all data entries within ergebnis.lila and adds them
    to their respective stations in dictionaries

    Returns pandas DataFrame.
    """
    result_list = []
    curr_ident = ""
    curr_rtype = ""
    curr_rtype2 = ""

    if not os.path.isfile(file_path):
        raise OSError('No such file or directory', file_path)

    with open(file_path, "r", encoding="ISO-8859-15") as ergebnis_file:
        readCSV = csv.reader(ergebnis_file, delimiter=";")
        for line in readCSV:
            if line[0] == "Stationskennung":
                curr_ident = line[1]

            #TODO Change this - examine Datenursprung
            if line[0] == "Kommentar":
                if "Abfluss Messung" in line[1]:
                    curr_rtype = "Abfluss Messung"
                    curr_rtype2 = "Abfluss Simulation"
                else:
                    #if line[1] == "Abfluss Simulation + Vorhersage ohne ARIMA":
                    if "Abfluss Simulation" in line[1]:
                        curr_rtype = "Abfluss Simulation"
                    else:
                        curr_rtype = line[1]
                        curr_rtype2 = ""

            if re.match("\d\d\.\d\d\.\d*", line[0]) and (
                    curr_rtype == "Abfluss Messung" or curr_rtype == "Abfluss Simulation"):
                timestemp = pd.datetime.strptime(line[0], '%d.%m.%Y %H:%M')

                if (line[1] != "-") and (line[1] != "-\n"):
                    result_list.append((int(index_run), curr_ident, curr_rtype, timestemp, float(line[1])))
                else:
                    result_list.append((int(index_run), curr_ident, curr_rtype, timestemp, None))

                if curr_rtype2 != "":
                    if (line[1] != "-") and (line[1] != "-\n"):
                        result_list.append((int(index_run), curr_ident, curr_rtype2, timestemp, float(line[2])))
                    else:
                        result_list.append((int(index_run), curr_ident, curr_rtype2, timestemp, None))

    labels = ['Index_run','Stationskennung', 'Type', 'TimeStamp', 'Value']
    result = pd.DataFrame.from_records(result_list, columns=labels)
    return result


def lila_parser_toPandas(file_path, index_run=0):
    """
    Function parses all data entries within measured lila file

    Returns pandas DataFrame.
    """
    result_list = []
    # TODO Change this so that any lila file (not just wq) can be read to pandas DataFrame
    type_of_data = "Ground Truth"
    stations_array = []

    if not os.path.isfile(file_path):
        raise OSError('No such file or directory', file_path)

    with open(file_path, "r", encoding="ISO-8859-15") as file:
        while 1:
            for lines in file:
                line = lines.split(";")
                if line[0] == "Stationskennung":
                    # make an array of all the stations
                    for idx, val in enumerate(line[1:]):
                        stations_array.append(val.rstrip('\n'))
                    #print(stations_array)
                    #print(len(stations_array))
                if line[0] == "Kommentar":
                    break
            break
        for lines in file:
            line = lines.split(";")
            if re.match("\d\d\.\d\d\.\d*", line[0]):
                timestemp = pd.datetime.strptime(line[0], '%d.%m.%Y %H:%M')
            # iterate through the rest of line array
            for idx, val in enumerate(line[1:]):
                if (val == "-") or (val == "-\n"):
                    result_list.append((int(index_run), stations_array[idx], type_of_data, timestemp, None))
                else:
                    result_list.append((int(index_run), stations_array[idx], type_of_data, timestemp, float(val)))

    labels = ['Index_run', 'Stationskennung', 'Type', 'TimeStamp', 'Value']
    result = pd.DataFrame.from_records(result_list, columns=labels)
    return result

##############################################################
##### Utility functions for deleteing files - cleaning folders for a clea start, etc.
##############################################################

def delete_larsim_output_files(curr_directory):

    result_file_path = os.path.abspath(os.path.join(curr_directory, 'ergebnis.lila'))
    larsim_ok_file_path = os.path.abspath(os.path.join(curr_directory, 'larsim.ok'))
    tape11_file_path = os.path.abspath(os.path.join(curr_directory, 'tape11'))
    karte_path = os.path.abspath(os.path.join(curr_directory, 'karten'))  # curr_working_dir + 'karten/*'
    tape10_path = os.path.abspath(os.path.join(curr_directory, 'tape10'))

    subprocess.run(["rm", "-f", result_file_path])
    subprocess.run(["rm", "-f", larsim_ok_file_path])
    subprocess.run(["rm", "-f", tape11_file_path])

    # subprocess.run(["rm", "-R", karte_path])
    # if not os.path.isdir(karte_path):
    #    subprocess.run(["mkdir", karte_path])

def _delete_larsim_lila_whm_files_2(curr_directory="./"):

    for single_file in glob(curr_directory + "/*.whm"):
        subprocess.run(["rm", single_file])
    for single_file in glob(curr_directory + "/*.lila"):
        subprocess.run(["rm", single_file])

    print("[LARSIM CONFIGURATION INFO] Cleaning - You have just deleted all .whm and .lila files from the working folder")

def delete_larsim_lila_whm_files():

    subprocess.run(["rm", "*.whm"])
    subprocess.run(["rm", "*.lila"])

def _delete_larsim_lila_whm_files(curr_directory="./"):

    os.chdir(curr_directory)
    subprocess.run(["rm", "-f", "*.whm"])
    subprocess.run(["rm", "-f", "*.lila"])
    os.chdir(osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe()))))

    print("[LARSIM CONFIGURATION INFO] Cleaning - You have just deleted all .whm and .lila files from the working folder")
##############################################################
##### Utility functions for working with pandas DataFrame created from Larsim Lila files - post-processing
##############################################################


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
def calculateGoodnessofFit(measuredDF, predictedDF, station="all", type_of_output_of_Interest="Abfluss Messung", dailyStatisict=False):
    result_dictionary = {}
    #calulcate statistics for all the stations

    if station == "all":
        station_predicted = predictedDF["Stationskennung"].unique()
        station_measured = measuredDF["Stationskennung"].unique()
        station = list(set(station_predicted).intersection(station_measured))

    if isinstance(station, list):
        #Iterate over STATIONS
        for particulaStation in station:
            result_tuple_ForSingleStation = _calculateGoodnessofFit_ForSingleStation(measuredDF, predictedDF, station=particulaStation, type_of_output_of_Interest=type_of_output_of_Interest, dailyStatisict=dailyStatisict)
            result_dictionary[particulaStation] = result_tuple_ForSingleStation

    #calulcate statistics for a particular station
    else:
        result_tuple_ForSingleStation = _calculateGoodnessofFit_ForSingleStation(measuredDF, predictedDF, station=station, type_of_output_of_Interest=type_of_output_of_Interest, dailyStatisict=dailyStatisict)
        result_dictionary[station] = result_tuple_ForSingleStation
    return result_dictionary

def _calculateGoodnessofFit_ForSingleStation(measuredDF, predictedDF, station="MARI", type_of_output_of_Interest="Abfluss Messung", dailyStatisict=False):

    #filter out particular station and data types from the bigger dataFrames
    streamflow_gt_currentStation = measuredDF.loc[
        (measuredDF['Stationskennung'] == station) & (measuredDF['Type'] == "Ground Truth")]

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

    return {"RMSE": rmse, "BIAS":bias, "NSE":nse, "LogNSE":logNse}
