"""
A set of utility functions for parsing some of the Larsim input and output files
and stroting data as Pandas objects

@author: Ivana Jovanovic
"""

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


##############################################################
##### Utility functions for parsing ergebnis files and storing the output timeseries into pandas.DataFrame object
##############################################################

def ergebnis_parser_toPandas(file_path, index_run=0):
    """
    Function parses all data entries within ergebnis.lila and adds them
    to their respective stations in dictionaries

    :return pandas.DataFrame
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
                if "Niederschlag" in line[1]:
                    curr_rtype = "Niederschlag"
                    curr_rtype2 = ""
                elif "Abfluss Messung" in line[1]:
                    curr_rtype = "Abfluss Messung"
                    curr_rtype2 = "Abfluss Simulation"
                elif "Abfluss Simulation" in line[1]:
                    curr_rtype = "Abfluss Simulation"
                else:
                    curr_rtype = line[1]
                    curr_rtype2 = ""

            #if re.match("\d\d\.\d\d\.\d*", line[0]) and (curr_rtype == "Abfluss Messung" or curr_rtype == "Abfluss Simulation"):
            if re.match("\d\d\.\d\d\.\d*", line[0]):
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

    labels = ['Index_run', 'Stationskennung', 'Type', 'TimeStamp', 'Value']
    result = pd.DataFrame.from_records(result_list, columns=labels)
    result['Value'] = result['Value'].astype(float)
    return result



def ergebnisQ_parser_toPandas(file_path, index_run=0):
    """
    Function parses all data entries within ergebnis.lila and adds them
    to their respective stations in dictionaries

    :return pandas.DataFrame
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

    labels = ['Index_run', 'Stationskennung', 'Type', 'TimeStamp', 'Value']
    result = pd.DataFrame.from_records(result_list, columns=labels)
    result['Value'] = result['Value'].astype(float)
    return result

##############################################################
##### Utility functions for parsing all other lila files and  storing the output timeseries into pandas.DataFrame object
##############################################################


def q_lila_parser_toPandas(file_path, index_run=0):
    """
    Function parses all data entries within measured (ground truth) big lila file

    :return pandas.DataFrame
    """
    result_list = []
    # TODO Change this so that any lila file (not just wq) can be read to pandas DataFrame
    type_of_data = "Ground Truth"
    #type_of_data = "Measured"
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
    result['Value'] = result['Value'].astype(float)
    return result


def q_lila_parser_toPandas(file_path, index_run=0):
    """
    Function parses all data entries within measured (ground truth) big lila file

    :return pandas.DataFrame
    """
    result_list = []
    # TODO Change this so that any lila file (not just wq) can be read to pandas DataFrame
    type_of_data = "Ground Truth"
    #type_of_data = "Measured"
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
    result['Value'] = result['Value'].astype(float)
    return result


def any_lila_parser_toPandas(file_path, index_run=0, write_in_file=False):
    """
    Function parses all data entries within lila file
    and returns a pandas.DataFrame object

    :return pandas.DataFrame
    """
    if not os.path.isfile(file_path):
        raise OSError('No such file or directory', file_path)

    result_list = []

    file_name = os.path.basename(file_path)
    if file_name.split('-')[0] == "station":
        parameter = file_name.split('-')[1]
    else:
        parameter = file_name.split('_')[0]

    type_of_data = parameter

    stations_array = []

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
    result['Value'] = result['Value'].astype(float)

    if write_in_file:
        # Save complete panda object
        new_name = os.path.splitext(file_path)[0] + ".pkl"
        result.to_pickle(new_name)

    return result


def timevalues_lila_parser_toPandas(file_path, stations_dict, write_in_file=True):
    """
    This function transforms lila time-values file into the pandas object
    One can filter out NaN/missing values and save the DataFrame object

    Return: pandas.DataFrame object
    """
    if not os.path.isfile(file_path):
        raise OSError('No such file or directory', file_path)

    # Read the lila file, parse it, create pandas.DataFrame object of it and pickle it
    file_name = os.path.basename(file_path)
    if file_name.split('-')[0] == "station":
        parameter = file_name.split('-')[1]
    else:
        parameter = file_name.split('_')[0]
    curr_stations = stations_dict[parameter]  # array containing all measuring stations for the parameter
    names = ["TimeStamp"] + curr_stations
    parse_dates = ["TimeStamp"]
    dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')
    dtype_dict = {key: np.float64 for key in curr_stations}
    df = pd.read_csv(file_path, sep=';', header=None, names=names, dtype=dtype_dict, na_values=['-'],
                     parse_dates=parse_dates, date_parser=dateparse)

    # df["TimeStamp"] =  pd.to_datetime(df["TimeStamp"], format='%d.%m.%Y %H:%M')

    if write_in_file:
        # Save complete panda object
        new_name = os.path.splitext(file_path)[0] + ".pkl"
        df.to_pickle(new_name)

    return df




