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
    return result

##############################################################
##### Utility functions for parsing all other lila files and  storing the output timeseries into pandas.DataFrame object
##############################################################

def big_q_lila_parser_toPandas(file_path, index_run=0):
    """
    Function parses all data entries within measured (ground truth) big lila file

    :return pandas.DataFrame
    """
    result_list = []
    # TODO Change this so that any lila file (not just wq) can be read to pandas DataFrame
    #type_of_data = "Ground Truth"
    type_of_data = "Measured"
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

def any_small_lila_parser_toPandas(file_path, index_run=0):
    pass

def small_q_lila_parser_toPandas(file_path, index_run=0):
    pass
