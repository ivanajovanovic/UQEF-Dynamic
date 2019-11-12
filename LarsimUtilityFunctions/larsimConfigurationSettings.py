"""
A set of utility functions for the proper setting of the environment
for running Larsim simulation

@author: Ivana Jovanovic
"""

import csv
import datetime
from decimal import Decimal
from distutils.util import strtobool
from glob import glob
import json
import inspect
import linecache
import pandas as pd
import re #for regular expressions
import os
import os.path as osp
import numpy as np
import subprocess
import time


##############################################################
##### Utility functions for deleteing files - cleaning folders for a clea start, etc.
##############################################################

def cleanDirecory(curr_directory="./"):
    _delete_larsim_output_files(curr_directory)
    _delete_larsim_lila_whm_files_2(curr_directory)


def _delete_larsim_output_files(curr_directory="./"):

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


def _delete_larsim_lila_whm_files(curr_directory="./"):

    os.chdir(curr_directory)
    subprocess.run(["rm", "*.whm"])
    subprocess.run(["rm", "*.lila"])
    os.chdir(osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe()))))


def _delete_larsim_lila_whm_files_2(curr_directory="./"):

    for single_file in glob.glob(curr_directory + "/*"):
        if single_file.endswith(".whm") or single_file.endswith(".lila"):
            subprocess.run(["rm", "-r", single_file])
        else:
            pass


def check_larsim_ok_file(curr_working_dir):
    """
    check for existence of larsim.ok and whether the file is readable
    Important to be done before data postprocessing
    """
    larsim_ok = False

    if os.path.isfile(curr_working_dir):
        larsim_ok_file = curr_working_dir
    else:
        larsim_ok_file = curr_working_dir + "/larsim.ok"

    while larsim_ok is False:
        lines = linecache.getlines(larsim_ok_file)

        for l in lines:
            if "ok" in l:
                larsim_ok = True

        linecache.clearcache()
        time.sleep(0.1)
        print("[LARSIM INFO] retries reading larsim_ok")
##############################################################
##### WHM Files
##############################################################


def copy_whm_files(timeframe, all_whms_path="./", new_path="./",start_date_min_3_bool=False):
    """
    Filter out whm files based on timeframe

    :param
    :return
    """
    # list all the whm files
    all_whm_fils = []
    for root, dirs, files in os.walk(all_whms_path):
        all_whm_fils.extend(glob(os.path.join(root, '*.whm')))
    all_whm_fils.sort()
    all_whm_basename = [osp.basename(one_whm_path) for one_whm_path in all_whm_fils]
    #print(all_whm_basename)

    # filter only whms between two dates
    new_whms_files = []
    if start_date_min_3_bool:
        start_date = timeframe[0] - datetime.timedelta(days=3)
    else:
        start_date = timeframe[0]

    day_count = (timeframe[1] - start_date).days + 1
    for single_date in (start_date + datetime.timedelta(n) for n in range(day_count)):
        single_whm = str(single_date.year) + str(single_date.month).zfill(2) + str(single_date.day).zfill(2) + "00.whm"
        if osp.exists(osp.abspath(os.path.join(all_whms_path, single_whm))):
            new_whms_files.append(single_whm)
    print("[LARSIM CONFIGURATION INFO] You've just filtered out the following .whm files: \n {}".format(new_whms_files))

    if all_whms_path!=new_path:
        for idx, one_whm_file in enumerate(new_whms_files):
            #call(["cp " + os.path.abspath(os.path.join(all_whms_path, one_whm_file)) + " " + os.path.abspath(os.path.join(new_path, one_whm_file))], shell=True)
            subprocess.run(["cp", os.path.abspath(os.path.join(all_whms_path, one_whm_file)),os.path.abspath(os.path.join(new_path, one_whm_file))])


##############################################################
##### Tape35
##############################################################

def _var_limits(var, limits):
    if var < limits[0]: var = limits[0]
    elif var > limits[1]: var = limits[1]
    return var


def tape35_clean(tape35_path):
    # Check if local tape35 file exists
    if not os.path.exists(tape35_path):
        raise IOError('File does not exist: %s. %s' % (tape35_path, IOError.strerror))

    tape = pd.read_csv(tape35_path, index_col=False, delimiter=";")
    tape.loc[:, "Unnamed: 32"] = ""
    tape.rename(columns={"Unnamed: 32": ""}, inplace=True)
    # Skip strip cause it might confuse Larsim afterwards
    #tape.rename(columns=lambda x: x.strip(), inplace=True)
    tape.to_csv(tape35_path, index=False, sep=";")


def tape35_configurations(parameters, tape35_path, configurationObject, TGB=None, addSampledValue=True):
    """
    Set the calibrated parameters inside the tape35 configuration file

    :param configurationObject dictionary containing variable names, values, limits, etc.
    :return
    """

    variable_names = []
    limits = []
    for i in configurationObject["Variables"]:
        variable_names.append(i["name"])
        limits.append((i["lower_limit"], i["upper_limit"]))

    # Check if local tape35 file exists
    #while not os.path.exists(tape35_path):
    #    time.sleep(1)
    if not os.path.exists(tape35_path):
        raise IOError('File does not exist: %s. %s' % (tape35_path, IOError.strerror))

    tape = pd.read_csv(tape35_path, index_col=False, delimiter=";")
    tape.loc[:, "Unnamed: 32"] = ""
    tape.rename(columns={"Unnamed: 32": ""}, inplace=True)
    # Skip strip cause it might confuse Larsim afterwards
    #tape.rename(columns=lambda x: x.strip(), inplace=True)

    try:
        TGB = int(configurationObject["GaugeControlRegion"]["TGB"])
        addSampledValue = strtobool(configurationObject["GaugeControlRegion"]["addSampledValue"])
    except KeyError:
        TGB = None
        addSampledValue = True


    # changes tape35 entries
    if TGB is not None:
        # original entry + added value
        if addSampledValue:
            for j in range(0, len(variable_names)):
                tape.loc[tape['   TGB'] == TGB, variable_names[j]] = round(Decimal(_var_limits((tape.loc[tape['   TGB'] == TGB, variable_names[j]] + parameters[j]), limits[j])), 2)
        # new value
        else:
            for j in range(0, len(variable_names)):
                tape.loc[tape['   TGB'] == TGB, variable_names[j]] = round(Decimal(_var_limits(parameters[j], limits[j])), 2)
    else:
        # original entry + added value
        if addSampledValue:
            for j in range(0, len(variable_names)):
                for i in range(0, len(tape[variable_names[0]])):
                    tape.loc[i, variable_names[j]] = round(Decimal(_var_limits((tape.loc[i, variable_names[j]] + parameters[j]), limits[j])), 2)
        # new value
        else:
            for j in range(0, len(variable_names)):
                for i in range(0, len(tape[variable_names[0]])):
                    tape.loc[i, variable_names[j]] = round(Decimal(_var_limits(parameters[j], limits[j])), 2)


    tape.to_csv(tape35_path, index=False, sep=";")

##############################################################
##### Parse big Lila files storing input metheorological data
##############################################################

def master_lila_parser_based_on_time_crete_new(timeframe, master_lila_paths, new_lila_paths, start_date_min_3_bool=False):
    """
    This function samples a list of big lila files
    (storing metheorological data for 2003-2018 period) based on the time interval
    It can as well sample single big lila file

    Args:
    timeframe - is tuple with begin and end of interval
    form of the date+time = dd.mm.yyyy hh:mm;

    """
    #configured_lila_paths = [os.path.abspath(os.path.join(new_path, i)) for i in new_lila_files]
    if start_date_min_3_bool:
        start_date = timeframe[0] - datetime.timedelta(days=3)
    else:
        start_date = timeframe[0]

    interval_of_interest = (str(start_date.day).zfill(2) + "." + str(start_date.month).zfill(2) + "." +
                            str(start_date.year) + " " + str(start_date.hour).zfill(2) + ":" + str(start_date.minute).zfill(2),

                str(timeframe[1].day).zfill(2) + "." + str(timeframe[1].month).zfill(2) + "." + str(timeframe[1].year) + " " +
                            str(timeframe[1].hour).zfill(2) + ":" + str(timeframe[1].minute).zfill(2))

    i = 0

    if not isinstance(master_lila_paths, list):
        master_lila_paths = [master_lila_paths,]

    if not isinstance(new_lila_paths, list):
        new_lila_paths = [new_lila_paths,]

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
    #print('You have successfully parsed master lila files based on input timespan')
