import csv
import datetime
from glob import glob
import json
import pandas as pd
import re #for regular expressions
import os
import os.path as osp
import subprocess
import time

#from paths import *

def datetime_parse(configuration):

    data = configuration["Timeframe"]
    start_date = datetime.datetime(data["start_year"],data["start_month"], data["start_day"],
                                   data["start_hour"], data["start_minute"])

    end_date = datetime.datetime(data["end_year"], data["end_month"], data["end_day"],
                                 data["end_hour"], data["end_minute"])
    return [start_date, end_date]

def tape10_configurations(timeframe, master_tape10_file, new_path):

    #new_path = new_path + '/tape10'
    parameter = ["EREIGNISBEGINN       ", "EREIGNISENDE       ", "VORHERSAGEBEGINN     ", "VORHERSAGEDAUER      "]

    ereignisbeginn = "EREIGNISBEGINN       " + str(timeframe[0].day).zfill(2) + " " + str(timeframe[0].month).zfill(
        2) + " " + str(timeframe[0].year) + " " + str(timeframe[0].hour).zfill(2) + " " + str(timeframe[0].minute).zfill(2) + "\n"

    ereignisende = "EREIGNISENDE         " + str(timeframe[1].day).zfill(2) + " " + str(timeframe[1].month).zfill(
        2) + " " + str(timeframe[1].year) + " " + str(timeframe[1].hour).zfill(2) + " " + str(timeframe[1].minute).zfill(2) + "\n"

    vorhersagebeginn = "VORHERSAGEBEGINN     53\n" # this is set by default, might be changed as well...
    changes = [ereignisbeginn, ereignisende, vorhersagebeginn]
    print(changes)

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
    def calc_duration(changes):

        temp = changes[0].split(" ")
        date_begin = datetime.datetime(int(temp[9]), int(temp[8]), int(temp[7]), int(temp[10]), int(temp[11]))
        temp = changes[1].split(" ")
        date_end = datetime.datetime(int(temp[11]), int(temp[10]), int(temp[9]), int(temp[12]), int(temp[13]))
        total_time = date_end - date_begin
        total_time = int(total_time.total_seconds() / 3600)
        temp = changes[2].split(" ")
        beginn = int(temp[5])
        changes.append(parameter[3] + str(total_time - beginn) + "\n")

    calc_duration(changes)
    print(changes)
    replace(master_tape10_file, new_path, parameter, changes)


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
def master_lila_parser_on_time_crete_new(timeframe, master_lila_paths, new_lila_files, new_path):
    """
    This function samples out ALL big lila files based on the time interval

    Args:
    timeframe - is tuple with begin and end of interval
    form of the date+time = dd.mm.yyyy hh:mm;

    """
    configured_lila_paths = [os.path.abspath(os.path.join(new_path, i)) for i in new_lila_files]
    start_date_min_3 = timeframe[0] - datetime.timedelta(days=3)
    interval_of_interest = (str(start_date_min_3.day).zfill(2) + "." + str(start_date_min_3.month).zfill(2) + "." +
                            str(start_date_min_3.year) + " " + str(start_date_min_3.hour).zfill(2) + ":" + str(start_date_min_3.minute).zfill(2),

                str(timeframe[1].day).zfill(2) + "." + str(timeframe[1].month).zfill(2) + "." + str(timeframe[1].year) + " " +
                            str(timeframe[1].hour).zfill(2) + ":" + str(timeframe[1].minute).zfill(2))

    i = 0
    for idx, file_path in enumerate(master_lila_paths):
        with open(master_lila_paths[idx], "r", encoding="ISO-8859-1") as f:
            with open(configured_lila_paths[idx], "w", encoding="ISO-8859-1") as out:
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
    print('You have successfully parsed master lila files based on input time span')


#calculates # of timesteps set in tape10 configuration
def tape10_timesteps(tape10_path):
    a = 0
    b = 0
    interval = 0
    with open(tape10_path, "r", encoding="ISO-8859-1") as tape:
        for lines in tape:
            line = lines.split(" ")
            if line[0] == "INTERVALLAENGE":
                interval = float(line[7])
            if line[0] == "VORHERSAGEBEGINN":
                a = float(line[5])
            if line[0] == "VORHERSAGEDAUER":
                b = float(line[6])

    grid_size = int((a+b)/ interval) + 1
    t = [i * interval for i in range(grid_size)]

    return t


def var_limits(var, limits):
    if var < limits[0]: var = limits[0]
    elif var > limits[1]: var = limits[1]
    return var


# tape 35 changes
def tape35_configurations(parameters, curr_working_dir, configurationObject):
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

    # changes tape35 entries: original entry + added value
    for j in range(0, len(variable_names)):
        for i in range(0, len(tape[variable_names[0]])):
            tape.loc[i, variable_names[j]] = var_limits((tape.loc[i, variable_names[j]] + parameters[j]), limits[j])

    tape.to_csv(curr_working_dir+"/tape35", index=False, sep=";")


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

            #TODO Change this
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

                if line[1] != "-":
                    result_list.append((int(index_run), curr_ident, curr_rtype, timestemp, float(line[1])))
                else:
                    result_list.append((int(index_run), curr_ident, curr_rtype, timestemp, None))

                if curr_rtype2 != "":
                    if line[1] != "-":
                        result_list.append((int(index_run), curr_ident, curr_rtype2, timestemp, float(line[2])))
                    else:
                        result_list.append((int(index_run), curr_ident, curr_rtype2, timestemp, None))

    labels = ['Index_run','Stationskennung', 'Type', 'TimeStamp', 'Value']
    result = pd.DataFrame.from_records(result_list, columns=labels)
    return result





def lila_parser_toPandas(file_path, index_run=0):
    """
    Function parses all data entries within some lila file

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
                if val != "-":
                    result_list.append((int(index_run), stations_array[idx], type_of_data, timestemp, float(val)))
                else:
                    result_list.append((int(index_run), stations_array[idx], type_of_data, timestemp, None))

    labels = ['Index_run', 'Stationskennung', 'Type', 'TimeStamp', 'Value']
    result = pd.DataFrame.from_records(result_list, columns=labels)
    return result
