
"""
A set of utility functions for time settings and calculation of different time variables
needed for running Larsim simulations

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


def parse_datetime_tuple(interval_of_interest):
    start_date = datetime.datetime.strptime(interval_of_interest[0], '%d.%m.%Y  %H:%M')
    end_date = datetime.datetime.strptime(interval_of_interest[1], '%d.%m.%Y  %H:%M')
    return [start_date, end_date]


def parse_datetime_configuration(configuration):
    """
    Reads configuration directoy and determins the start end end date of the simulation
    """

    data = configuration["Timeframe"]
    start_date = datetime.datetime(data["start_year"],data["start_month"], data["start_day"],
                                   data["start_hour"], data["start_minute"])

    end_date = datetime.datetime(data["end_year"], data["end_month"], data["end_day"],
                                 data["end_hour"], data["end_minute"])
    return [start_date, end_date]


def tape10_configuration(timeframe, master_tape10_file="./tape10_master", new_path="./tape10"):
    """
    Filter out whm files based on timeframe

    :param
    :return
    """

    parameter = ["EREIGNISBEGINN       ", "EREIGNISENDE       ", "VORHERSAGEBEGINN     ", "VORHERSAGEDAUER      "]

    ereignisbeginn = "EREIGNISBEGINN       " + str(timeframe[0].day).zfill(2) + " " + str(timeframe[0].month).zfill(
        2) + " " + str(timeframe[0].year) + " " + str(timeframe[0].hour).zfill(2) + " " + str(timeframe[0].minute).zfill(2) + "\n"

    ereignisende = "EREIGNISENDE         " + str(timeframe[1].day).zfill(2) + " " + str(timeframe[1].month).zfill(
        2) + " " + str(timeframe[1].year) + " " + str(timeframe[1].hour).zfill(2) + " " + str(timeframe[1].minute).zfill(2) + "\n"

    vorhersagebeginn = "VORHERSAGEBEGINN     53\n" # TODO Change this this is set by default, might be changed as well...
    changes = [ereignisbeginn, ereignisende, vorhersagebeginn]
    print(changes)

    def _replace(master_tape10_file, new_path, parameter, changes):
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
    def _calc_duration(changes, timeframe):

        temp = changes[0].split(" ")
        #date_begin = datetime.datetime(int(temp[9]), int(temp[8]), int(temp[7]), int(temp[10]), int(temp[11]))
        date_begin = timeframe[0]
        temp = changes[1].split(" ")
        #date_end = datetime.datetime(int(temp[11]), int(temp[10]), int(temp[9]), int(temp[12]), int(temp[13]))
        date_end = timeframe[1]
        total_time = date_end - date_begin
        total_time = int(total_time.total_seconds() / 3600)
        temp = changes[2].split(" ")
        beginn = int(temp[5]) #TODO Check this
        changes.append(parameter[3] + str(total_time - beginn) + "\n")

    _calc_duration(changes, timeframe)
    #print(changes)
    _replace(master_tape10_file, new_path, parameter, changes)


def get_tape10_timesteps(tape10_path="./tape10"):
    """
    Calculates number of timesteps set in tape10 configuration - one way by using VORHERSAGEDAUER
    """
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


def get_tape10_timesteps(timeframe, interval=1):
    """
    calculates number of timesteps set in tape10 configuration - the second way by using difference in start and end time from tape10
    """
    start_timestep = timeframe[0]
    end_timestep = timeframe[1]
    dateTimeDifference = end_timestep - start_timestep
    dateTimeDifferenceInHours = int(dateTimeDifference.total_seconds() / 3600)

    t = [i * interval for i in range(dateTimeDifferenceInHours)]
    return t
