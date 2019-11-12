"""
A set of utility functions for playing mostly with big lila files, examing data, and and printing different statistics
and stroting data as Pandas objects

@author: Ivana Jovanovic
"""
import csv
import datetime
from decimal import Decimal
from distutils.util import strtobool
from glob import glob
import json
import pickle
import pandas as pd
import re #for regular expressions
import os
import os.path as osp
import numpy as np
import subprocess
import time


def master_lila_header_vs_timevalues(master_lila_paths, lila_time_value_paths, lila_header_paths):
    """
    This function devides ALL big lila files into:
    header (meta-data) part
    tail(timeStaps+values for all the station) part

    plus while iterating saves the dictionary containing all the measuring stations

    """
    if not isinstance(master_lila_paths, list):
        master_lila_paths = [master_lila_paths,]

    if not isinstance(lila_time_value_paths, list):
        lila_time_value_paths = [lila_time_value_paths,]

    if not isinstance(lila_header_paths, list):
        lila_header_paths = [lila_header_paths,]

    stationsInfo = {}
    for idx, file_path in enumerate(master_lila_paths):
        with open(file_path, "r", encoding="ISO-8859-1") as f:
            file_name = os.path.basename(file_path)
            if file_name.split('-')[0] == "station":
                parameter = file_name.split('-')[1]
            else:
                parameter = file_name.split('_')[0]
            stationsInfo[parameter] = {}
            with open(lila_time_value_paths[idx], "w", encoding="ISO-8859-1") as out_tail:
                with open(lila_header_paths[idx], "w", encoding="ISO-8859-1") as out_header:
                    for lines in f:
                        out_header.writelines(lines)
                        if lines.split(";")[0] == "Stationsnummer":
                            # stationsnummer[idx]=[parameter + "_" + stanum for stanum in lines.split(";")[1:]]
                            stationsInfo[parameter]['StationNumbers'] = [parameter + "_" + stanum.rstrip() for stanum in
                                                         lines.split(";")[1:]]
                        elif lines.split(";")[0] == "Stationskennung":
                            stationsInfo[parameter]['StationNames'] = [parameter + "_" + stanum.rstrip() for stanum in
                                                         lines.split(";")[1:]]
                        elif lines.split(";")[0] == "Kommentar":
                            break
                    for lines in f:
                        out_tail.writelines(lines)
    pickle_out = open("stations_dict.pickle", "wb")
    pickle.dump(stationsInfo, pickle_out)
    pickle_out.close()

    #TODO Delete afterwards
    # how to read pickled dictionaries afterwards
    #pickle_in = open(stations_dic_path, "rb")
    #stations_dict = pickle.load(pickle_in)
    #print(stations_dict)
    #pickle_in.close()


def creat_all_stations_dictionary(master_lila_paths, dic_path="./stations_dict.pickle"):
    """
    This function iterates through all the master lila files
    and creates the dictionary containing all the measuring stations
    The function should be called only once at the beginning

    """
    if not isinstance(master_lila_paths, list):
        master_lila_paths = [master_lila_paths,]

    stationsInfo = {}
    for idx, file_path in enumerate(master_lila_paths):
        with open(file_path, "r", encoding="ISO-8859-1") as f:
            file_name = os.path.basename(file_path)
            if file_name.split('-')[0] == "station":
                parameter = file_name.split('-')[1]
            else:
                parameter = file_name.split('_')[0]
            stationsInfo[parameter] = {}
            for lines in f:
                if lines.split(";")[0] == "Stationsnummer":
                    # stationsnummer[idx]=[parameter + "_" + stanum for stanum in lines.split(";")[1:]]
                    stationsInfo[parameter]['StationNumbers'] = [parameter + "_" + stanum.rstrip() for stanum in
                                                                 lines.split(";")[1:]]
                elif lines.split(";")[0] == "Stationskennung":
                    stationsInfo[parameter]['StationNames'] = [parameter + "_" + stanum.rstrip() for stanum in
                                                               lines.split(";")[1:]]
                elif lines.split(";")[0] == "Kommentar":
                    break

    pickle_out = open(dic_path, "wb")
    pickle.dump(stationsInfo, pickle_out)
    pickle_out.close()


def lila_header_to_stations_table(lila_files, stations_tabel_path):
    """
    This function iterates throuh all lila files and creates a table/csv file
    containing the information ablut all the stations ['Stationsnummer','Stationskennung',
    'X-Koordinate','Y-Koordinate','Koordinatensystem','Source']
    """
    if not isinstance(lila_files, list):
        lila_files = [lila_files,]

    stations_table_dict = {}
    stationsInfo = {}
    stations_table_dict['Stationsnummer'] = []
    stations_table_dict['Station'] = []
    stations_table_dict['Stationskennung'] = []
    stations_table_dict['X-Koordinate'] = []
    stations_table_dict['Y-Koordinate'] = []
    stations_table_dict['Koordinatensystem'] = []
    # sources = []
    for idx, one_file in enumerate(lila_files):
        with open(one_file, "r", encoding="ISO-8859-1") as f:
            file_name = os.path.basename(one_file)
            if file_name.split('-')[0] == "station":
                parameter = file_name.split('-')[1]
            else:
                parameter = file_name.split('_')[0]
            stationsInfo[parameter] = {}
            for lines in f:
                if lines.split(";")[0] == "Stationsnummer":
                    stationsInfo[parameter]['Stationsnummer'] = [parameter + "_" + stanum.rstrip() for stanum in
                                                                 lines.split(";")[1:]]
                    # stations_table_dict['Stationsnummer'].append([element.rstrip() for element in lines.split(";")[1:]])
                    #for element in lines.split(";")[1:]: stations_table_dict['Stationsnummer'].append(element.rstrip())
                elif lines.split(";")[0] == "Station":
                    # stations_table_dict['Station'].append([element.rstrip() for element in lines.split(";")[1:]])
                    #for element in lines.split(";")[1:]: stations_table_dict['Station'].append(element.rstrip())
                    stationsInfo[parameter]['Station'] = [parameter + "_" + stanum.rstrip() for stanum in
                                                                 lines.split(";")[1:]]
                elif lines.split(";")[0] == "Stationskennung":
                    # stations_table_dict['Stationskennung'].append([element.rstrip() for element in lines.split(";")[1:]])
                    #for element in lines.split(";")[1:]: stations_table_dict['Stationskennung'].append(element.rstrip())
                    stationsInfo[parameter]['Stationskennung'] = [parameter + "_" + stanum.rstrip() for stanum in
                                                                 lines.split(";")[1:]]
                elif lines.split(";")[0] == "X-Koordinate":
                    # stations_table_dict['X-Koordinate'].append([element.rstrip() for element in lines.split(";")[1:]])
                    #for element in lines.split(";")[1:]: stations_table_dict['X-Koordinate'].append(element.rstrip())
                    stationsInfo[parameter]['X-Koordinate'] = [parameter + "_" + stanum.rstrip() for stanum in
                                                                 lines.split(";")[1:]]
                elif lines.split(";")[0] == "Y-Koordinate":
                    # stations_table_dict['Y-Koordinate'].append([element.rstrip() for element in lines.split(";")[1:]])
                    #for element in lines.split(";")[1:]: stations_table_dict['Y-Koordinate'].append(element.rstrip())
                    stationsInfo[parameter]['Y-Koordinate'] = [parameter + "_" + stanum.rstrip() for stanum in
                                                                 lines.split(";")[1:]]
                elif lines.split(";")[0] == "Koordinatensystem":
                    #stations_table_dict['Koordinatensystem'].append([element.rstrip() for element in lines.split(";")[1:]])
                    #for element in lines.split(";")[1:]: stations_table_dict['Koordinatensystem'].append(element.rstrip())
                    stationsInfo[parameter]['Koordinatensystem'] = [parameter + "_" + stanum.rstrip() for stanum in
                                                                 lines.split(";")[1:]]
                elif lines.split(";")[0] == "Kommentar":
                    break

    pickle_out = open(stations_tabel_path, "wb")
    pickle.dump(stations_table_dict, pickle_out)
    pickle_out.close()


