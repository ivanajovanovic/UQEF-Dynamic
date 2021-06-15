import json
import os
import pathlib
import sys
import time

import LarsimUtilityFunctions.larsimPaths as paths
from LarsimUtilityFunctions import larsimTimeUtility

from larsim import LarsimModel
from larsim import LarsimStatistics

sys.path.insert(0, os.getcwd())


def main():
    config_file = pathlib.Path(
        '/work/ga45met/mnt/linux_cluster_2/Larsim-UQ/configurations_Larsim/configuration_larsim_long_run.json')
    outputResultDir = os.path.abspath(os.path.join(paths.scratch_dir, "larsim_runs", 'larsim_run_01_06_one_year'))
    inputModelDir = paths.larsim_data_path
    sourceDir = paths.sourceDir
    outputModelDir = outputResultDir

    with open(config_file) as f:
        configurationObject = json.load(f)
    print(f"configurationObject: {configurationObject}")
    try:
        time_settings_list = larsimTimeUtility.parse_datetime_configuration(configurationObject)
    except KeyError as e:
        print(e)

    start_setup = time.time()
    larsimModelSetUpObject = LarsimModel.LarsimModelSetUp(configurationObject=config_file,
                                                          workingDir=outputModelDir,
                                                          get_measured_discharge=True,
                                                          get_Larsim_saved_simulations=True,
                                                          run_unaltered_sim=False)
    end_setup = time.time()
    runtime_setup = end_setup - start_setup
    print(f"Larsim SetUp runtime: {runtime_setup}!")

    start = time.time()
    larsimModel = LarsimModel.LarsimModel(configurationObject=config_file,
                                          workingDir=outputResultDir,
                                          max_retries=10)

    results_array = larsimModel.run()
    print(f"Type of results_array:{results_array}")
    print(f"results_array: {results_array}")

    start_Larsim_date = time_settings_list[0]
    end_Larsim_date = time_settings_list[1]

    end = time.time()
    runtime = end - start
    print(f"Larsim Runtime from date: {start_Larsim_date} to date: {end_Larsim_date} was: {runtime}!")


if __name__ == "__main__":
    main()
