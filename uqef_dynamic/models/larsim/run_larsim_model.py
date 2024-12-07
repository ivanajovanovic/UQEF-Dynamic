import json
import os
import pathlib
import sys
import time

import LarsimUtilityFunctions.larsimPaths as paths
from LarsimUtilityFunctions import larsimTime
from LarsimUtilityFunctions import larsimModel

# from larsim import LarsimModel
# from larsim import LarsimStatistics

sys.path.insert(0, os.getcwd())


def main():
    config_file = pathlib.Path(
        '/work/ga45met/mnt/linux_cluster_2/Larsim-UQ/configurations_Larsim/configuration_larsim_long_run.json')
    outputResultDir = os.path.abspath(os.path.join(paths.scratch_dir, "larsim_runs", 'larsim_run_01_06_one_year'))
    inputModelDir = paths.larsim_data_path
    sourceDir = paths.sourceDir
    outputModelDir = outputResultDir

    start = time.time()

    with open(config_file) as f:
        configurationObject = json.load(f)
    print(f"configurationObject: {configurationObject}")
    try:
        time_settings_list = larsimTime.parse_datetime_configuration(configurationObject)
    except KeyError as e:
        print(e)

    larsimConfigurationsObject = larsimModel.LarsimConfigurations(
        configurationObject=configurationObject,
        configure_tape10=True,
        copy_and_configure_whms=True,
        copy_and_configure_lila_input_files=True,
        make_local_copy_of_master_dir=True,
        get_measured_discharge=True,
        read_date_from_saved_data_dir=True,
        get_saved_simulations=True,
        run_unaltered_sim=False,
        raise_exception_on_model_break=True,
        max_retries=10
    )

    larsimModelObject = larsimModel.LarsimModel(
        configurationObject=larsimConfigurationsObject,
        inputModelDir=inputModelDir,
        workingDir=outputResultDir,
        disable_statistics=True
    )

    start_setup = time.time()
    larsimModelObject.prepare(infoModel=True)
    end_setup = time.time()
    runtime_setup = end_setup - start_setup
    print(f"Larsim SetUp runtime: {runtime_setup}!")

    results_array = larsimModelObject.run(
        raise_exception_on_model_break=True,
        createNewFolder=False,
        deleteFolderAfterwards=False
    )
    print(f"Type of results_array:{results_array}")
    print(f"results_array: {results_array}")

    start_Larsim_date = time_settings_list[0]
    end_Larsim_date = time_settings_list[1]

    end = time.time()
    runtime = end - start
    print(f"Larsim Runtime from date: {start_Larsim_date} to date: {end_Larsim_date} was: {runtime}!")


if __name__ == "__main__":
    main()
