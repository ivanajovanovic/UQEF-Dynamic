# Forward Uncertainty Quantification of the Larsim Water Balance Model
This repository contains code for performing forward uncertainty quantification of the Larsim Model.

### Requirements
Compatible with Python 3.6+
In order to run UQ simulation a user has to have an access to 
Larsim_Utility_Set and UQEF library, as well as necessary data to run and stimulate the Larsim model.

Additional Software requirements:
- chaospy
- requirements of the UQEF library and UQEF library itself
- requirements of the Larsim_Utility_Set library and Larsim_Utility_Set library itself

### How to run the code/simulation

### Runing the simulation on HPC

### Things to be done before each run 
- delete all subfolders in folder `model_runs`
- optional - delete all whm and lila files inside `../Larsim-data/WHM Regen/master_configuration`

### Paths Definitions
#### Paths Definitions from LarsimUtilityFunctions
* home_dir
* data_dir (the parent folder of the **Larsim-data** folder)
* scratch_dir (folder to save the output)
* sourceDir (folder where source code is)
* working_dir
* larsim_data_path
*sim_folder*
#### Paths Definitions from UQEF
* inputModelDir - larsim_data_path
* outputModelDir - outputResultDir
* outputResultDir - scratch_dir/folder_for_current_results (save dictionary with arguments, save nodes dist, save nodes files, save statistic plot reuslts, save statistics file.)
                  - uqsim.configuration_object["Directories"]["working_dir"] = outputResultDir/model_runs
* sourceDir - sourceDir

### Output files
* outputResultDir/df_measured.pkl
* outputResultDir/df_simulated.pkl
* outputResultDir/df_unaltered_ergebnis.pkl
* outputResultDir/gof_past_sim_meas.pkl
* outputResultDir/gof_unaltered_meas.pkl
* outputResultDir/master_configuration/
* if LarsimModel.run_and_save_simulations is True then the following files will be saved as well:
** df_Larsim_run_i.pkl
** parameters_Larsim_run_i.pkl
** goodness_of_fit_i.pkl
*** df_Larsim_run_processed_i.pkl

