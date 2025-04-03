#!/bin/bash
conda config --get channel_priority
conda config --show channels
conda config --set channel_priority flexible
conda create -n my_uq_env python=3.11
conda install -n my_uq_env --file requirements.txt --update-deps
# if one wants to run batter model - pybamm 24.11.2
conda install -n my_uq_env -c conda-forge pybamm --no-update-deps
## Activate the environment
conda activate my_uq_env
## From this point on installation happens vis pip
## When you want to uninstall previous installation of some package (if any)
# $(which pip) uninstall -y your_package_name
## Chaospy
$(which pip) install chaospy
# after installing chaospy (4.3.17) I have Successfully installed chaospy-4.3.17 numpoly-1.3.6 numpy-2.2.2/2.0.2 scipy-1.15.1/1.13.1
## UQEF - clone the repo first
cd ../UQEF/
git checkout parallel_statistics
$(which pip) install -e .
## For working with the Larsim model
cd ../Larsim_Utility_Set/
git checkout master
git pull
$(which pip) install -e .
## For working with the sparseSpACE toolbox
cd ../sparseSpACE/
$(which pip) install -e .
#$(which pip) install umbridge