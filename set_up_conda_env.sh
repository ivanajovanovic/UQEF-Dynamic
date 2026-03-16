#!/bin/bash
conda config --get channel_priority
conda config --show channels
conda config --add channels conda-forge
  conda config --set channel_priority strict
# conda config --set channel_priority flexible
conda create -n uqef_env python=3.11
conda install -n uqef_env --file requirements.txt --update-deps
# conda install -n uqef_env --file requirements_py13_version.txt --update-deps
## Activate the environment
conda activate uqef_env
# if one wants to run batter model - pybamm 24.11.2
conda install -n uqef_env -c conda-forge pybamm --no-update-deps
## From this point on installation happens via pip
## When you want to uninstall previous installation of some package (if any)
# $(which pip) uninstall -y your_package_name
# Make pip prefer wheels and refuse sdists
export PIP_ONLY_BINARY=":all:"
export PIP_PREFER_BINARY=1
export PIP_NO_BUILD_ISOLATION=1
## In case Chaospy was not installed properly via conda, install it via pip from the source code, 
## however, this is not recommended and you make sure that the dependencies are not broken
$(which pip) install --no-deps chaospy
## UQEF - clone the repo first; Note: This will be update soon!
## Install from pypi
pip install uqef
## Install from source
# git clone https://github.com/flo2k/UQEF.git
cd ../UQEF/
git checkout integrate_parallel_statistics
$(which pip) install --no-deps -e .
# Install UQEF-Dynamic
cd UQEF-Dynamic/
git checkout master
git pull
$(which pip) install -e .
## For working with the Larsim model
# cd ../Larsim_Utility_Set/
# git checkout master
# git pull
# $(which pip) install -e .
## For working with the sparseSpACE toolbox
cd ../sparseSpACE/
$(which pip) install -e .
#$(which pip) install umbridge
# Upgrading
# conda update --all
# pip list --outdated | grep -Ff <(cut -d= -f1 requirements.txt)
# pip install -U -r requirements.txt --upgrade