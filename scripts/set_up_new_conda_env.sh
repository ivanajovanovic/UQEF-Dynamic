#!/bin/bash

## Some conda update and setting channels commands
# conda update -n base --override-channels -c defaults conda
# conda update -n base conda
# conda config --set channel_priority flexible
# conda config --show channels
# conda config --show default_channels

conda create -p /work/ga45met/anaconda3/envs/py3115_uq python=3.11.5 --file requirements_py311.txt
# or 
# conda install -p /work/ga45met/anaconda3/envs/py3115_uq --file requirements_py311.txt -v
# or 
# conda create -n py311_uq python=3.11
# conda install -n py3115_uq --file requirements_py311.txt -v

# conda update -n py3115_uq --override-channels -c defaults conda

# conda install -c conda-forge nb_conda_kernels
conda install -p /work/ga45met/anaconda3/envs/py3115_uq -c conda-forge nb_conda_kernels
conda install -p /work/ga45met/anaconda3/envs/py3115_uq -c conda-forge pyproj

conda activate py3115_uq

$(which pip) install chaospy

cd UQEF/
git checkout parallel_statistics
$(which python) setup_new.py install
cd ../

### sparseSpACE
##cd sparseSpACE/
##git checkout master
##git pull
##$(which python) setup.py install
##cd ../

cd Larsim_Utility_Set/
git checkout master
git pull
$(which python) setup.py install.