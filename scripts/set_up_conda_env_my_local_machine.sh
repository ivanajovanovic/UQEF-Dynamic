#!/bin/bash
conda config --get channel_priority
conda config --show channels
conda config --set channel_priority flexible
conda create -p /Users/aleksandar/anaconda3/envs/my_uq python=3.11
conda install -p /Users/aleksandar/anaconda3/envs/my_uq --file requirements_no_version.txt --update-deps
#conda install -p /Users/aleksandar/anaconda3/envs/my_uq -c conda-forge pyproj
#conda install -p /Users/aleksandar/anaconda3/envs/my_uq -c conda-forge nb_conda_kernels
## Activate the environment
conda activate my_uq
$(which pip) install chaospy
# after installing chaospy I have Successfully installed chaospy-4.3.17 numpoly-1.3.6 numpy-2.2.2 scipy-1.15.1
cd UQEF/
git checkout parallel_statistics
$(which pip) install -e . #$(which python) setup_new.py install
cd ../
cd Larsim_Utility_Set/
git checkout master
git pull
$(which pip) install -e . #$(which python) setup.py install.
#$(which pip) install umbridge
cd sparseSpACE/
$(which pip) install -e .
conda install -n my_uq -c conda-forge pybamm --no-update-deps
#conda install -p /dss/dsshome1/lxc0C/ga45met2/.conda/envs/my_uq_env -c conda-forge pybamm --no-update-deps


# commands I've run on the new instance of the Linux Cluster (Dec 2024)
conda create -n my_uq_env python=3.11
conda activate my_uq_env
conda install -p /dss/dsshome1/lxc0C/ga45met2/.conda/envs/my_uq_env --file requirements_no_version.txt --update-deps
conda install -p /dss/dsshome1/lxc0C/ga45met2/.conda/envs/my_uq_env -c conda-forge pybamm --no-update-deps
# pybamm 24.11.2
$(which pip) install chaospy
# after installing chaospy I have Successfully installed chaospy-4.3.17 numpoly-1.3.6 numpy-2.0.2 scipy-1.13.1
# chaospy 4.3.17
# numpy 1.26.4 (conda) now it is 2.2.0; numpy 2.0.2 (pip); original version installed by conda 2.1.3; was downgraded because of pybamm
cd UQEF/
git checkout parallel_statistics
$(which pip) install -e .
cd ../
cd Larsim_Utility_Set/
git checkout master
git pull
$(which pip) install -e .
cd sparseSpACE/
$(which pip) install -e .
# packages from uq_env - pybamm=22.9; numpy=1.19.4; chaospy=4.2.2