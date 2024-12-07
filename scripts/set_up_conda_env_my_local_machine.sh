#!/bin/bash
conda config --get channel_priority
conda config --set channel_priority flexible
conda create -p /Users/aleksandar/anaconda3/envs/my_uq python=3.11
conda install -p /Users/aleksandar/anaconda3/envs/my_uq --file requirements requirements_no_version.txt --update-deps
#conda install -p /Users/aleksandar/anaconda3/envs/my_uq -c conda-forge pyproj
#conda install -p /Users/aleksandar/anaconda3/envs/my_uq -c conda-forge nb_conda_kernels
$(which pip) install chaospy
cd UQEF/
git checkout parallel_statistics
$(which python) setup_new.py install
cd ../
cd Larsim_Utility_Set/
git checkout master
git pull
$(which python) setup.py install.
#conda install -n my_uq -c conda-forge pybamm
$(which pip) install umbridge
cd sparseSpACE/
$(which pip) install -e .