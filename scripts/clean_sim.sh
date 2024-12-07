#!/bin/bash

#clean this repo
#git clean -dxf .
#-d Remove untracked directories in addition to untracked files
#-x ignored files are also removed; remove all untracked files, including build products
#-f or -n If the git configuration specifies clean.requireForce as true, git-clean will refuse to run unless given -f or -n

#clean the executions
#rm -r $SCRATCH/path_to_executions

#clean
find "$1" -iname "*.pyc" -exec rm {} \;
find "$1" -iname "*pycache*" -exec rm -R {} \;
find "$1" -iname "*.cmd" -exec rm {} \;
find "$1" -iname "*.out" -exec rm {} \;

#prepare simulation environment
##./prepare_sim.sh
