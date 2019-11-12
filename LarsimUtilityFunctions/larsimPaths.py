import inspect
import os
import os.path as osp
import sys
import subprocess
import socket

#####################################
### All basic paths
#####################################
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)

if socket.gethostname().startswith("mpp2"): #TODO Add branch for execution on jHub server!
    home_dir = "/home/hpc/pr63so/ga45met2"
    # new data_dri on dss linux cluster, the old one "/naslx/projects/pr63so/ga45met2/Repositories"
    data_dir = "/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/Repositories"
else:
    home_dir = current_dir
    data_dir = current_dir


working_dir = os.path.abspath(os.path.join(data_dir, "model_runs"))

#if not os.path.isdir(working_dir):
#    subprocess.run(["mkdir", working_dir])

#####################################
### Larsim related paths
#####################################

if socket.gethostname().startswith("mpp2"):
    larsim_data_path = os.path.abspath(os.path.join(data_dir, 'Larsim-data'))
else:
    larsim_data_path = os.path.abspath(os.path.join(parent_dir, 'Larsim-data'))


larsim_exe_dir = os.path.abspath(os.path.join(larsim_data_path, 'Larsim-exe'))
regen_data_path = os.path.abspath(os.path.join(larsim_data_path,'WHM Regen'))
master_dir = os.path.abspath(os.path.join(regen_data_path,'master_configuration'))
#sys.path.insert(0, parentdir)
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, regen_data_path)
#sys.path.append("../Larsim-data")

all_whms_path = os.path.abspath(os.path.join(regen_data_path,'var/WHM Regen WHMS'))

master_lila_files = ["q_2003-11-01_2018-01-01.lila",
              "n_2003-11-01_2018-01-01.lila",
              "tlu_2003-11-01_2018-01-01.lila",
              "xglob_2003-11-01_2018-01-01.lila",
              "xludr_2003-11-01_2018-01-01.lila",
              "zsos_2003-11-01_2018-01-01.lila",
              "rflu_2003-11-01_2018-01-01.lila",
              "ttau_2003-11-01_2018-01-01.lila",
              "xwind_2003-11-01_2018-01-01.lila"]

master_lila_paths = [osp.abspath(osp.join(regen_data_path, i)) for i in master_lila_files]

lila_files = ["station-wq.lila","station-n.lila", "station-tlu.lila",
              "station-xglob.lila", "station-xludr.lila", "station-zsos.lila",
              "station-rflu.lila", "station-ttau.lila","station-xwind.lila"]

lila_time_value_files = ["q_time_value.lila","n_time_value.lila","tlu_time_value.lila",
              "xglob_time_value.lila","xludr_time_value.lila","zsos_time_value.lila",
              "rflu_time_value.lila","ttau_time_value.lila", "xwind_time_value.lila"]

lila_header_files = ["q_header.lila","n_header.lila","tlu_header.lila",
              "xglob_header.lila","xludr_header.lila","zsos_header.lila",
              "rflu_header.lila","ttau_header.lila", "xwind_header.lila"]

global_start_data = "02.11.2003 00:00"
global_end_data = "31.12.2017 00:00"
interval_of_interest = ("01.01.2016 00:00","01.10.2017 00:00")

master_tape10_file = os.path.abspath(os.path.join(master_dir, 'tape10_master'))
tape35_master = os.path.abspath(os.path.join(master_dir, 'tape35'))

regen_local_data_root = osp.abspath(osp.join(larsim_data_path, 'Larsim-local', 'testumgebung/models/Bayern/WHM Regen'))