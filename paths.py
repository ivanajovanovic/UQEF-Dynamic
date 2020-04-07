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

if socket.gethostname().startswith("cm2") or socket.gethostname().startswith("mpp3"):
    home_dir = "/dss/dsshome1/lxc0C/ga45met2"
    # new data_dri on dss linux cluster, the old one "/naslx/projects/pr63so/ga45met2/Repositories"
    data_dir = "/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/Repositories"
    scratch_dir = "/gpfs/scratch/pr63so/ga45met2/Larsim_runs"
elif socket.gethostname().startswith("atsccs70"):
    home_dir = current_dir
    data_dir = "/import/home/ga45met/Repositories/Larsim"
    scratch_dir = data_dir
elif socket.gethostname().startswith("hydrobits-dataex"):
    home_dir = "/home/jupyter-ivana/Larsim_Utility_Set"
    data_dir = "/home/jupyter-ivana"
    scratch_dir = data_dir
else:
    home_dir = current_dir
    data_dir = parent_dir
    scratch_dir = data_dir

if socket.gethostname().startswith("cm2") or socket.gethostname().startswith("mpp3"):
    #working_dir = os.path.abspath(os.path.join(data_dir, "larsim_runs"))
    working_dir = scratch_dir
elif socket.gethostname().startswith("atsccs70"):
    working_dir = os.path.abspath(os.path.join(home_dir, "model_runs"))
elif socket.gethostname().startswith("hydrobits-dataex"):
    working_dir = os.path.abspath(os.path.join(home_dir, "model_runs"))
else:
    working_dir = os.path.abspath(os.path.join(current_dir, "model_runs"))

#if not os.path.isdir(working_dir):
#    subprocess.run(["mkdir", working_dir])

#####################################
### Larsim related paths
#####################################
larsim_data_path = os.path.abspath(os.path.join(data_dir, 'Larsim-data'))
larsim_exe_dir = os.path.abspath(os.path.join(larsim_data_path, 'Larsim-exe'))
larsim_exe = os.path.abspath(os.path.join(larsim_exe_dir, 'larsim-linux-intel-1000.exe'))
regen_data_path = os.path.abspath(os.path.join(larsim_data_path,'WHM Regen')) # Regen_data_root = data_working_dir
master_dir = os.path.abspath(os.path.join(larsim_data_path,'WHM Regen','master_configuration'))
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

global_start_data = "02.11.2003 00:00"
global_end_data = "31.12.2017 00:00"

master_lila_paths = [osp.abspath(osp.join(regen_data_path, i)) for i in master_lila_files]
lila_files = ["station-wq.lila","station-n.lila", "station-tlu.lila",
              "station-xglob.lila", "station-xludr.lila", "station-zsos.lila",
              "station-rflu.lila", "station-ttau.lila","station-xwind.lila"]

#lila_configured_paths = [os.path.abspath(os.path.join(master_dir, i)) for i in lila_files]

#timeconfigurations_json = os.path.abspath(os.path.join(master_dir, "configurations.json"))

master_tape10_file = os.path.abspath(os.path.join(master_dir, 'tape10_master'))
configured_tape10_file = master_dir

figureFileName = "statisticsFigure"
