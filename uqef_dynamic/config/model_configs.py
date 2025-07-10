"""
Model-specific configuration classes.
"""

import pathlib
from typing import Optional, Union
from .base_config import UQConfiguration


class ModelConfiguration:
    """Base class for model-specific configurations."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model_paths = {}
        self.model_settings = {}
        self.config_file = None

    def get_config_file_path(self) -> Optional[pathlib.Path]:
        """Get the path to the model's configuration file."""
        return None
        
    def validate_paths(self) -> bool:
        """Validate that required paths exist."""
        return True

    def set_config_file_path(self, config_file: str):
        """Set the path to the configuration file."""
        self.config_file = config_file


class HBVSASKConfig(ModelConfiguration):
    """Configuration for HBV-SASK model."""
    
    def __init__(self):
        super().__init__("hbvsask")
        
        # HBV-SASK specific paths
        self.basin = "Oldman_Basin"
        
        # Default configuration files
        self.config_files = {
            "10D": "/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/data/configurations/configuration_hbv_10D.json",
            "10D_single_qoi": "/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/data/configurations/configuration_hbv_10D_single_qoi.json",
            "10D_short": "/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/data/configurations/configuration_hbv_10D_short.json",
            "10D_MC": "/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/configurations/configuration_hbv_10D_MC.json",
            "10D_MC_banff": "/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/configurations/configuration_hbv_10D_MC_banff.json",
            "12D_MC": "/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/configurations/configuration_hbv_12D_MC.json",
            "7D": "/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/data/configurations/configuration_hbv_7D.json",
            "11D_single_qoi_autoregressive":"/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/data/configurations/configuration_hbv_10D_single_qoi_autoregressive"
        }
        
        # Default paths
        self.default_input_dir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data")
        self.default_source_dir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data")
        

    def get_config_file_path(self, variant: str = "10D") -> pathlib.Path:
        if self.config_file:
            return pathlib.Path(self.config_file)
        elif variant in self.config_files:
            """Get configuration file path for specific variant."""
            self.config_file = self.config_files[variant]
            return pathlib.Path(self.config_file)
        else:
            raise ValueError(f"Unknown HBV-SASK variant: {variant}. Available: {list(self.config_files.keys())}")
    
    def get_output_dir(self, run_type: str, **kwargs) -> pathlib.Path:
        """Generate output directory path based on run type."""
        base_dir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/hbvsask_runs")
        
        run_dirs = {
            "mc_sobol": "mc_with_sobol_computation_delta_q",
            "sc_sliding": "beta_2007_sc_sliding_window_rmse",
            "ensemble": "ensemble_q6_p3_6d_2006_banff",
            "mc_short": "mc_10d_short_banff",
            "sc_sparse": "sc_10d_p2_sg_l5_ct07_short"
        }
        
        if run_type in run_dirs:
            return base_dir / run_dirs[run_type]
        else:
            # Generate custom name
            custom_name = kwargs.get('custom_name', f'hbv_{run_type}')
            return base_dir / custom_name


class LarsimConfig(ModelConfiguration):
    """Configuration for LARSIM model."""
    
    def __init__(self):
        super().__init__("larsim")
        
        # LARSIM specific paths
        self.default_input_dir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/Larsim-data")
        
        # Configuration files
        self.config_files = {
            "boundary_values": "/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/configurations_Larsim/configurations_larsim_boundery_values.json",
            "may": "/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/configurations_Larsim/configurations_larsim_4_may.json",
            "high_flow": "/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/configurations_Larsim/configurations_larsim_high_flow.json"
        }
        
    def get_config_file_path(self, variant: str = "boundary_values") -> pathlib.Path:
        if self.config_file:
            return pathlib.Path(self.config_file)
        elif variant in self.config_files:
            """Get configuration file path for specific variant."""
            self.config_file = self.config_files[variant]
            return pathlib.Path(self.config_file)
        else:
            raise ValueError(f"Unknown LARSIM variant: {variant}. Available: {list(self.config_files.keys())}")
    
    def get_output_dir(self, run_type: str, **kwargs) -> pathlib.Path:
        """Generate output directory path."""
        base_dir = pathlib.Path("/gpfs/scratch/pr63so/ga45met2/Larsim_runs")
        
        run_dirs = {
            "ensemble_2013": "larsim_run_ensemble_2013_all_tgb",
            "lai_may": "larsim_run_lai_may_cc_q_6_p_4_stat_trial",
            "sc_kpu": "larsim_run_sc_kpu_l_6_d_5_p_3_2013"
        }
        
        if run_type in run_dirs:
            return base_dir / run_dirs[run_type]
        else:
            custom_name = kwargs.get('custom_name', f'larsim_{run_type}')
            return base_dir / custom_name


class BatteryConfig(ModelConfiguration):
    """Configuration for Battery (PyBaMM) model."""
    
    def __init__(self):
        super().__init__("battery")
        
        # Battery specific paths
        self.default_input_dir = pathlib.Path("/dss/dsshome1/lxc0C/ga45met2/.conda/envs/my_uq_env/lib/python3.11/site-packages/pybamm/input/drive_cycles")
        self.config_file_path = "/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/uqef_dynamic/models/pybamm/configuration_battery.json"
        
    def get_config_file_path(self) -> pathlib.Path:
        """Get configuration file path."""
        return pathlib.Path(self.config_file_path)
    
    def get_output_dir(self, run_type: str = "mc_6d_10000_lhc") -> pathlib.Path:
        """Generate output directory path."""
        base_dir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/battery_runs")
        return base_dir / run_type


class IshigamiConfig(ModelConfiguration):
    """Configuration for Ishigami test function."""
    
    def __init__(self):
        super().__init__("ishigami")
        
        # Ishigami specific settings
        self.config_file_path = "/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic/data/configurations/configuration_ishigami.json"
        
    def get_config_file_path(self) -> pathlib.Path:
        """Get configuration file path."""
        return pathlib.Path(self.config_file_path)
    
    def get_output_dir(self, run_type: str = "sc_full_p5_q10_ct07") -> pathlib.Path:
        """Generate output directory path."""
        base_dir = pathlib.Path("/work/ga45met/ishigami_runs/simulations_sep_2024")
        return base_dir / run_type


class SimpleOscillatorConfig(ModelConfiguration):
    """Configuration for Simple Oscillator model."""
    
    def __init__(self):
        super().__init__("simple_oscillator")
        
        # Simple Oscillator specific settings
        self.config_file_path = "/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/data/configurations/configuration_simple_oscillator.json"
        
    def get_config_file_path(self) -> pathlib.Path:
        """Get configuration file path."""
        return pathlib.Path(self.config_file_path)
    
    def get_output_dir(self, run_type: str = "sc_kl10_l7_p3_generalized") -> pathlib.Path:
        """Generate output directory path."""
        base_dir = pathlib.Path("/gpfs/scratch/pr63so/ga45met2/simple_oscillator_model")
        return base_dir / run_type


class ProductFunctionConfig(ModelConfiguration):
    """Configuration for Product Function test model."""
    
    def __init__(self):
        super().__init__("productFunction")
        
        # Product function typically doesn't need external files
        self.requires_input_files = False


class LinearDampedOscillatorConfig(ModelConfiguration):
    """Configuration for Linear Damped Oscillator model."""
    
    def __init__(self):
        super().__init__("oscillator")
        
        # Oscillator specific settings
        self.atol = 1e-10
        self.rtol = 1e-10
        self.requires_input_files = False
