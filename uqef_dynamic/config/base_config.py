"""
Base configuration classes for UQEF-Dynamic.
"""

import pathlib
from typing import Optional, Dict, Any, Union
from abc import ABC, abstractmethod


class UQConfiguration(ABC):
    """Base class for all UQ configuration objects."""
    
    def __init__(self):
        # TODO think about setting all default values to False, as it is in UQsim
        # that would require to change mpi, sampleFromStandardDist, parallel_statistics, sc_poly_normed
        # Core simulation parameters
        self.model: Optional[str] = None
        self.uncertain: str = "all"
        self.uq_method: Optional[str] = None
        
        # Directory settings
        self.sourceDir: Optional[Union[str, pathlib.Path]] = None
        self.inputModelDir: Optional[Union[str, pathlib.Path]] = None
        self.outputResultDir: Optional[Union[str, pathlib.Path]] = None
        self.outputModelDir: Optional[Union[str, pathlib.Path]] = None
        self.workingDir: Optional[Union[str, pathlib.Path]] = None
        self.config_file: Optional[Union[str, pathlib.Path]] = None
        
        # Parallel / MPI computing settings
        self.mpi: bool = False
        self.mpi_method: str = "MpiPoolSolver"  # "LinearSolver"
        self.mpi_combined_parallel: bool = False
        self.parallel: bool = False
        self.num_cores: int = 1
        self.mpi_chunksize: int = 1
        self.chunksize: int = 1

        # Sampling and distribution settings
        self.sampleFromStandardDist: bool = False
        self.transformToStandardDist: bool = False  # TODO I think this one is deprecated   
        
        # Node/parameter file settings
        self.read_nodes_from_file: bool = False
        self.parameters_file: Optional[Union[str, pathlib.Path]] = None
        self.parameters_setup_file: Optional[Union[str, pathlib.Path]] = None

        # Storing the uqef simulation as one big file        
        self.uqsim_store_to_file: bool = False

        # Statistics settings
        self.disable_statistics: bool = False
        self.disable_calc_statistics: bool = False
        self.parallel_statistics: bool = False
        self.instantly_save_results_for_each_time_step: bool = False
                
        # Sobol indices settings
        self.compute_Sobol_m: bool = False
        self.compute_Sobol_t: bool = False
        self.compute_Sobol_m2: bool = False
        
        # Data storage settings
        self.save_all_simulations: bool = False
        self.store_qoi_data_in_stat_dict: bool = False
        self.store_gpce_surrogate_in_stat_dict: bool = False
        self.collect_and_save_state_data: bool = False
        
        # Regression settings
        self.regression: bool = False
        self.regression_model_type: Optional[str] = None  # "OLS" | "LARS"
        
        # Polynomial settings
        self.sc_poly_rule: str = "three_terms_recurrence"  # "gram_schmidt" | "cholesky"
        self.sc_poly_normed: bool = False
        self.cross_truncation: float = 1.0  #0.7
        
        # Advanced analysis options - in case PCE surrogate is learned
        self.save_gpce_surrogate: bool = True
        self.compute_other_stat_besides_pce_surrogate: bool = True

        # In the case of MC - rank based computation of the Sobol Indices
        self.compute_sobol_indices_with_samples: bool = False
        
        # KL expansion settings
        self.compute_kl_expansion_of_qoi: bool = False
        self.kl_expansion_order: int = 10
        self.compute_timewise_gpce_next_to_kl_expansion: bool = False
        
        # Generalized Sobol indices settings
        self.compute_generalized_sobol_indices: bool = False
        self.compute_generalized_sobol_indices_over_time: bool = False
        
        # Covariance matrix settings
        self.compute_covariance_matrix_in_time: bool = False
        
        # Conditional analysis settings
        self.allow_conditioning_results_based_on_metric: bool = False
        self.condition_results_based_on_metric: str = 'NSE'
        self.condition_results_based_on_metric_value: float = 0.2
        self.condition_results_based_on_metric_sign: str = "greater_or_equal"
        
        # Statistics computation configuration
        self.dict_stat_to_compute: Dict[str, bool] = {
            "Var": True, "StdDev": True, "P10": True, "P90": True,
            "E_minus_std": False, "E_plus_std": False,
            "Skew": False, "Kurt": False, "Sobol_m": True, "Sobol_m2": False, "Sobol_t": True
        }
        
        # Plotting configuration
        self.dict_what_to_plot: Dict[str, bool] = {
            "E_minus_std": False, "E_plus_std": False, "E_minus_2std": True, "E_plus_2std": True, 
            "P10": True, "P90": True,
            "StdDev": True, "Skew": False, "Kurt": False, "Sobol_m": True, "Sobol_m2": False, "Sobol_t": True,
            "generalized_sobol_total_index": True, "generalized_sobol_main_index": True,
        }

    def apply_to_uqsim(self, uqsim):
        """Apply this configuration to a UQsim instance."""
        for attr_name, attr_value in self.__dict__.items():
            if hasattr(uqsim.args, attr_name):
                existing_value = getattr(uqsim.args, attr_name)
                # if isinstance(attr_value, type(existing_value)):
                #     setattr(uqsim.args, attr_name, attr_value)
                setattr(uqsim.args, attr_name, attr_value)
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)
            # if hasattr(self, key):
            #     setattr(self, key, value)
            # else:
            #     raise ValueError(f"Unknown configuration parameter: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary."""
        instance = cls()
        instance.update(**config_dict)
        return instance
    
    def validate(self) -> bool:
        """Validate configuration parameters. Override in subclasses."""
        # Validate conditional analysis settings
        valid_metric_signs = ["greater", "greater_or_equal", "less", "less_or_equal", "equal"]
        if self.condition_results_based_on_metric_sign not in valid_metric_signs:
            raise ValueError(f"condition_results_based_on_metric_sign must be one of {valid_metric_signs}")
        
        # Validate KL expansion order
        if self.kl_expansion_order <= 0:
            raise ValueError("kl_expansion_order must be positive")
        
        # Validate MPI chunksize
        if self.mpi_chunksize <= 0:
            raise ValueError("mpi_chunksize must be positive")
        if self.chunksize <= 0:
            raise ValueError("chunksize must be positive")

        # Validate dictionary configurations
        if not isinstance(self.dict_stat_to_compute, dict):
            raise ValueError("dict_stat_to_compute must be a dictionary")
        
        if not isinstance(self.dict_what_to_plot, dict):
            raise ValueError("dict_what_to_plot must be a dictionary")
        
        return True
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model}, uq_method={self.uq_method})"
