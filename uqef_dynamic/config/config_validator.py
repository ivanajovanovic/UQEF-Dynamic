"""
Configuration validation for UQEF-Dynamic.
"""

import os
import pathlib
from typing import List, Dict, Any
from .base_config import UQConfiguration
from .uq_configs import SAMPLING_RULES, QUADRATURE_RULES, POLY_RULES, REGRESSION_TYPES


class ValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass


class ConfigurationValidator:
    """Validates UQ configurations."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, config: UQConfiguration) -> bool:
        """
        Validate a UQ configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Basic validation
        self._validate_basic_parameters(config)
        
        # UQ method specific validation
        self._validate_uq_method(config)
        
        # File-based configuration validation
        self._validate_file_based_parameters(config)
        
        # Path validation
        self._validate_paths(config)
        
        # Parameter consistency validation
        self._validate_parameter_consistency(config)
        
        # Model-specific validation
        self._validate_model_specific(config)
        
        # Check for errors
        if self.errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in self.errors)
            if self.warnings:
                error_msg += "\nWarnings:\n" + "\n".join(f"  - {warning}" for warning in self.warnings)
            raise ValidationError(error_msg)
        
        # Print warnings if any
        if self.warnings:
            print("Configuration warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        return True
    
    def _validate_basic_parameters(self, config: UQConfiguration):
        """Validate basic configuration parameters."""
        
        # Check required parameters
        if config.model is None:
            self.errors.append("model must be specified")
        
        if config.uq_method is None:
            self.errors.append("uq_method must be specified")
        
        # Check numeric parameters
        if config.chunksize <= 0:
            self.errors.append("chunksize must be positive")
        
        if config.num_cores <= 0:
            self.errors.append("num_cores must be positive")
        
        # TODO This is not really necessary, but can be useful
        if hasattr(config, 'cross_truncation') and config.cross_truncation <= 0 or config.cross_truncation > 1:
            self.errors.append("cross_truncation must be between 0 and 1")
    
    def _validate_uq_method(self, config: UQConfiguration):
        """Validate UQ method specific parameters."""
        
        if config.config_uq_method == "mc":
            self._validate_mc_parameters(config)
        elif config.config_uq_method == "mc_regression":
            self._validate_mc_regression_parameters(config)
        elif config.config_uq_method == "sc":
            self._validate_sc_parameters(config)
        elif config.config_uq_method == "psp":
            self._validate_psp_parameters(config)
        elif config.config_uq_method == "saltelli":
            self._validate_saltelli_parameters(config)
        elif config.config_uq_method == "ensemble":
            self._validate_ensemble_parameters(config)
        elif config.config_uq_method == "sparse_grid":
            self._validate_sparse_grid_parameters(config)
        else:
            self.warnings.append(f"Unknown UQ method: {config.config_uq_method}")
    
    def _validate_mc_parameters(self, config: UQConfiguration):
        """Validate Monte Carlo specific parameters."""
        
        if not hasattr(config, 'mc_numevaluations'):
            self.errors.append("mc_numevaluations must be specified for MC method")
        elif config.mc_numevaluations <= 0:
            self.errors.append("mc_numevaluations must be positive")
        
        if not hasattr(config, 'sampling_rule'):
            self.errors.append("sampling_rule must be specified for MC method")
        elif config.sampling_rule not in SAMPLING_RULES:
                self.errors.append(f"sampling_rule must be one of {SAMPLING_RULES}")

        if hasattr(config, 'regression'):
            if config.regression:
                if not hasattr(config, 'sc_p_order'):
                    self.errors.append("sc_p_order must be specified for SC method")
                elif config.sc_p_order <= 0:
                    self.errors.append("sc_p_order must be positive")   

                # Check polynomial rule
                if not hasattr(config, 'sc_poly_rule'):
                    self.errors.append("sc_poly_rule must be specified for SC method")
                elif config.sc_poly_rule not in POLY_RULES:
                        self.errors.append(f"sc_poly_rule must be one of {POLY_RULES}")
 
    def _validate_sc_parameters(self, config: UQConfiguration):
        """Validate Stochastic Collocation specific parameters."""
        
        if not hasattr(config, 'sc_p_order'):
            self.errors.append("sc_p_order must be specified for SC method")
        elif config.sc_p_order <= 0:
            self.errors.append("sc_p_order must be positive")
        
        # Check polynomial rule
        if not hasattr(config, 'sc_poly_rule'):
            self.errors.append("sc_poly_rule must be specified for SC method")
        elif config.sc_poly_rule not in POLY_RULES:
                self.errors.append(f"sc_poly_rule must be one of {POLY_RULES}")

        # Only validate quadrature parameters if not reading nodes from file
        if not config.read_nodes_from_file or config.parameters_file is None:
            if not hasattr(config, 'sc_q_order'):
                self.errors.append("sc_q_order must be specified for SC method")
            elif config.sc_q_order <= 0:
                self.errors.append("sc_q_order must be positive")

            if not hasattr(config, 'sc_quadrature_rule'):
                self.errors.append("sc_quadrature_rule must be specified for SC method")
            elif config.sc_quadrature_rule not in QUADRATURE_RULES:
                    self.errors.append(f"sc_quadrature_rule must be one of {QUADRATURE_RULES}")
    
    def _validate_saltelli_parameters(self, config: UQConfiguration):
        """Validate Saltelli specific parameters."""
        
        if not hasattr(config, 'mc_numevaluations'):
            self.errors.append("mc_numevaluations must be specified for Saltelli method")
        elif config.mc_numevaluations <= 0:
            self.errors.append("mc_numevaluations must be positive")

        if not hasattr(config, 'sampling_rule'):
            self.errors.append("sampling_rule must be specified for MC method")
        elif config.sampling_rule not in SAMPLING_RULES:
                self.errors.append(f"sampling_rule must be one of {SAMPLING_RULES}")

        # Saltelli requires Sobol indices
        if not config.compute_Sobol_m and not config.compute_Sobol_t:
            self.warnings.append("Saltelli method typically computes Sobol indices")
    
    def _validate_mc_regression_parameters(self, config: UQConfiguration):
        """Validate Monte Carlo + Regression specific parameters."""
        
        # First validate basic MC parameters
        if not hasattr(config, 'mc_numevaluations'):
            self.errors.append("mc_numevaluations must be specified for MC regression method")
        elif config.mc_numevaluations <= 0:
            self.errors.append("mc_numevaluations must be positive")
        
        if not hasattr(config, 'sampling_rule'):
            self.errors.append("sampling_rule must be specified for MC regression method")
        elif config.sampling_rule not in SAMPLING_RULES:
            self.errors.append(f"sampling_rule must be one of {SAMPLING_RULES}")
        
        # Validate regression-specific parameters
        if not hasattr(config, 'sc_p_order'):
            self.errors.append("sc_p_order must be specified for MC regression method")
        elif config.sc_p_order <= 0:
            self.errors.append("sc_p_order must be positive")
        
        # Check polynomial rule
        if not hasattr(config, 'sc_poly_rule'):
            self.errors.append("sc_poly_rule must be specified for MC regression method")
        elif config.sc_poly_rule not in POLY_RULES:
            self.errors.append(f"sc_poly_rule must be one of {POLY_RULES}")
        
        # Ensure regression is enabled
        if hasattr(config, 'regression') and not config.regression:
            self.errors.append("regression must be True for MC regression method")
    
    def _validate_psp_parameters(self, config: UQConfiguration):
        """Validate Pseudo-Spectral Projection specific parameters."""
        
        # PSP is essentially SC without regression
        if not hasattr(config, 'sc_p_order'):
            self.errors.append("sc_p_order must be specified for PSP method")
        elif config.sc_p_order <= 0:
            self.errors.append("sc_p_order must be positive")
        
        # Check polynomial rule
        if not hasattr(config, 'sc_poly_rule'):
            self.errors.append("sc_poly_rule must be specified for PSP method")
        elif config.sc_poly_rule not in POLY_RULES:
            self.errors.append(f"sc_poly_rule must be one of {POLY_RULES}")
        
        # Only validate quadrature parameters if not reading nodes from file
        if not config.read_nodes_from_file or config.parameters_file is None:
            if not hasattr(config, 'sc_q_order'):
                self.errors.append("sc_q_order must be specified for PSP method")
            elif config.sc_q_order <= 0:
                self.errors.append("sc_q_order must be positive")

            if not hasattr(config, 'sc_quadrature_rule'):
                self.errors.append("sc_quadrature_rule must be specified for PSP method")
            elif config.sc_quadrature_rule not in QUADRATURE_RULES:
                self.errors.append(f"sc_quadrature_rule must be one of {QUADRATURE_RULES}")
        
        # Ensure regression is disabled for PSP
        if hasattr(config, 'regression') and config.regression:
            self.errors.append("regression must be False for PSP method")
    
    def _validate_sparse_grid_parameters(self, config: UQConfiguration):
        """Validate Sparse Grid specific parameters."""
        
        # Sparse grid configurations should have file-based nodes
        if not config.read_nodes_from_file:
            self.errors.append("read_nodes_from_file must be True for sparse grid method")
        
        if config.parameters_file is None:
            self.errors.append("parameters_file must be specified for sparse grid method")
        
        # Check sparse grid specific parameters if they exist
        if hasattr(config, 'sparse_grid_level'):
            if config.sparse_grid_level is not None and config.sparse_grid_level <= 0:
                self.errors.append("sparse_grid_level must be positive")
        
        if hasattr(config, 'sparse_grid_dimension'):
            if config.sparse_grid_dimension is not None and config.sparse_grid_dimension <= 0:
                self.errors.append("sparse_grid_dimension must be positive")
        
        # Validate polynomial parameters
        if not hasattr(config, 'sc_p_order'):
            self.errors.append("sc_p_order must be specified for sparse grid method")
        elif config.sc_p_order <= 0:
            self.errors.append("sc_p_order must be positive")
        
        # Check polynomial rule
        if not hasattr(config, 'sc_poly_rule'):
            self.errors.append("sc_poly_rule must be specified for sparse grid method")
        elif config.sc_poly_rule not in POLY_RULES:
            self.errors.append(f"sc_poly_rule must be one of {POLY_RULES}")
        
        # Sparse quadrature should be enabled
        if hasattr(config, 'sc_sparse_quadrature') and not config.sc_sparse_quadrature:
            self.warnings.append("sc_sparse_quadrature should typically be True for sparse grid method")
    
    def _validate_ensemble_parameters(self, config: UQConfiguration):
        """Validate Ensemble specific parameters."""
        # Ensemble method has fewer constraints
        pass
    
    def _validate_file_based_parameters(self, config: UQConfiguration):
        """Validate file-based configuration parameters."""
        
        # Check if configuration uses file-based nodes
        if hasattr(config, 'read_nodes_from_file') and config.read_nodes_from_file:
            if config.parameters_file is None:
                self.errors.append("parameters_file must be specified when read_nodes_from_file is True")
            else:
                # Check if parameters file exists
                param_path = pathlib.Path(config.parameters_file)
                if not param_path.exists():
                    self.warnings.append(f"parameters_file does not exist: {config.parameters_file}")
                elif not param_path.is_file():
                    self.errors.append(f"parameters_file is not a file: {config.parameters_file}")
                else:
                    # Check file extension for common formats
                    valid_extensions = ['.asc', '.txt', '.dat', '.csv']
                    if param_path.suffix.lower() not in valid_extensions:
                        self.warnings.append(f"parameters_file has unusual extension: {param_path.suffix}. Expected one of {valid_extensions}")
        
        # Validate parameters_setup_file if present
        if hasattr(config, 'parameters_setup_file') and config.parameters_setup_file is not None:
            setup_path = pathlib.Path(config.parameters_setup_file)
            if not setup_path.exists():
                self.warnings.append(f"parameters_setup_file does not exist: {config.parameters_setup_file}")
            elif not setup_path.is_file():
                self.errors.append(f"parameters_setup_file is not a file: {config.parameters_setup_file}")
        
        # Check for file-based configuration class indicators
        config_class_name = config.__class__.__name__
        if 'FileBased' in config_class_name:
            if not hasattr(config, 'read_nodes_from_file') or not config.read_nodes_from_file:
                self.warnings.append(f"Configuration class {config_class_name} suggests file-based usage but read_nodes_from_file is False")
        
        # Validate sparse grid file naming convention
        if (hasattr(config, 'parameters_file') and config.parameters_file is not None and 
            hasattr(config, 'sparse_grid_level') and hasattr(config, 'sparse_grid_dimension')):
            param_path = pathlib.Path(config.parameters_file)
            filename = param_path.name
            
            # Check if filename follows sparse grid convention (e.g., "KPU_d10_l7.asc")
            import re
            match = re.search(r'd(\d+)_l(\d+)', filename)
            if match:
                file_dimension = int(match.group(1))
                file_level = int(match.group(2))
                
                if (hasattr(config, 'sparse_grid_dimension') and config.sparse_grid_dimension is not None and 
                    file_dimension != config.sparse_grid_dimension):
                    self.warnings.append(f"Filename dimension ({file_dimension}) doesn't match sparse_grid_dimension ({config.sparse_grid_dimension})")
                
                if (hasattr(config, 'sparse_grid_level') and config.sparse_grid_level is not None and 
                    file_level != config.sparse_grid_level):
                    self.warnings.append(f"Filename level ({file_level}) doesn't match sparse_grid_level ({config.sparse_grid_level})")
        
    def _validate_paths(self, config: UQConfiguration):
        """Validate file and directory paths."""
        
        # Validate path types
        # Check outputResultDir type (allow str, os.PathLike, pathlib.Path)
        if config.outputResultDir is not None:
            if not isinstance(config.outputResultDir, (str, os.PathLike, pathlib.Path)):
                self.warnings.append(f"outputResultDir should be of type str, os.PathLike, or pathlib.Path, got {type(config.outputResultDir).__name__}")
        
        # Check workingDir type (allow str, os.PathLike only)
        if config.workingDir is not None:
            if not isinstance(config.workingDir, (str, os.PathLike)):
                self.warnings.append(f"workingDir should be of type str or os.PathLike, got {type(config.workingDir).__name__}")
        
        # Check if paths exist (only warn, don't error)
        paths_to_check = [
            ('inputModelDir', config.inputModelDir),
            ('sourceDir', config.sourceDir),
            ('config_file', config.config_file),
            ('parameters_file', config.parameters_file)
        ]
        
        for path_name, path_value in paths_to_check:
            if path_value is not None:
                path_obj = pathlib.Path(path_value)
                if not path_obj.exists():
                    self.warnings.append(f"{path_name} does not exist: {path_value}")
        
        # Check output directory parent exists
        if config.outputResultDir is not None:
            output_path = pathlib.Path(config.outputResultDir)
            if not output_path.parent.exists():
                self.errors.append(f"Parent directory of outputResultDir does not exist: {output_path.parent}")
    
    def _validate_parameter_consistency(self, config: UQConfiguration):
        """Validate parameter consistency."""
        
        # MPI and parallel settings
        if config.mpi and config.parallel:
            self.warnings.append("Both MPI and parallel are enabled - MPI will take precedence")
        
        # Regression settings
        if hasattr(config, 'regression') and config.regression and hasattr(config, 'regression_model_type') and config.regression_model_type is None:
            self.warnings.append("regression is enabled but regression_model_type is not specified")
        
        if hasattr(config, 'regression_model_type') and config.regression_model_type is not None:
            if config.regression_model_type not in REGRESSION_TYPES:
                self.errors.append(f"regression_model_type must be one of {REGRESSION_TYPES}")
        
        # Statistics settings
        if config.disable_statistics and (config.compute_Sobol_m or config.compute_Sobol_t):
            self.warnings.append("Statistics are disabled but Sobol indices computation is enabled; Sobol indices won't be computed")
        
        # Sparse grid settings
        if hasattr(config, 'sc_sparse_quadrature') and config.sc_sparse_quadrature:
            if not config.read_nodes_from_file:
                self.warnings.append("Sparse quadrature is enabled but read_nodes_from_file is False; Sparse fleg from chaospy will be triggered/used")
    
        # File-based node reading consistency
        if hasattr(config, 'read_nodes_from_file') and config.read_nodes_from_file and config.parameters_file is None:
            self.errors.append("parameters_file must be specified when read_nodes_from_file is True")
    
    def _validate_model_specific(self, config: UQConfiguration):
        """Validate model-specific requirements."""
        
        if config.model == "battery":
            # Check if PyBaMM is available (this would be done at runtime)
            self.warnings.append("Ensure PyBaMM is installed for battery model")
        
        # elif config.model == "hbvsask":
        #     # HBV-SASK specific validations
        #     if config.inputModelDir is not None:
        #         hbv_path = pathlib.Path(config.inputModelDir)
        #         if not (hbv_path / "Oldman_Basin").exists():
        #             self.warnings.append("Default basin directory 'Oldman_Basin' not found in inputModelDir")
        
        # elif config.model == "larsim":
        #     # LARSIM specific validations
        #     if config.inputModelDir is not None:
        #         larsim_path = pathlib.Path(config.inputModelDir)
        #         if not larsim_path.exists():
        #             self.warnings.append("LARSIM input directory does not exist")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy(),
            'is_valid': len(self.errors) == 0
        }
