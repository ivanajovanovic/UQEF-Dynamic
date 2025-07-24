"""
Configuration factory for creating and managing UQ configurations.
"""

import pathlib
from typing import Dict, Any, Optional, Union

from .uq_configs import (
    MCConfiguration, MCRegressionConfiguration, 
    SCConfiguration, PSPConfiguration, 
    SaltelliConfiguration, 
    EnsembleConfiguration, SparseGridConfiguration, SparseGridConfigurationBuilder,
    FileBasedMCConfiguration, FileBasedMCRegressionConfiguration,
    FileBasedSCConfiguration, FileBasedPSPConfiguration
)
from .model_configs import (
    HBVSASKConfig, BatteryConfig, LarsimConfig, IshigamiConfig,
    SimpleOscillatorConfig, ProductFunctionConfig, LinearDampedOscillatorConfig
)
from .config_validator import ConfigurationValidator


class ConfigurationFactory:
    """Factory for creating UQ configurations."""

    # Registry of UQ method configurations
    UQ_CONFIGS = {
        "mc": MCConfiguration,
        "mc_regression": MCRegressionConfiguration,
        "sc": SCConfiguration,
        "psp": PSPConfiguration,
        "saltelli": SaltelliConfiguration,
        "ensemble": EnsembleConfiguration,
        "sparse_grid": SparseGridConfiguration  # legacy, these is PSP which relies on pre-computed sparse grid quadrature nodes
    }
    
    # Registry of model configurations
    MODEL_CONFIGS = {
        "hbvsask": HBVSASKConfig,
        "battery": BatteryConfig,
        "larsim": LarsimConfig,
        "ishigami": IshigamiConfig,
        "simple_oscillator": SimpleOscillatorConfig,
        "productFunction": ProductFunctionConfig,
        "oscillator": LinearDampedOscillatorConfig
    }
    
    @classmethod
    def create_configuration(
        cls, 
        model_type: str, 
        uq_method: str, 
        validate: bool = True,
        **overrides
    ):
        """
        Create a complete configuration for a model and UQ method.
        
        Args:
            model_type: Type of model (e.g., 'hbvsask', 'battery')
            uq_method: UQ method (e.g., 'mc', 'mc_regression', 'sc', 'psp', 'saltelli', 'ensemble')
            validate: Whether to validate the configuration
            **overrides: Additional parameters to override defaults
            
        Returns:
            UQConfiguration: Configured UQ configuration object
        """
        # Create UQ configuration
        if uq_method not in cls.UQ_CONFIGS:
            raise ValueError(f"Unknown UQ method: {uq_method}. Available: {list(cls.UQ_CONFIGS.keys())}")
        
        uq_config = cls.UQ_CONFIGS[uq_method]()
        
        # Set model type
        uq_config.model = model_type
        
        # Get model-specific configuration
        if model_type in cls.MODEL_CONFIGS:
            model_config = cls.MODEL_CONFIGS[model_type]()
            cls._apply_model_config(uq_config, model_config, **overrides)
        
        # Apply any overrides
        uq_config.update(**overrides)
        
        # Apply method-specific logic for compute_sobol_indices_with_samples
        cls._apply_sobol_indices_logic(uq_config)
        
        # Validate if requested
        if validate:
            validator = ConfigurationValidator()
            validator.validate(uq_config)
        
        return uq_config
    
    @classmethod
    def _apply_sobol_indices_logic(cls, config):
        """Apply logic for compute_sobol_indices_with_samples based on UQ method and settings."""
        # For MC method: if compute_Sobol_m is True, then compute_sobol_indices_with_samples should be True
        if config.uq_method == "mc" and config.compute_Sobol_m:
            config.compute_sobol_indices_with_samples = True
        # For Saltelli method: compute_sobol_indices_with_samples should be False
        elif config.uq_method == "saltelli":
            config.compute_sobol_indices_with_samples = False
    
    @classmethod
    def _apply_model_config(cls, uq_config, model_config, **overrides):
        """Apply model-specific configuration to UQ configuration.
        This will actually just fetch default values from model configuration if 
        these arguments/paths are not specified in overrides
        Be carefule, if fetched from mode, some path such as model_config.config_file and model_config.outputResultDir
        will be of type pathlib.Path
        """
        
        # Set default paths if not overridden
        if hasattr(model_config, 'default_input_dir') and 'inputModelDir' not in overrides:
            uq_config.inputModelDir = model_config.default_input_dir
            
        if hasattr(model_config, 'default_source_dir') and 'sourceDir' not in overrides:
            uq_config.sourceDir = model_config.default_source_dir
        
        # Set configuration file if available and not overridden
        if 'config_file' not in overrides:
            try:
                config_file = model_config.get_config_file_path()
                if config_file:
                    uq_config.config_file = config_file
            except (AttributeError, TypeError) as e:
                print(f"Model configuration error when setting up config_file: {e}")
                raise
                #pass
        
        # Set output directory if specified
        if 'outputResultDir' not in overrides and hasattr(model_config, 'get_output_dir'):
            try:
                run_type = overrides.pop('run_type', 'default')
                output_dir = model_config.get_output_dir(run_type=run_type, **overrides)
                uq_config.outputResultDir = output_dir
                uq_config.outputModelDir = output_dir
            except (AttributeError, TypeError) as e:
                print(f"Model configuration error when setting up outputResultDir: {e}")
                raise
                #pass

        if hasattr(uq_config, 'config_file') and isinstance(uq_config.config_file, pathlib.Path):
            uq_config.config_file = str(uq_config.config_file)
            # uq_config.config_file = os.path.normpath(str(uq_config.config_file))
        if hasattr(uq_config, 'outputResultDir') and isinstance(uq_config.outputResultDir, pathlib.Path):
            uq_config.outputResultDir = str(uq_config.outputResultDir)
            # uq_config.outputResultDir = os.path.normpath(str(uq_config.outputResultDir))

    @classmethod
    def from_json_file(cls, json_file: Union[str, pathlib.Path]):
        """Create configuration from JSON file."""
        import json
        
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        
        model_type = config_dict.pop('model_type')
        uq_method = config_dict.pop('uq_method')
        
        return cls.create_configuration(model_type, uq_method, **config_dict)
    
    @classmethod
    def get_available_models(cls):
        """Get list of available model types."""
        return list(cls.MODEL_CONFIGS.keys())
    
    @classmethod
    def get_available_uq_methods(cls):
        """Get list of available UQ methods."""
        return list(cls.UQ_CONFIGS.keys())
    
    # New methods for sparse grid configurations
    @classmethod
    def create_sparse_grid_configuration(
        cls, 
        base_uq_method: str, 
        level: int, 
        dimension: int, 
        base_path: Union[str, pathlib.Path] = "/dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights",
        model_type: str = "hbvsask",
        validate: bool = True,
        **overrides
    ):
        """
        Create a sparse grid configuration for any UQ method.
        
        Args:
            base_uq_method: The base UQ method ("mc", "mc_regression", "sc", "psp")
            level: Sparse grid level
            dimension: Problem dimension
            base_path: Base path to sparse grid files
            model_type: Type of model (default: "hbvsask")
            validate: Whether to validate the configuration
            **overrides: Additional configuration parameters
            
        Returns:
            File-based configuration with sparse grid settings
        """
        # Create the sparse grid configuration using the builder
        config = SparseGridConfigurationBuilder.create_sparse_grid_config(
            base_uq_method, level, dimension, base_path, **overrides
        )
        
        # Set model type
        config.model = model_type
        
        # Apply model-specific configuration
        if model_type in cls.MODEL_CONFIGS:
            model_config = cls.MODEL_CONFIGS[model_type]()
            cls._apply_model_config(config, model_config, **overrides)
        
        # Validate if requested
        if validate:
            validator = ConfigurationValidator()
            validator.validate(config)
        
        return config
    
    @classmethod
    def create_file_based_configuration(
        cls,
        base_uq_method: str,
        parameters_file: Union[str, pathlib.Path],
        model_type: str = "hbvsask",
        validate: bool = True,
        **overrides
    ):
        """
        Create a file-based configuration for any UQ method.
        
        Args:
            base_uq_method: The base UQ method  ('mc', 'mc_regression', 'sc', 'psp')
            parameters_file: Path to the parameters/nodes file
            model_type: Type of model (default: "hbvsask")
            validate: Whether to validate the configuration
            **overrides: Additional configuration parameters
            
        Returns:
            File-based configuration
        """
        # Registry of file-based configuration classes
        file_based_configs = {
            "mc": FileBasedMCConfiguration,
            "mc_regression": FileBasedMCRegressionConfiguration,
            "sc": FileBasedSCConfiguration,
            "psp": FileBasedPSPConfiguration
        }
        
        if base_uq_method not in file_based_configs:
            raise ValueError(f"Unsupported UQ method: {base_uq_method}. "
                           f"Available: {list(file_based_configs.keys())}")
        
        # Create the file-based configuration
        config = file_based_configs[base_uq_method]()
        config.parameters_file = parameters_file
        config.model = model_type
        
        # Apply model-specific configuration
        if model_type in cls.MODEL_CONFIGS:
            model_config = cls.MODEL_CONFIGS[model_type]()
            cls._apply_model_config(config, model_config, **overrides)
        
        # Apply any overrides
        config.update(**overrides)
        
        # Validate if requested
        if validate:
            validator = ConfigurationValidator()
            validator.validate(config)
        
        return config
    
    @classmethod
    def get_available_sparse_grid_uq_methods(cls):
        """Get list of available UQ methods for sparse grid configurations."""
        return SparseGridConfigurationBuilder.get_available_uq_methods()

    # ======================================================================    
 
    @classmethod
    def from_uqsim_args(cls, args, validate: bool = True):
        """
        Create configuration from UQEF UQsim parsed arguments.
        
        This method bridges the gap between the command-line arguments parsed by
        UQEF's argparse system and the new configuration management system.
        It properly handles extended UQ method variants like 'mc_regression' and 'psp'.
        
        Args:
            args: Parsed arguments from UQsim (uqsim.args)
            validate: Whether to validate the configuration
            
        Returns:
            UQConfiguration: Configuration object created from arguments
        """
        # Extract basic configuration parameters
        model_type = args.model
        base_uq_method = args.uq_method
        
        # Determine the actual configuration type based on method and flags
        config_uq_method = cls._determine_config_uq_method(base_uq_method, args)
        
        # Build configuration parameters from args
        config_params = {}
        
        # Basic paths and directories
        if hasattr(args, 'inputModelDir') and args.inputModelDir:
            config_params['inputModelDir'] = args.inputModelDir
        if hasattr(args, 'outputResultDir') and args.outputResultDir:
            config_params['outputResultDir'] = args.outputResultDir
        if hasattr(args, 'outputModelDir') and args.outputModelDir:
            config_params['outputModelDir'] = args.outputModelDir
        if hasattr(args, 'sourceDir') and args.sourceDir:
            config_params['sourceDir'] = args.sourceDir
        if hasattr(args, 'config_file') and args.config_file:
            config_params['config_file'] = args.config_file
        
        # UQ method specific parameters
        # if base_uq_method in ['mc', 'saltelli']:
        if hasattr(args, 'mc_numevaluations'):
            config_params['mc_numevaluations'] = args.mc_numevaluations
        if hasattr(args, 'sampling_rule'):
            config_params['sampling_rule'] = args.sampling_rule
                
        # elif base_uq_method == 'sc':
        if hasattr(args, 'sc_q_order'):
            config_params['sc_q_order'] = args.sc_q_order
        if hasattr(args, 'sc_p_order'):
            config_params['sc_p_order'] = args.sc_p_order
        if hasattr(args, 'sc_quadrature_rule'):
            config_params['sc_quadrature_rule'] = args.sc_quadrature_rule
        if hasattr(args, 'sc_sparse_quadrature'):
            config_params['sc_sparse_quadrature'] = args.sc_sparse_quadrature
        if hasattr(args, 'sc_poly_normed'):
            config_params['sc_poly_normed'] = args.sc_poly_normed
        if hasattr(args, 'sc_poly_rule'):
            config_params['sc_poly_rule'] = args.sc_poly_rule
        
        # Common parameters across methods
        if hasattr(args, 'cross_truncation'):
            config_params['cross_truncation'] = args.cross_truncation
        if hasattr(args, 'regression'):
            config_params['regression'] = args.regression
        if hasattr(args, 'regression_model_type'):
            config_params['regression_model_type'] = args.regression_model_type
        if hasattr(args, 'transformToStandardDist'):
            config_params['transformToStandardDist'] = args.transformToStandardDist
        if hasattr(args, 'sampleFromStandardDist'):
            config_params['sampleFromStandardDist'] = args.sampleFromStandardDist
        
        # File-based parameters
        if hasattr(args, 'read_nodes_from_file'):
            config_params['read_nodes_from_file'] = args.read_nodes_from_file
        if hasattr(args, 'parameters_file') and args.parameters_file:
            config_params['parameters_file'] = args.parameters_file
        if hasattr(args, 'parameters_setup_file') and args.parameters_setup_file:
            config_params['parameters_setup_file'] = args.parameters_setup_file
        
        # Parallel and MPI settings
        if hasattr(args, 'parallel'):
            config_params['parallel'] = args.parallel
        if hasattr(args, 'num_cores'):
            config_params['num_cores'] = args.num_cores
        if hasattr(args, 'mpi'):
            config_params['mpi'] = args.mpi
        if hasattr(args, 'mpi_method'):
            config_params['mpi_method'] = args.mpi_method
        if hasattr(args, 'mpi_combined_parallel'):
            config_params['mpi_combined_parallel'] = args.mpi_combined_parallel
        if hasattr(args, 'chunksize'):
            config_params['chunksize'] = args.chunksize
        if hasattr(args, 'mpi_chunksize'):
            config_params['mpi_chunksize'] = args.mpi_chunksize
        
        # Statistics settings
        if hasattr(args, 'parallel_statistics'):
            config_params['parallel_statistics'] = args.parallel_statistics
        if hasattr(args, 'compute_Sobol_t'):
            config_params['compute_Sobol_t'] = args.compute_Sobol_t
        if hasattr(args, 'compute_Sobol_m'):
            config_params['compute_Sobol_m'] = args.compute_Sobol_m
        if hasattr(args, 'compute_Sobol_m2'):
            config_params['compute_Sobol_m2'] = args.compute_Sobol_m2
        if hasattr(args, 'save_all_simulations'):
            config_params['save_all_simulations'] = args.save_all_simulations
        if hasattr(args, 'store_qoi_data_in_stat_dict'):
            config_params['store_qoi_data_in_stat_dict'] = args.store_qoi_data_in_stat_dict
        if hasattr(args, 'store_gpce_surrogate_in_stat_dict'):
            config_params['store_gpce_surrogate_in_stat_dict'] = args.store_gpce_surrogate_in_stat_dict
        if hasattr(args, 'collect_and_save_state_data'):
            config_params['collect_and_save_state_data'] = args.collect_and_save_state_data
        if hasattr(args, 'instantly_save_results_for_each_time_step'):
            config_params['instantly_save_results_for_each_time_step'] = args.instantly_save_results_for_each_time_step
        if hasattr(args, 'disable_statistics'):
            config_params['disable_statistics'] = args.disable_statistics
        if hasattr(args, 'disable_calc_statistics'):
            config_params['disable_calc_statistics'] = args.disable_calc_statistics
        if hasattr(args, 'uqsim_store_to_file'):
            config_params['uqsim_store_to_file'] = args.uqsim_store_to_file

        # Model variant
        if hasattr(args, 'model_variant'):
            config_params['model_variant'] = args.model_variant
        
        # Uncertain parameters
        if hasattr(args, 'uncertain'):
            config_params['uncertain'] = args.uncertain
        
        # Handle special case for sparse grid configurations
        if (hasattr(args, 'parameters_file') and args.parameters_file and 
            'sparse_grid_nodes_weights' in str(args.parameters_file)):
            # This looks like a sparse grid configuration
            try:
                # Extract level and dimension from filename (e.g., "KPU_d10_l7.asc")
                import re
                filename = str(args.parameters_file)
                match = re.search(r'd(\d+)_l(\d+)', filename)
                if match:
                    dimension = int(match.group(1))
                    level = int(match.group(2))
                    
                    # Create sparse grid configuration
                    config = cls.create_sparse_grid_configuration(
                        base_uq_method=config_uq_method,
                        level=level,
                        dimension=dimension,
                        model_type=model_type,
                        validate=False,  # Validate later
                        **config_params
                    )
                else:
                    # Fallback to file-based configuration
                    config = cls.create_file_based_configuration(
                        base_uq_method=config_uq_method,
                        parameters_file=args.parameters_file,
                        model_type=model_type,
                        validate=False,
                        **config_params
                    )
            except Exception:
                # If sparse grid creation fails, fall back to regular configuration
                config = cls.create_configuration(model_type, config_uq_method, validate=False, **config_params)
        
        elif hasattr(args, 'parameters_file') and args.parameters_file:
            # File-based configuration
            config = cls.create_file_based_configuration(
                base_uq_method=config_uq_method,
                parameters_file=args.parameters_file,
                model_type=model_type,
                validate=False,
                **config_params
            )
        else:
            # Standard configuration
            config = cls.create_configuration(model_type, config_uq_method, validate=False, **config_params)
        
        # Generate run type based on configuration if not set
        if not hasattr(config, 'run_type') or not config.run_type:
            config.run_type = cls._generate_run_type_from_args(args, config)
        
        # Apply method-specific logic for compute_sobol_indices_with_samples
        cls._apply_sobol_indices_logic(config)
        
        # Validate if requested
        if validate:
            validator = ConfigurationValidator()
            validator.validate(config)
        
        return config
    
    @classmethod
    def apply_configuration_overrides(cls, config, **overrides):
        """
        Apply configuration overrides to an existing configuration.
        
        This method allows updating configuration parameters that are not yet
        supported in the UQEF argument system, enabling the use of new features
        even when starting from batch scripts or command-line arguments.
        
        Args:
            config: Existing configuration object
            **overrides: Configuration parameters to override
            
        Returns:
            Updated configuration object
        """
        # Apply overrides
        config.update(**overrides)
        
        # Re-apply method-specific logic after overrides
        cls._apply_sobol_indices_logic(config)
        
        return config
    
    @classmethod
    def _determine_config_uq_method(cls, base_uq_method: str, args):
        """
        Determine the actual configuration UQ method based on base method and flags.
        
        This handles the mapping from UQEF's base UQ methods to the extended
        configuration variants based on flags like regression.
        
        Args:
            base_uq_method: Base UQ method from args.uq_method
            args: Parsed arguments from UQsim
            
        Returns:
            str: The configuration UQ method to use
        """
        regression = getattr(args, 'regression', False)
        
        if base_uq_method == 'mc':
            if regression:
                return 'mc_regression'
            else:
                return 'mc'
        elif base_uq_method == 'sc':
            if not regression:
                return 'psp'  # PSP is SC with regression
            else:
                return 'sc'
        elif base_uq_method == 'saltelli':
            return 'saltelli'
        elif base_uq_method == 'ensemble':
            return 'ensemble'
        else:
            # For any unknown methods, return as-is
            return base_uq_method
    
    @classmethod
    def _generate_run_type_from_args(cls, args, config):
        """Generate a descriptive run type from arguments."""
        parts = []
        
        # Determine the actual method being used
        base_method = args.uq_method
        regression = getattr(args, 'regression', False)
        
        if base_method == 'mc':
            if regression:
                parts.append('mc_regression')
                if hasattr(args, 'sc_p_order'):
                    parts.append(f'p{args.sc_p_order}')
                if hasattr(args, 'cross_truncation') and args.cross_truncation != 1.0:
                    parts.append(f'ct{str(args.cross_truncation).replace(".", "")}')
            else:
                parts.append('mc')
            if hasattr(args, 'mc_numevaluations'):
                parts.append(str(args.mc_numevaluations))
            if hasattr(args, 'sampling_rule'):
                parts.append(args.sampling_rule)
        elif base_method == 'sc':
            if not regression:
                parts.append('psp')
            else:
                parts.append('sc')
            if hasattr(args, 'sc_p_order'):
                parts.append(f'p{args.sc_p_order}')
            if hasattr(args, 'sc_q_order'):
                parts.append(f'q{args.sc_q_order}')
            if hasattr(args, 'cross_truncation') and args.cross_truncation != 1.0:
                parts.append(f'ct{str(args.cross_truncation).replace(".", "")}')
        elif base_method == 'saltelli':
            parts.append('saltelli')
            if hasattr(args, 'mc_numevaluations'):
                parts.append(str(args.mc_numevaluations))
        else:
            parts.append(base_method)
        
        # Add model info
        parts.append(args.model)
        
        # Add sparse grid info if applicable
        if (hasattr(args, 'parameters_file') and args.parameters_file and 
            'sparse_grid_nodes_weights' in str(args.parameters_file)):
            import re
            filename = str(args.parameters_file)
            match = re.search(r'd(\d+)_l(\d+)', filename)
            if match:
                dimension = int(match.group(1))
                level = int(match.group(2))
                parts.append(f'd{dimension}_l{level}')
        
        return '_'.join(parts)

    # ======================================================================    

    @classmethod
    def create_hbv_mc_configuration(cls, **overrides):
        """Create a standard HBV-SASK Monte Carlo configuration."""
        defaults = {
            'mc_numevaluations': 10000,
            'sampling_rule': 'latin_hypercube',
            'run_type': 'mc_short',
            'config_variant': '10D'
        }
        defaults.update(overrides)
        
        config = cls.create_configuration('hbvsask', 'mc', **defaults)
        
        # Set HBV-specific config file
        if 'config_file' not in overrides:
            model_config = cls.MODEL_CONFIGS['hbvsask']()
            config.config_file = model_config.get_config_file_path(defaults['config_variant'])
        
        return config
    
    @classmethod
    def create_hbv_sc_configuration(cls, **overrides):
        """Create a standard HBV-SASK Stochastic Collocation configuration."""
        defaults = {
            'sc_q_order': 5,
            'sc_p_order': 2,
            'sc_quadrature_rule': 'g',
            'run_type': 'sc_sparse',
            'config_variant': '10D'
        }
        defaults.update(overrides)
        
        config = cls.create_configuration('hbvsask', 'sc', **defaults)
        
        # Set HBV-specific config file
        if 'config_file' not in overrides:
            model_config = cls.MODEL_CONFIGS['hbvsask']()
            config.config_file = model_config.get_config_file_path(defaults['config_variant'])
        
        return config
    
    @classmethod
    def create_hbv_sparse_grid_configuration(cls, level: int = 5, dimension: int = 6, **overrides):
        """Create HBV-SASK sparse grid configuration."""
        defaults = {
            'sc_p_order': 2,
            'cross_truncation': 0.7,
            'run_type': f'sc_sparse_l{level}_d{dimension}',
            'config_variant': '10D'
        }
        defaults.update(overrides)
        
        config = cls.create_configuration('hbvsask', 'sparse_grid', **defaults)
        
        # Set sparse grid file
        sparse_grid_path = pathlib.Path("/dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights")
        config.set_sparse_grid_file(level, dimension, sparse_grid_path)
        
        # Set HBV-specific config file
        if 'config_file' not in overrides:
            model_config = cls.MODEL_CONFIGS['hbvsask']()
            config.config_file = model_config.get_config_file_path(defaults['config_variant'])
        
        return config
    
    @classmethod
    def create_battery_mc_configuration(cls, **overrides):
        """Create a standard Battery Monte Carlo configuration."""
        defaults = {
            'mc_numevaluations': 10000,
            'sampling_rule': 'latin_hypercube',
            'run_type': 'mc_6d_10000_lhc'
        }
        defaults.update(overrides)
        
        return cls.create_configuration('battery', 'mc', **defaults)
    
    @classmethod
    def create_battery_sc_configuration(cls, **overrides):
        """Create a standard Battery Stochastic Collocation configuration."""
        defaults = {
            'sc_q_order': 10,
            'sc_p_order': 3,
            'run_type': 'sc_kl10_p3_l7_battery_terminal_vol'
        }
        defaults.update(overrides)
        
        return cls.create_configuration('battery', 'sc', **defaults)
    
    @classmethod
    def create_ishigami_configuration(cls, uq_method: str = 'sc', **overrides):
        """Create Ishigami test function configuration."""
        if uq_method == 'sc':
            defaults = {
                'sc_q_order': 10,
                'sc_p_order': 5,
                'cross_truncation': 0.7,
                'run_type': 'sc_full_p5_q10_ct07'
            }
        elif uq_method == 'mc':
            defaults = {
                'mc_numevaluations': 10000,
                'sampling_rule': 'sobol',
                'run_type': 'mc_10000_sobol'
            }
        else:
            defaults = {}
        
        defaults.update(overrides)
        return cls.create_configuration('ishigami', uq_method, **defaults)
