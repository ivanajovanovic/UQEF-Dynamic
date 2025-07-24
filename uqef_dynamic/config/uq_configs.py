"""
UQ method-specific configuration classes.
"""

import pathlib
from typing import Optional, Union
from .base_config import UQConfiguration

SAMPLING_RULES =  ["random", "sobol", "latin_hypercube", "halton", "hammersley"]
QUADRATURE_RULES = ["g", "p", "genz_keister_24", "leja", "clenshaw_curtis"]
POLY_RULES = ["three_terms_recurrence", "gram_schmidt", "cholesky"]
REGRESSION_TYPES = ["OLS", "LARS"]


class FileBasedNodesMixin:
    """Mixin for configurations that can read nodes/samples from files."""
    
    def __init__(self):
        # File-based node reading parameters
        self.read_nodes_from_file: bool = True
        self.parameters_file: Optional[Union[str, pathlib.Path]] = None

    def set_parameters_file(parameters_file: Optional[Union[str, pathlib.Path]]):
        self.parameters_file = parameters_file
        
    def validate_file_based_nodes(self) -> bool:
        """Validate file-based node parameters."""
        if self.read_nodes_from_file and self.parameters_file is None:
            raise ValueError("parameters_file must be set when read_nodes_from_file is True")
        return True

class MCConfiguration(UQConfiguration):
    """Configuration for Monte Carlo simulations."""
    
    def __init__(self):
        super().__init__()
        self.uq_method = "mc"
        self.config_uq_method = "mc"
        
        # MC-specific parameters
        self.mc_numevaluations: int = 10000
        self.sampling_rule: str = "latin_hypercube"  # "random" | "sobol" | "halton" | "hammersley"

        self.regression = False
        
        # For MC method, if compute_Sobol_m is True, then compute_sobol_indices_with_samples should be True
        # This will be dynamically set based on compute_Sobol_m in the configuration factory
                
    def validate(self) -> bool:
        """Validate MC-specific parameters."""
        if self.mc_numevaluations <= 0:
            raise ValueError("mc_numevaluations must be positive")
        
        valid_sampling_rules = SAMPLING_RULES
        if self.sampling_rule not in valid_sampling_rules:
            raise ValueError(f"sampling_rule must be one of {valid_sampling_rules}")

        if self.regression:
            if not hasattr(self, 'sc_p_order') or self.sc_p_order <= 0:
                raise ValueError(f"Regression is turned on but sc_p_order is not propely defined")

        return super().validate()

class MCRegressionConfiguration(MCConfiguration):
    """Configuration for Monte Carlo + PCE regression simulations."""

    def __init__(self):
        super().__init__()
        self.config_uq_method = "mc_regression"
        self.sc_p_order: int = 2  # number of terms in PCE
        self.regression = True
        
    def validate(self) -> bool:
        """Validate MC+PSP-specific parameters."""
        if not self.regression:
            raise ValueError(f"You want to run MC+PSP method but the regression option is turned on!")

        if not hasattr(self, 'sc_p_order') or self.sc_p_order <= 0:
            raise ValueError(f"Regression is turned on but sc_p_order is not propely defined")

        return super().validate()


class SCConfiguration(UQConfiguration):
    """Configuration for Stochastic Collocation simulations."""
    
    def __init__(self):
        super().__init__()
        self.uq_method = "sc"
        self.config_uq_method = "sc"

        # SC-specific parameters
        self.sc_q_order: int = 5  # number of collocation points in each direction
        self.sc_p_order: int = 2  # number of terms in PCE
        self.sc_quadrature_rule: str = "g"  # "p" | "genz_keister_24" | "leja" | "clenshaw_curtis"
        self.sc_sparse_quadrature: bool = False
        self.regression = True
        
        # # Override defaults for SC
        # self.store_gpce_surrogate_in_stat_dict = False
        
    def validate(self) -> bool:
        """Validate SC-specific parameters."""
        if self.sc_q_order <= 0:
            raise ValueError("sc_q_order must be positive")
        
        if self.sc_p_order <= 0:
            raise ValueError("sc_p_order must be positive")
        
        valid_quad_rules = QUADRATURE_RULES
        if self.sc_quadrature_rule not in valid_quad_rules:
            raise ValueError(f"sc_quadrature_rule must be one of {valid_quad_rules}")
        
        return super().validate()


class PSPConfiguration(SCConfiguration):
    """Configuration for Pseudo-spectral projection (PSP) simulations."""
    
    def __init__(self):
        super().__init__()
        self.config_uq_method = "psp"
        self.regression = False
        
    def validate(self) -> bool:
        """Validate PSP-specific parameters."""
        if self.regression:
            raise ValueError(f"You want to run PSP method but the regression option is turned on!")

        return super().validate()


class SaltelliConfiguration(UQConfiguration):
    """Configuration for Saltelli (Sobol) simulations."""
    
    def __init__(self):
        super().__init__()
        self.uq_method = "saltelli"
        self.config_uq_method = "saltelli"

        # Saltelli-specific parameters
        self.mc_numevaluations: int = 10000
        self.sampling_rule: str = "sobol"  # Usually sobol for Saltelli
        
        # Override defaults for Saltelli - enable Sobol indices computation
        self.compute_Sobol_m = True
        self.compute_Sobol_t = True
        self.compute_sobol_indices_with_samples = False  # For Saltelli method this should be False
        
    def validate(self) -> bool:
        """Validate Saltelli-specific parameters."""
        if self.mc_numevaluations <= 0:
            raise ValueError("mc_numevaluations must be positive")
        
        valid_sampling_rules = SAMPLING_RULES
        if self.sampling_rule not in valid_sampling_rules:
            raise ValueError(f"sampling_rule must be one of {valid_sampling_rules}")
        
        return super().validate()


class EnsembleConfiguration(UQConfiguration):
    """Configuration for Ensemble simulations."""
    
    def __init__(self):
        super().__init__()
        self.uq_method = "ensemble"
        self.config_uq_method = "ensemble"

        # Ensemble-specific settings; TODO Change this
        self.read_nodes_from_file = False  # For now, should be changes soon
        
        # Override defaults for Ensemble
        self.save_all_simulations = True
        self.compute_Sobol_m = False  # Usually not computed for ensemble
        self.compute_Sobol_t = False
        
    def validate(self) -> bool:
        """Validate Ensemble-specific parameters."""
        # Ensemble method has fewer constraints
        return super().validate()


# File-based configuration classes that combine UQ methods with file reading capability
class FileBasedMCConfiguration(MCConfiguration, FileBasedNodesMixin):
    """Monte Carlo configuration with file-based node reading capability."""
    
    def __init__(self):
        MCConfiguration.__init__(self)
        FileBasedNodesMixin.__init__(self)
        
    def validate(self) -> bool:
        """Validate both MC and file-based parameters."""
        return MCConfiguration.validate(self) and self.validate_file_based_nodes()


class FileBasedMCRegressionConfiguration(MCRegressionConfiguration, FileBasedNodesMixin):
    """Monte Carlo + Regression configuration with file-based node reading capability."""
    
    def __init__(self):
        MCRegressionConfiguration.__init__(self)
        FileBasedNodesMixin.__init__(self)
        
    def validate(self) -> bool:
        """Validate both MC regression and file-based parameters."""
        return MCRegressionConfiguration.validate(self) and self.validate_file_based_nodes()


class FileBasedSCConfiguration(SCConfiguration, FileBasedNodesMixin):
    """Stochastic Collocation configuration with file-based node reading capability."""
    
    def __init__(self):
        SCConfiguration.__init__(self)
        FileBasedNodesMixin.__init__(self)
        
    def validate(self) -> bool:
        """Validate both SC and file-based parameters."""
        return SCConfiguration.validate(self) and self.validate_file_based_nodes()


class FileBasedPSPConfiguration(PSPConfiguration, FileBasedNodesMixin):
    """Pseudo-spectral Projection configuration with file-based node reading capability."""
    
    def __init__(self):
        PSPConfiguration.__init__(self)
        FileBasedNodesMixin.__init__(self)
        
    def validate(self) -> bool:
        """Validate both PSP and file-based parameters."""
        return PSPConfiguration.validate(self) and self.validate_file_based_nodes()


class SparseGridConfigurationBuilder:
    """Builder for creating sparse grid configurations of any UQ method type."""
    
    # Registry of file-based configuration classes
    CONFIG_CLASSES = {
        "mc": FileBasedMCConfiguration,
        "mc_regression": FileBasedMCRegressionConfiguration,
        "sc": FileBasedSCConfiguration,
        "psp": FileBasedPSPConfiguration
    }
    
    @classmethod
    def create_sparse_grid_config(
        cls, 
        base_uq_method: str, 
        level: int, 
        dimension: int, 
        base_path: Union[str, pathlib.Path], 
        **kwargs
    ):
        """
        Create a sparse grid configuration for any UQ method.
        
        Args:
            base_uq_method: The base UQ method ("mc", "mc_regression", "sc", "psp")
            level: Sparse grid level
            dimension: Problem dimension
            base_path: Base path to sparse grid files
            **kwargs: Additional configuration parameters
            
        Returns:
            File-based configuration with sparse grid settings
        """
        if base_uq_method not in cls.CONFIG_CLASSES:
            raise ValueError(f"Unsupported UQ method: {base_uq_method}. "
                           f"Available: {list(cls.CONFIG_CLASSES.keys())}")
        
        # Create the appropriate file-based configuration
        config = cls.CONFIG_CLASSES[base_uq_method]()
        
        # Set sparse grid specific parameters
        config.sparse_grid_level = level
        config.sparse_grid_dimension = dimension
        
        # Set the sparse grid file path
        base_path = pathlib.Path(base_path)
        config.parameters_file = base_path / f"KPU_d{dimension}_l{level}.asc"
        
        # For SC-based methods, enable sparse quadrature
        if hasattr(config, 'sc_sparse_quadrature'):
            config.sc_sparse_quadrature = True
        
        # Apply any additional parameters
        config.update(**kwargs)
        
        return config
    
    @classmethod
    def get_available_uq_methods(cls):
        """Get list of available UQ methods for sparse grid configurations."""
        return list(cls.CONFIG_CLASSES.keys())


# Backward compatibility: Keep the original SparseGridConfiguration as an alias
class SparseGridConfiguration(FileBasedSCConfiguration):
    """
    Configuration for Sparse Grid + Pseudo-spectral projection simulations.
    
    DEPRECATED: This class is kept for backward compatibility.
    Use SparseGridConfigurationBuilder.create_sparse_grid_config() for new code.
    """
    
    def __init__(self):
        super().__init__()
        
        # Sparse grid specific parameters
        self.sc_sparse_quadrature = True
        
        # Sparse grid file parameters
        self.sparse_grid_level: Optional[int] = None
        self.sparse_grid_dimension: Optional[int] = None
        
    def set_sparse_grid_file(self, level: int, dimension: int, base_path: Union[str, pathlib.Path]):
        """Set the sparse grid file based on level and dimension."""
        self.sparse_grid_level = level
        self.sparse_grid_dimension = dimension
        
        base_path = pathlib.Path(base_path)
        self.parameters_file = base_path / f"KPU_d{dimension}_l{level}.asc"
        
    def validate(self) -> bool:
        """Validate Sparse Grid-specific parameters."""
        if self.sparse_grid_level is not None and self.sparse_grid_level <= 0:
            raise ValueError("sparse_grid_level must be positive")
        
        if self.sparse_grid_dimension is not None and self.sparse_grid_dimension <= 0:
            raise ValueError("sparse_grid_dimension must be positive")
        
        return super().validate()
