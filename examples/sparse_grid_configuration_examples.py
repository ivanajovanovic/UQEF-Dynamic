"""
Examples demonstrating the new flexible sparse grid configuration system.

This script shows how to create sparse grid configurations for different UQ methods
using the new SparseGridConfigurationBuilder and file-based configuration classes.
"""

import sys
import pathlib

# Add the UQEF-Dynamic path to sys.path if needed
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from uqef_dynamic.config import ConfigurationFactory
from uqef_dynamic.config.uq_configs import SparseGridConfigurationBuilder


def example_1_sparse_grid_mc():
    """Example 1: Monte Carlo with sparse grid nodes."""
    print("=== Example 1: Sparse Grid Monte Carlo ===")
    
    config = ConfigurationFactory.create_sparse_grid_configuration(
        base_uq_method="mc",
        level=5,
        dimension=6,
        model_type="hbvsask",
        mc_numevaluations=5000,
        sampling_rule="sobol"
    )
    
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"MC Evaluations: {config.mc_numevaluations}")
    print(f"Sampling Rule: {config.sampling_rule}")
    print(f"Read from file: {config.read_nodes_from_file}")
    print(f"Parameters file: {config.parameters_file}")
    print(f"Sparse grid level: {config.sparse_grid_level}")
    print(f"Sparse grid dimension: {config.sparse_grid_dimension}")
    print()


def example_2_sparse_grid_mc_regression():
    """Example 2: Monte Carlo + Regression with sparse grid nodes."""
    print("=== Example 2: Sparse Grid Monte Carlo + Regression ===")
    
    config = ConfigurationFactory.create_sparse_grid_configuration(
        base_uq_method="mc_regression",
        level=7,
        dimension=10,
        model_type="hbvsask",
        mc_numevaluations=10000,
        sc_p_order=3,
        sampling_rule="latin_hypercube"
    )
    
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"MC Evaluations: {config.mc_numevaluations}")
    print(f"PCE Order: {config.sc_p_order}")
    print(f"Regression: {config.regression}")
    print(f"Parameters file: {config.parameters_file}")
    print()


def example_3_sparse_grid_sc():
    """Example 3: Stochastic Collocation with sparse grid nodes."""
    print("=== Example 3: Sparse Grid Stochastic Collocation ===")
    
    config = ConfigurationFactory.create_sparse_grid_configuration(
        base_uq_method="sc",
        level=6,
        dimension=8,
        model_type="battery",
        sc_q_order=10,
        sc_p_order=4,
        sc_quadrature_rule="g"
    )
    
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"Q Order: {config.sc_q_order}")
    print(f"P Order: {config.sc_p_order}")
    print(f"Quadrature Rule: {config.sc_quadrature_rule}")
    print(f"Sparse Quadrature: {config.sc_sparse_quadrature}")
    print(f"Parameters file: {config.parameters_file}")
    print()


def example_4_sparse_grid_psp():
    """Example 4: Pseudo-spectral Projection with sparse grid nodes."""
    print("=== Example 4: Sparse Grid Pseudo-spectral Projection ===")
    
    config = ConfigurationFactory.create_sparse_grid_configuration(
        base_uq_method="psp",
        level=4,
        dimension=5,
        model_type="ishigami",
        sc_q_order=8,
        sc_p_order=3,
        cross_truncation=0.8
    )
    
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"Q Order: {config.sc_q_order}")
    print(f"P Order: {config.sc_p_order}")
    print(f"Regression: {config.regression}")
    print(f"Cross Truncation: {config.cross_truncation}")
    print(f"Parameters file: {config.parameters_file}")
    print()


def example_5_direct_builder_usage():
    """Example 5: Using SparseGridConfigurationBuilder directly."""
    print("=== Example 5: Direct Builder Usage ===")
    
    # Custom sparse grid path
    custom_path = pathlib.Path("/work/ga45met/mnt/linux_cluster_2/sparse_grid_nodes_weights")
    
    config = SparseGridConfigurationBuilder.create_sparse_grid_config(
        base_uq_method="sc",
        level=3,
        dimension=3,
        base_path=custom_path,
        sc_p_order=2,
        cross_truncation=0.7
    )
    
    print(f"UQ Method: {config.uq_method}")
    print(f"Sparse Grid Level: {config.sparse_grid_level}")
    print(f"Sparse Grid Dimension: {config.sparse_grid_dimension}")
    print(f"Parameters file: {config.parameters_file}")
    print(f"Available UQ methods: {SparseGridConfigurationBuilder.get_available_uq_methods()}")
    print()


def example_6_file_based_configuration():
    """Example 6: Generic file-based configuration."""
    print("=== Example 6: Generic File-based Configuration ===")
    
    # Create a file-based MC configuration with custom nodes file
    custom_nodes_file = "/work/ga45met/mnt/linux_cluster_2/sparse_grid_nodes_weights/GQN_d3_l4.asc"
    
    config = ConfigurationFactory.create_file_based_configuration(
        base_uq_method="mc",
        parameters_file=custom_nodes_file,
        model_type="hbvsask",
        mc_numevaluations=2000,
        sampling_rule="sobol"
    )
    
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"MC Evaluations: {config.mc_numevaluations}")
    print(f"Read from file: {config.read_nodes_from_file}")
    print(f"Parameters file: {config.parameters_file}")
    print()


def example_7_backward_compatibility():
    """Example 7: Backward compatibility with original SparseGridConfiguration."""
    print("=== Example 7: Backward Compatibility ===")
    
    from uqef_dynamic.config.uq_configs import SparseGridConfiguration
    
    # This still works as before
    config = SparseGridConfiguration()
    config.set_sparse_grid_file(
        level=5, 
        dimension=6, 
        base_path="/work/ga45met/mnt/linux_cluster_2/sparse_grid_nodes_weights"
    )
    config.model = "hbvsask"
    config.sc_p_order = 3
    
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"P Order: {config.sc_p_order}")
    print(f"Sparse Quadrature: {config.sc_sparse_quadrature}")
    print(f"Parameters file: {config.parameters_file}")
    print("Note: This class is deprecated. Use the new builder pattern for new code.")
    print()


def example_8_available_methods():
    """Example 8: Show available methods and configurations."""
    print("=== Example 8: Available Methods ===")
    
    print(f"Available models: {ConfigurationFactory.get_available_models()}")
    print(f"Available UQ methods: {ConfigurationFactory.get_available_uq_methods()}")
    print(f"Available sparse grid UQ methods: {ConfigurationFactory.get_available_sparse_grid_uq_methods()}")
    print()


def example_9_validation():
    """Example 9: Configuration validation."""
    print("=== Example 9: Configuration Validation ===")
    
    try:
        # This should work fine
        config = ConfigurationFactory.create_sparse_grid_configuration(
            base_uq_method="mc",
            level=5,
            dimension=6,
            mc_numevaluations=1000
        )
        print("✓ Valid configuration created successfully")
        
        # This should fail validation
        invalid_config = ConfigurationFactory.create_sparse_grid_configuration(
            base_uq_method="mc",
            level=-1,  # Invalid: negative level
            dimension=6,
            mc_numevaluations=1000,
            validate=True
        )
        
    except ValueError as e:
        print(f"✓ Validation correctly caught error: {e}")
    
    print()


def main():
    """Run all sparse grid configuration examples."""
    print("UQEF-Dynamic Sparse Grid Configuration System Examples")
    print("=" * 60)
    print()
    
    examples = [
        example_1_sparse_grid_mc,
        example_2_sparse_grid_mc_regression,
        example_3_sparse_grid_sc,
        example_4_sparse_grid_psp,
        example_5_direct_builder_usage,
        example_6_file_based_configuration,
        example_7_backward_compatibility,
        example_8_available_methods,
        example_9_validation
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
            print()
    
    print("Examples completed!")


if __name__ == "__main__":
    main()
