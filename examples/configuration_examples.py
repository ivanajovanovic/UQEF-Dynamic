"""
Examples demonstrating the UQEF-Dynamic configuration management system.

This script shows various ways to create and use configurations for
uncertainty quantification simulations.
"""

import sys
import pathlib

# Add the UQEF-Dynamic path to sys.path if needed
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from uqef_dynamic.config import ConfigurationFactory, ConfigurationValidator, ValidationError


def example_1_basic_hbv_mc():
    """Example 1: Basic HBV-SASK Monte Carlo configuration."""
    print("=== Example 1: Basic HBV-SASK Monte Carlo ===")
    
    config = ConfigurationFactory.create_hbv_mc_configuration(
        mc_numevaluations=10000,
        sampling_rule="latin_hypercube"
    )
    
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"MC Evaluations: {config.mc_numevaluations}")
    print(f"Sampling Rule: {config.sampling_rule}")
    print(f"Output Directory: {config.outputResultDir}")
    print()


def example_2_hbv_stochastic_collocation():
    """Example 2: HBV-SASK Stochastic Collocation configuration."""
    print("=== Example 2: HBV-SASK Stochastic Collocation ===")
    
    config = ConfigurationFactory.create_hbv_sc_configuration(
        sc_q_order=7,
        sc_p_order=3,
        sc_quadrature_rule="g"
    )
    
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"Q Order: {config.sc_q_order}")
    print(f"P Order: {config.sc_p_order}")
    print(f"Quadrature Rule: {config.sc_quadrature_rule}")
    print()


def example_3_sparse_grid():
    """Example 3: HBV-SASK Sparse Grid configuration."""
    print("=== Example 3: HBV-SASK Sparse Grid ===")
    
    config = ConfigurationFactory.create_hbv_sparse_grid_configuration(
        level=5,
        dimension=6,
        cross_truncation=0.8
    )
    
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"Cross Truncation: {config.cross_truncation}")
    print(f"Sparse Grid File: {config.parameters_file}")
    print()


def example_4_battery_configuration():
    """Example 4: Battery model configuration."""
    print("=== Example 4: Battery Model Configuration ===")
    
    config = ConfigurationFactory.create_battery_mc_configuration(
        mc_numevaluations=5000,
        sampling_rule="sobol"
    )
    
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"MC Evaluations: {config.mc_numevaluations}")
    print(f"Input Directory: {config.inputModelDir}")
    print()


def example_5_custom_configuration():
    """Example 5: Custom configuration using generic factory method."""
    print("=== Example 5: Custom Configuration ===")
    
    config = ConfigurationFactory.create_configuration(
        model_type="ishigami",
        uq_method="sc",
        sc_q_order=10,
        sc_p_order=5,
        cross_truncation=0.7,
        mpi=True,
        num_cores=4,
        compute_Sobol_m=True,
        compute_Sobol_t=True
    )
    
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"MPI Enabled: {config.mpi}")
    print(f"Number of Cores: {config.num_cores}")
    print(f"Compute Sobol Main: {config.compute_Sobol_m}")
    print()


def example_6_json_template():
    """Example 6: Loading configuration from JSON template."""
    print("=== Example 6: JSON Template Configuration ===")
    
    try:
        # This would work if the JSON file exists
        template_path = pathlib.Path(__file__).parent.parent / "uqef_dynamic/config/templates/hbv_mc.json"
        
        if template_path.exists():
            config = ConfigurationFactory.from_json_file(template_path)
            print(f"Loaded from JSON: {template_path}")
            print(f"Model: {config.model}")
            print(f"UQ Method: {config.uq_method}")
            print(f"Description: {getattr(config, 'description', 'N/A')}")
        else:
            print(f"Template file not found: {template_path}")
            
    except Exception as e:
        print(f"Error loading JSON template: {e}")
    
    print()


def example_7_configuration_validation():
    """Example 7: Configuration validation."""
    print("=== Example 7: Configuration Validation ===")
    
    # Create a valid configuration
    config = ConfigurationFactory.create_hbv_mc_configuration()
    
    validator = ConfigurationValidator()
    
    try:
        validator.validate(config)
        print("✓ Configuration is valid!")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
    
    # Create an invalid configuration to demonstrate validation
    try:
        invalid_config = ConfigurationFactory.create_configuration(
            model_type="hbvsask",
            uq_method="mc",
            mc_numevaluations=-1000,  # Invalid: negative number
            cross_truncation=1.5      # Invalid: > 1.0
        )
        validator.validate(invalid_config)
    except ValidationError as e:
        print(f"✓ Validation correctly caught errors: {e}")
    
    print()


def example_8_configuration_modification():
    """Example 8: Modifying existing configurations."""
    print("=== Example 8: Configuration Modification ===")
    
    # Start with a base configuration
    base_config = ConfigurationFactory.create_hbv_mc_configuration()
    print(f"Original MC evaluations: {base_config.mc_numevaluations}")
    
    # Modify specific parameters
    base_config.update(
        mc_numevaluations=50000,
        sampling_rule="sobol",
        compute_Sobol_m=True,
        compute_Sobol_t=True,
        mpi=True,
        num_cores=8
    )
    
    print(f"Modified MC evaluations: {base_config.mc_numevaluations}")
    print(f"Modified sampling rule: {base_config.sampling_rule}")
    print(f"Modified number of cores: {base_config.num_cores}")
    print()


def example_9_batch_configurations():
    """Example 9: Creating multiple configurations for parameter studies."""
    print("=== Example 9: Batch Configurations ===")
    
    # Create configurations for different sample sizes
    sample_sizes = [1000, 5000, 10000, 25000]
    configs = []
    
    for n_samples in sample_sizes:
        config = ConfigurationFactory.create_hbv_mc_configuration(
            mc_numevaluations=n_samples,
            run_type=f"mc_{n_samples}_samples",
            sampling_rule="latin_hypercube"
        )
        configs.append(config)
        print(f"Created config for {n_samples} samples")
    
    print(f"Total configurations created: {len(configs)}")
    print()


def example_10_available_options():
    """Example 10: Show available models and UQ methods."""
    print("=== Example 10: Available Options ===")
    
    models = ConfigurationFactory.get_available_models()
    uq_methods = ConfigurationFactory.get_available_uq_methods()
    
    print(f"Available models: {models}")
    print(f"Available UQ methods: {uq_methods}")
    print()


def main():
    """Run all configuration examples."""
    print("UQEF-Dynamic Configuration System Examples")
    print("=" * 50)
    print()
    
    examples = [
        example_1_basic_hbv_mc,
        example_2_hbv_stochastic_collocation,
        example_3_sparse_grid,
        example_4_battery_configuration,
        example_5_custom_configuration,
        example_6_json_template,
        example_7_configuration_validation,
        example_8_configuration_modification,
        example_9_batch_configurations,
        example_10_available_options
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
