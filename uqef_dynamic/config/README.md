# UQEF-Dynamic Configuration Management System

This document describes the new configuration management system for UQEF-Dynamic, which provides a structured and elegant way to manage input configuration arguments for uncertainty quantification simulations.

## Overview

The configuration system replaces the previous hardcoded parameter blocks with a clean, object-oriented approach that includes:

- **Type-safe configuration classes** for different UQ methods and models
- **Automatic validation** of parameters and dependencies
- **Reusable configuration templates** for common scenarios
- **Factory pattern** for easy configuration creation
- **JSON support** for external configuration files

## Quick Start

### Basic Usage

```python
from uqef_dynamic.config import ConfigurationFactory

# Create HBV-SASK Monte Carlo configuration
config = ConfigurationFactory.create_hbv_mc_configuration(
    mc_numevaluations=10000,
    sampling_rule="latin_hypercube"
)

# Apply to UQsim instance
config.apply_to_uqsim(uqsim)
```

### Extended Analysis Features (New!)

```python
# Create configuration with advanced analysis features
config = ConfigurationFactory.create_configuration(
    model_type="hbvsask",
    uq_method="sc",
    sc_p_order=3,
    sc_q_order=7,
    # KL expansion analysis
    compute_kl_expansion_of_qoi=True,
    kl_expansion_order=12,
    # Generalized Sobol indices
    compute_generalized_sobol_indices=True,
    compute_generalized_sobol_indices_over_time=True,
    # Covariance matrix computation
    compute_covariance_matrix_in_time=True,
    # Conditional analysis
    allow_conditioning_results_based_on_metric=True,
    condition_results_based_on_metric="NSE",
    condition_results_based_on_metric_value=0.7
)
```

### Configuration Overrides for Batch Scripts

```python
# For compatibility with existing batch scripts
config = ConfigurationFactory.from_uqsim_args(uqsim.args)

# Apply advanced features
ConfigurationFactory.apply_configuration_overrides(
    config,
    compute_kl_expansion_of_qoi=True,
    compute_generalized_sobol_indices=True,
    dict_what_to_plot={
        "Sobol_m": True, "Sobol_t": True,
        "generalized_sobol_total_index": True,
        "P10": True, "P90": True
    }
)
```

### Extended Command-Line Arguments (New!)

All advanced features can now be accessed directly through command-line arguments:

```bash
python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --model "hbvsask" \
    --uq_method "sc" \
    --sc_p_order 3 \
    --sc_q_order 7 \
    --config_file data/configurations/configuration_hbv_10D.json \
    --compute_kl_expansion_of_qoi \
    --kl_expansion_order 10 \
    --compute_generalized_sobol_indices \
    --compute_covariance_matrix_in_time \
    --allow_conditioning_results_based_on_metric \
    --condition_results_based_on_metric NSE \
    --condition_results_based_on_metric_value 0.7
```

For a complete list of extended arguments, see `docs/extended_arguments.md`.

### Available Factory Methods

```python
# HBV-SASK configurations
config = ConfigurationFactory.create_hbv_mc_configuration()
config = ConfigurationFactory.create_hbv_sc_configuration()
config = ConfigurationFactory.create_hbv_sparse_grid_configuration(level=5, dimension=6)

# Battery configurations
config = ConfigurationFactory.create_battery_mc_configuration()
config = ConfigurationFactory.create_battery_sc_configuration()

# Ishigami test function
config = ConfigurationFactory.create_ishigami_configuration('sc')

# Generic configuration
config = ConfigurationFactory.create_configuration(
    model_type="hbvsask",
    uq_method="mc",
    mc_numevaluations=5000
)
```

### Using JSON Templates

```python
# Load from JSON template
config = ConfigurationFactory.from_json_file(
    'uqef_dynamic/config/templates/hbv_mc.json'
)
```

## Architecture

### Class Hierarchy

```
UQConfiguration (Abstract Base)
├── MCConfiguration
├── SCConfiguration  
├── SaltelliConfiguration
├── EnsembleConfiguration
└── SparseGridConfiguration

ModelConfiguration (Base)
├── HBVSASKConfig
├── BatteryConfig
├── LarsimConfig
├── IshigamiConfig
└── SimpleOscillatorConfig
```

### Core Components

1. **Base Configuration (`UQConfiguration`)**: Abstract base class with common parameters
2. **UQ Method Configurations**: Specialized classes for different UQ methods (MC, SC, etc.)
3. **Model Configurations**: Model-specific path and parameter management
4. **Configuration Factory**: Creates and combines configurations
5. **Configuration Validator**: Validates parameters and dependencies

## Configuration Classes

### UQ Method Configurations

#### MCConfiguration
```python
config = MCConfiguration()
config.mc_numevaluations = 10000
config.sampling_rule = "latin_hypercube"
```

#### SCConfiguration
```python
config = SCConfiguration()
config.sc_q_order = 5
config.sc_p_order = 2
config.sc_quadrature_rule = "g"
```

#### SparseGridConfiguration
```python
config = SparseGridConfiguration()
config.set_sparse_grid_file(level=5, dimension=6, base_path="/path/to/grids")
```

## Extended Configuration Parameters

The configuration system now supports advanced analysis features that were previously hardcoded:

### KL Expansion Analysis
```python
config.compute_kl_expansion_of_qoi = True          # Enable KL expansion
config.kl_expansion_order = 12                     # Number of KL modes
config.compute_timewise_gpce_next_to_kl_expansion = True  # Combine with gPCE
```

### Generalized Sobol Indices
```python
config.compute_generalized_sobol_indices = True           # Enable generalized Sobol
config.compute_generalized_sobol_indices_over_time = True # Time-dependent analysis
```

### Covariance Matrix Analysis
```python
config.compute_covariance_matrix_in_time = True    # Enable covariance computation
```

### Conditional Analysis
```python
config.allow_conditioning_results_based_on_metric = True
config.condition_results_based_on_metric = "NSE"           # Metric name
config.condition_results_based_on_metric_value = 0.7       # Threshold value
config.condition_results_based_on_metric_sign = "greater_or_equal"  # Condition
```

### Advanced Analysis Options
```python
config.save_gpce_surrogate = True                          # Save gPCE surrogates
config.compute_other_stat_besides_pce_surrogate = True     # Additional statistics
config.compute_sobol_indices_with_samples = False          # Method-specific setting
```

### Custom Statistics and Plotting
```python
# Custom statistics computation
config.dict_stat_to_compute = {
    "Var": True, "StdDev": True, "P10": True, "P90": True,
    "E_minus_std": True, "E_plus_std": True,
    "Skew": True, "Kurt": True, 
    "Sobol_m": True, "Sobol_t": True
}

# Custom plotting configuration
config.dict_what_to_plot = {
    "E_minus_std": True, "E_plus_std": True, 
    "E_minus_2std": True, "E_plus_2std": True,
    "P10": True, "P90": True, "StdDev": True,
    "Sobol_m": True, "Sobol_t": True,
    "generalized_sobol_total_index": True, 
    "generalized_sobol_main_index": True
}
```

### Parameter Defaults

All extended parameters have sensible defaults for backward compatibility:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compute_kl_expansion_of_qoi` | `False` | Enable KL expansion analysis |
| `kl_expansion_order` | `10` | Number of KL modes |
| `compute_generalized_sobol_indices` | `False` | Enable generalized Sobol indices |
| `compute_covariance_matrix_in_time` | `False` | Enable covariance matrix computation |
| `allow_conditioning_results_based_on_metric` | `False` | Enable conditional analysis |
| `condition_results_based_on_metric` | `"NSE"` | Performance metric for conditioning |
| `condition_results_based_on_metric_value` | `0.2` | Threshold value |
| `save_gpce_surrogate` | `True` | Save gPCE surrogate models |
| `compute_sobol_indices_with_samples` | `False` | Method-dependent (auto-set) |

### Model Configurations

#### HBVSASKConfig
```python
model_config = HBVSASKConfig()
config_file = model_config.get_config_file_path("10D")
output_dir = model_config.get_output_dir("mc_short")
```

## Validation

The system includes comprehensive validation:

```python
from uqef_dynamic.config import ConfigurationValidator

validator = ConfigurationValidator()
try:
    validator.validate(config)
    print("Configuration is valid!")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Validation Features

- **Parameter range checking**: Ensures numeric parameters are within valid ranges
- **Dependency validation**: Checks parameter dependencies (e.g., regression settings)
- **Path validation**: Verifies file and directory paths exist
- **UQ method compatibility**: Ensures parameters match the selected UQ method
- **Model-specific validation**: Checks model-specific requirements

## JSON Templates

Pre-configured JSON templates are available in `uqef_dynamic/config/templates/`:

- `hbv_mc.json`: HBV-SASK Monte Carlo
- `hbv_sc.json`: HBV-SASK Stochastic Collocation  
- `battery_mc.json`: Battery Monte Carlo

### Template Structure

```json
{
    "model_type": "hbvsask",
    "uq_method": "mc",
    "mc_numevaluations": 10000,
    "sampling_rule": "latin_hypercube",
    "mpi": true,
    "compute_Sobol_m": true,
    "description": "Standard HBV-SASK Monte Carlo configuration"
}
```

## Advanced Usage

### Custom Configuration

```python
# Create custom configuration
config = ConfigurationFactory.create_configuration(
    model_type="hbvsask",
    uq_method="sc",
    sc_q_order=7,
    sc_p_order=3,
    outputResultDir="/custom/output/path",
    custom_name="my_experiment"
)
```

### Configuration Inheritance

```python
# Start with a base configuration
base_config = ConfigurationFactory.create_hbv_mc_configuration()

# Modify specific parameters
base_config.update(
    mc_numevaluations=50000,
    sampling_rule="sobol",
    compute_Sobol_m=True
)
```

### Batch Configuration

```python
# Create multiple configurations for parameter studies
configs = []
for n_evals in [1000, 5000, 10000]:
    config = ConfigurationFactory.create_hbv_mc_configuration(
        mc_numevaluations=n_evals,
        run_type=f"mc_{n_evals}"
    )
    configs.append(config)
```

## Migration Guide

### Before (Old System)
```python
# Old hardcoded approach
uqsim.args.model = "hbvsask"
uqsim.args.uq_method = "mc"
uqsim.args.mc_numevaluations = 10000
uqsim.args.sampling_rule = "latin_hypercube"
uqsim.args.inputModelDir = pathlib.Path("/path/to/input")
uqsim.args.outputResultDir = "/path/to/output"
# ... 50+ more lines of configuration
```

### After (New System)
```python
# New structured approach
config = ConfigurationFactory.create_hbv_mc_configuration(
    mc_numevaluations=10000,
    sampling_rule="latin_hypercube"
)
config.apply_to_uqsim(uqsim)
```

## Benefits

1. **Maintainability**: Centralized configuration management
2. **Reusability**: Easy to create and share configurations
3. **Validation**: Built-in parameter validation and consistency checks
4. **Flexibility**: Easy to override specific parameters
5. **Documentation**: Self-documenting configuration structure
6. **Type Safety**: Proper typing and IDE support
7. **Testing**: Easier to test different configuration scenarios

## Best Practices

1. **Use factory methods** for common scenarios
2. **Validate configurations** before use
3. **Use JSON templates** for reproducible experiments
4. **Document custom configurations** with descriptions
5. **Version control** configuration files
6. **Test configurations** in isolation before running simulations

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure the config module is in your Python path
2. **Validation Error**: Check parameter ranges and dependencies
3. **Path Not Found**: Verify input/output directory paths exist
4. **Model Not Available**: Check if required model dependencies are installed

### Debug Mode

```python
# Enable detailed validation output
validator = ConfigurationValidator()
summary = validator.get_validation_summary()
print(f"Errors: {summary['errors']}")
print(f"Warnings: {summary['warnings']}")
```

## Examples

See the `examples/` directory for complete working examples of different configuration scenarios.
