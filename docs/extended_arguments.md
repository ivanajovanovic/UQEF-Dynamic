# Extended Command-Line Arguments for UQEF-Dynamic

This document describes the extended command-line argument system that allows users to access advanced analysis features through bash scripts without modifying the original UQEF code.

## Overview

The extended argument parser adds new command-line arguments to UQEF-Dynamic that enable:

- **KL Expansion Analysis**: Karhunen-Loève expansion of quantities of interest
- **Generalized Sobol Indices**: Advanced sensitivity analysis
- **Covariance Matrix Analysis**: Time-dependent covariance computation
- **Conditional Analysis**: Results filtering based on performance metrics
- **Custom Statistics and Plotting**: Configurable computation and visualization options

## Key Features

### 1. **No UQEF Modification Required**
The extended arguments are added to UQSim's parser without modifying the original UQEF code, ensuring compatibility and maintainability.

### 2. **Full Backward Compatibility**
All existing bash scripts continue to work unchanged. Extended arguments are optional with sensible defaults.

### 3. **Command-Line Interface**
All new features can be accessed through standard command-line arguments, maintaining the familiar workflow.

### 4. **Method-Specific Intelligence**
The system automatically adjusts parameters based on the UQ method (e.g., Saltelli vs MC Sobol indices).

## Available Extended Arguments

### Advanced Analysis Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--save_gpce_surrogate` | flag | True | Save gPCE surrogate models for each QoI and time step |
| `--compute_other_stat_besides_pce_surrogate` | flag | True | Compute additional statistics besides PCE surrogate |
| `--compute_sobol_indices_with_samples` | flag | False | Compute Sobol indices using samples (auto-set based on method) |

### KL Expansion Analysis

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--compute_kl_expansion_of_qoi` | flag | False | Enable Karhunen-Loève expansion analysis |
| `--kl_expansion_order` | int | 10 | Number of KL modes to compute |
| `--compute_timewise_gpce_next_to_kl_expansion` | flag | False | Compute time-wise gPCE alongside KL expansion |

### Generalized Sobol Indices

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--compute_generalized_sobol_indices` | flag | False | Enable computation of generalized Sobol indices |
| `--compute_generalized_sobol_indices_over_time` | flag | False | Compute generalized Sobol indices over time |

### Covariance Analysis

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--compute_covariance_matrix_in_time` | flag | False | Compute covariance matrix in time |

### Conditional Analysis

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--allow_conditioning_results_based_on_metric` | flag | False | Enable conditional analysis based on performance metrics |
| `--condition_results_based_on_metric` | string | "NSE" | Performance metric for conditional analysis |
| `--condition_results_based_on_metric_value` | float | 0.2 | Threshold value for conditional analysis |
| `--condition_results_based_on_metric_sign` | choice | "greater_or_equal" | Comparison operator (greater, greater_or_equal, less, less_or_equal, equal) |

### Custom Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dict_stat_to_compute_json` | string | None | JSON string defining which statistics to compute |
| `--dict_what_to_plot_json` | string | None | JSON string defining what to plot |
| `--extended_config_file` | string | None | JSON file with extended configuration parameters |

## Usage Examples

### Basic KL Expansion

```bash
python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --model hbvsask \
    --uq_method sc \
    --sc_p_order 3 \
    --sc_q_order 7 \
    --config_file data/configurations/configuration_hbv_10D.json \
    --compute_kl_expansion_of_qoi \
    --kl_expansion_order 15
```

### Comprehensive Advanced Analysis

```bash
python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --model hbvsask \
    --uq_method sc \
    --sc_p_order 3 \
    --sc_q_order 7 \
    --config_file data/configurations/configuration_hbv_10D.json \
    --mpi \
    --parallel_statistics \
    --compute_kl_expansion_of_qoi \
    --kl_expansion_order 12 \
    --compute_timewise_gpce_next_to_kl_expansion \
    --compute_generalized_sobol_indices \
    --compute_generalized_sobol_indices_over_time \
    --compute_covariance_matrix_in_time \
    --allow_conditioning_results_based_on_metric \
    --condition_results_based_on_metric NSE \
    --condition_results_based_on_metric_value 0.7
```

### Custom Statistics via JSON

```bash
python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --model hbvsask \
    --uq_method mc \
    --mc_numevaluations 10000 \
    --config_file data/configurations/configuration_hbv_10D.json \
    --dict_stat_to_compute_json '{"Var": true, "StdDev": true, "P10": true, "P90": true, "Skew": true}' \
    --dict_what_to_plot_json '{"P10": true, "P90": true, "StdDev": true, "Sobol_m": true}'
```

### Using Extended Config File

First, create an extended configuration file:

```json
{
    "compute_kl_expansion_of_qoi": true,
    "kl_expansion_order": 20,
    "compute_generalized_sobol_indices": true,
    "compute_covariance_matrix_in_time": true,
    "allow_conditioning_results_based_on_metric": true,
    "condition_results_based_on_metric": "NSE",
    "condition_results_based_on_metric_value": 0.8,
    "dict_stat_to_compute": {
        "Var": true,
        "StdDev": true,
        "P10": true,
        "P90": true,
        "Sobol_m": true,
        "Sobol_t": true
    }
}
```

Then use it in your script:

```bash
python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --model hbvsask \
    --uq_method sc \
    --config_file data/configurations/configuration_hbv_10D.json \
    --extended_config_file /path/to/extended_config.json
```

## Method-Specific Behavior

### Monte Carlo (MC)
- When `--compute_Sobol_m` is used, `compute_sobol_indices_with_samples` is automatically set to `True`
- Supports all extended analysis features

### Saltelli
- `compute_sobol_indices_with_samples` is automatically set to `False`
- Sobol indices are computed using the Saltelli method, not samples
- Supports generalized Sobol indices and conditional analysis

### Stochastic Collocation (SC)
- Supports all extended analysis features
- KL expansion works particularly well with SC methods
- gPCE surrogates are available for additional analysis

## JSON Configuration Format

### Statistics Dictionary
```json
{
    "Var": true,
    "StdDev": true,
    "P10": true,
    "P90": true,
    "E_minus_std": false,
    "E_plus_std": false,
    "Skew": false,
    "Kurt": false,
    "Sobol_m": true,
    "Sobol_m2": false,
    "Sobol_t": true
}
```

### Plotting Dictionary
```json
{
    "E_minus_std": false,
    "E_plus_std": false,
    "E_minus_2std": true,
    "E_plus_2std": true,
    "P10": true,
    "P90": true,
    "StdDev": true,
    "Skew": false,
    "Kurt": false,
    "Sobol_m": true,
    "Sobol_m2": false,
    "Sobol_t": true,
    "generalized_sobol_total_index": true,
    "generalized_sobol_main_index": true
}
```

## Migration from Hardcoded Values

### Before (Hardcoded)
Previously, these parameters were hardcoded in the simulation script:

```python
compute_kl_expansion_of_qoi = False
kl_expansion_order = 10
compute_generalized_sobol_indices = False
# ... many more hardcoded values
```

### After (Command-Line Configurable)
Now these can be set via command-line arguments:

```bash
--compute_kl_expansion_of_qoi \
--kl_expansion_order 15 \
--compute_generalized_sobol_indices
```

## Integration with Existing Workflows

### HPC/Cluster Scripts
The extended arguments integrate seamlessly with existing SLURM/PBS scripts:

```bash
#!/bin/bash
#SBATCH -J advanced_uq
#SBATCH --nodes=4
#SBATCH --time=2:00:00

module load python/3.11
source activate uqef_env

mpiexec -n $SLURM_NTASKS python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
    --model hbvsask \
    --uq_method sc \
    --config_file data/configurations/configuration_hbv_10D.json \
    --mpi \
    --parallel_statistics \
    --compute_kl_expansion_of_qoi \
    --kl_expansion_order 15 \
    --compute_generalized_sobol_indices \
    --compute_covariance_matrix_in_time
```

### Batch Parameter Studies
Extended arguments work well with parameter studies:

```bash
for kl_order in 5 10 15 20; do
    python uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
        --model hbvsask \
        --uq_method sc \
        --config_file data/configurations/configuration_hbv_10D.json \
        --outputResultDir results/kl_order_${kl_order} \
        --compute_kl_expansion_of_qoi \
        --kl_expansion_order ${kl_order}
done
```

## Troubleshooting

### Common Issues

1. **Unknown argument error**: Make sure you're using the updated `uq_simulation_uqsim.py` script that includes the extended argument parser.

2. **JSON parsing errors**: Ensure JSON strings are properly quoted and escaped in bash scripts:
   ```bash
   --dict_stat_to_compute_json '{"Var": true, "StdDev": true}'
   ```

3. **Configuration conflicts**: Extended arguments override configuration file values. Command-line arguments have the highest priority.

### Getting Help

To see all available extended arguments:

```python
from uqef_dynamic.config import ExtendedUQSimArgumentParser
import uqef

uqsim = uqef.UQsim()
parser = ExtendedUQSimArgumentParser(uqsim)
print(parser.get_extended_help())
```

## Performance Considerations

- **KL Expansion**: Higher orders increase computational cost
- **Generalized Sobol Indices**: More expensive than standard Sobol indices
- **Covariance Matrix**: Memory-intensive for large time series
- **Conditional Analysis**: May reduce effective sample size

## Future Extensions

The extended argument system is designed to be easily extensible. New analysis features can be added by:

1. Adding new arguments to `ExtendedUQSimArgumentParser`
2. Adding corresponding parameters to the configuration classes
3. Updating the simulation scripts to use the new parameters

This ensures that UQEF-Dynamic can continue to evolve while maintaining compatibility with existing workflows.
