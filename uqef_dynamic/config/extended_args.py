"""
Extended argument parser for UQEF-Dynamic.

This module extends UQSim's argument parser with additional parameters
for advanced analysis features without modifying the original UQEF code.
"""

import argparse
from typing import Dict, Any


class ExtendedUQSimArgumentParser:
    """
    Extends UQSim's argument parser with additional UQEF-Dynamic specific arguments.
    
    This class adds new command-line arguments to UQSim's existing parser,
    allowing users to specify advanced analysis options through bash scripts
    without modifying the original UQEF UQSim.py file.
    """
    
    def __init__(self, uqsim_instance):
        """
        Initialize the extended argument parser.
        
        Args:
            uqsim_instance: The UQSim instance with its argument parser
        """
        self.uqsim = uqsim_instance
        self.parser = uqsim_instance.parser
        self._add_extended_arguments()
    
    def _add_extended_arguments(self):
        """Add all extended UQEF-Dynamic arguments to UQSim's parser."""
        
        # Advanced analysis options
        self.parser.add_argument(
            '--save_gpce_surrogate', 
            action='store_true', 
            default=True,
            help='Save gPCE surrogate models for each QoI and time step'
        )
        
        self.parser.add_argument(
            '--compute_other_stat_besides_pce_surrogate', 
            action='store_true', 
            default=True,
            help='Compute additional statistics besides PCE surrogate (relevant for SC method)'
        )
        
        self.parser.add_argument(
            '--compute_sobol_indices_with_samples', 
            action='store_true', 
            default=False,
            help='Compute Sobol indices using samples (method-specific, auto-set based on UQ method)'
        )
        
        # KL expansion settings
        self.parser.add_argument(
            '--compute_kl_expansion_of_qoi', 
            action='store_true', 
            default=False,
            help='Enable Karhunen-Loève expansion analysis of QoI'
        )
        
        self.parser.add_argument(
            '--kl_expansion_order', 
            type=int, 
            default=10,
            help='Number of KL modes to compute (default: 10)'
        )
        
        self.parser.add_argument(
            '--compute_timewise_gpce_next_to_kl_expansion', 
            action='store_true', 
            default=False,
            help='Compute time-wise gPCE alongside KL expansion'
        )
        
        # Generalized Sobol indices settings
        self.parser.add_argument(
            '--compute_generalized_sobol_indices', 
            action='store_true', 
            default=False,
            help='Enable computation of generalized Sobol indices'
        )
        
        self.parser.add_argument(
            '--compute_generalized_sobol_indices_over_time', 
            action='store_true', 
            default=False,
            help='Compute generalized Sobol indices over time'
        )
        
        # Covariance matrix settings
        self.parser.add_argument(
            '--compute_covariance_matrix_in_time', 
            action='store_true', 
            default=False,
            help='Compute covariance matrix in time'
        )
        
        # Conditional analysis settings
        self.parser.add_argument(
            '--allow_conditioning_results_based_on_metric', 
            action='store_true', 
            default=False,
            help='Enable conditional analysis based on performance metrics'
        )
        
        self.parser.add_argument(
            '--condition_results_based_on_metric', 
            type=str, 
            default='NSE',
            help='Performance metric for conditional analysis (default: NSE)'
        )
        
        self.parser.add_argument(
            '--condition_results_based_on_metric_value', 
            type=float, 
            default=0.2,
            help='Threshold value for conditional analysis (default: 0.2)'
        )
        
        self.parser.add_argument(
            '--condition_results_based_on_metric_sign', 
            type=str, 
            default='greater_or_equal',
            choices=['greater', 'greater_or_equal', 'less', 'less_or_equal', 'equal'],
            help='Comparison operator for conditional analysis (default: greater_or_equal)'
        )
        
        # Custom statistics computation dictionary arguments
        # Note: These are more complex and will be handled via JSON strings or separate files
        self.parser.add_argument(
            '--dict_stat_to_compute_json', 
            type=str, 
            default=None,
            help='JSON string defining which statistics to compute (e.g., \'{"Var": true, "StdDev": true}\')'
        )
        
        self.parser.add_argument(
            '--dict_what_to_plot_json', 
            type=str, 
            default=None,
            help='JSON string defining what to plot (e.g., \'{"P10": true, "P90": true}\')'
        )
        
        # Alternative: file-based approach for complex dictionaries
        self.parser.add_argument(
            '--extended_config_file', 
            type=str, 
            default=None,
            help='JSON file with extended configuration parameters'
        )
    
    def parse_extended_args(self) -> Dict[str, Any]:
        """
        Parse and extract extended arguments from the command line.
        
        Returns:
            Dict[str, Any]: Dictionary of extended configuration parameters
        """
        # Re-parse arguments to get the extended ones
        args = self.parser.parse_args()
        
        extended_params = {}
        
        # Extract extended arguments
        extended_arg_names = [
            'save_gpce_surrogate',
            'compute_other_stat_besides_pce_surrogate', 
            'compute_sobol_indices_with_samples',
            'compute_kl_expansion_of_qoi',
            'kl_expansion_order',
            'compute_timewise_gpce_next_to_kl_expansion',
            'compute_generalized_sobol_indices',
            'compute_generalized_sobol_indices_over_time',
            'compute_covariance_matrix_in_time',
            'allow_conditioning_results_based_on_metric',
            'condition_results_based_on_metric',
            'condition_results_based_on_metric_value',
            'condition_results_based_on_metric_sign',
            'dict_stat_to_compute_json',
            'dict_what_to_plot_json',
            'extended_config_file'
        ]
        
        for arg_name in extended_arg_names:
            if hasattr(args, arg_name):
                # TODO can I pop out the arguement...
                value = getattr(args, arg_name)
                # Only include non-default values to avoid overriding configuration defaults
                if value is not None:
                    extended_params[arg_name] = value
        
        # Handle JSON string arguments
        if 'dict_stat_to_compute_json' in extended_params:
            extended_params['dict_stat_to_compute'] = self._parse_json_string(
                extended_params.pop('dict_stat_to_compute_json')
            )
        
        if 'dict_what_to_plot_json' in extended_params:
            extended_params['dict_what_to_plot'] = self._parse_json_string(
                extended_params.pop('dict_what_to_plot_json')
            )
        
        # Handle extended config file
        if 'extended_config_file' in extended_params:
            file_params = self._load_extended_config_file(
                extended_params.pop('extended_config_file')
            )
            # File parameters have lower priority than command-line arguments
            for key, value in file_params.items():
                if key not in extended_params:
                    extended_params[key] = value
        
        return extended_params
    
    def _parse_json_string(self, json_string: str) -> Dict[str, Any]:
        """Parse a JSON string into a dictionary."""
        if not json_string:
            return {}
        
        try:
            import json
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON string '{json_string}': {e}")
            return {}
    
    def _load_extended_config_file(self, config_file: str) -> Dict[str, Any]:
        """Load extended configuration parameters from a JSON file."""
        try:
            import json
            import pathlib
            
            config_path = pathlib.Path(config_file)
            if not config_path.exists():
                print(f"Warning: Extended config file not found: {config_file}")
                return {}
            
            with open(config_path, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            print(f"Warning: Failed to load extended config file '{config_file}': {e}")
            return {}
    
    def get_extended_help(self) -> str:
        """
        Get help text for extended arguments.
        
        Returns:
            str: Formatted help text for extended arguments
        """
        help_text = """
Extended UQEF-Dynamic Arguments:

Advanced Analysis Options:
  --save_gpce_surrogate                     Save gPCE surrogate models
  --compute_other_stat_besides_pce_surrogate Compute additional statistics
  --compute_sobol_indices_with_samples      Compute Sobol indices with samples

KL Expansion Analysis:
  --compute_kl_expansion_of_qoi             Enable KL expansion analysis
  --kl_expansion_order N                    Number of KL modes (default: 10)
  --compute_timewise_gpce_next_to_kl_expansion Combine gPCE with KL expansion

Generalized Sobol Indices:
  --compute_generalized_sobol_indices       Enable generalized Sobol indices
  --compute_generalized_sobol_indices_over_time Enable time-dependent analysis

Covariance Analysis:
  --compute_covariance_matrix_in_time       Compute covariance matrix in time

Conditional Analysis:
  --allow_conditioning_results_based_on_metric Enable conditional analysis
  --condition_results_based_on_metric METRIC Performance metric (default: NSE)
  --condition_results_based_on_metric_value VALUE Threshold value (default: 0.2)
  --condition_results_based_on_metric_sign SIGN Comparison operator

Custom Configuration:
  --dict_stat_to_compute_json JSON          Statistics to compute (JSON string)
  --dict_what_to_plot_json JSON             What to plot (JSON string)
  --extended_config_file FILE               Extended config JSON file

Examples:
  # Enable KL expansion with 15 modes
  --compute_kl_expansion_of_qoi --kl_expansion_order 15
  
  # Enable conditional analysis
  --allow_conditioning_results_based_on_metric --condition_results_based_on_metric NSE --condition_results_based_on_metric_value 0.7
  
  # Custom statistics via JSON
  --dict_stat_to_compute_json '{"Var": true, "StdDev": true, "P10": true, "P90": true}'
"""
        return help_text


def extend_uqsim_parser(uqsim_instance):
    """
    Convenience function to extend a UQSim instance's argument parser.
    
    Args:
        uqsim_instance: The UQSim instance to extend
        
    Returns:
        ExtendedUQSimArgumentParser: The extended parser instance
    """
    return ExtendedUQSimArgumentParser(uqsim_instance)
