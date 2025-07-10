#!/usr/bin/env python3
"""
Test script to demonstrate the configuration management system integration.

This script shows how the system works in both modes:
1. Local debugging mode (local_debugging=True)
2. Cluster execution mode (local_debugging=False) with command-line arguments

Usage:
    # Test local debugging mode
    python test_configuration_integration.py --test-mode local

    # Test cluster execution mode (simulates bash script arguments)
    python test_configuration_integration.py --test-mode cluster --model hbvsask --uq_method sc --sc_q_order 7 --sc_p_order 5 --parameters_file /dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights/KPU_d10_l7.asc
"""

import sys
import os
import argparse
from unittest.mock import Mock

# Add the UQEF-Dynamic directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uqef_dynamic.config import ConfigurationFactory


def create_mock_uqsim_args(**kwargs):
    """Create a mock UQsim args object with the specified parameters."""
    
    # Default arguments that would typically be parsed by UQEF
    defaults = {
        'model': 'hbvsask',
        'uq_method': 'sc',
        'inputModelDir': '/dss/dsshome1/lxc0C/ga45met2/Repositories/HBV-SASK-data',
        'outputResultDir': '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/test_output',
        'sourceDir': '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic',
        'config_file': '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/data/configurations/configuration_hbv_10D.json',
        'sc_q_order': 7,
        'sc_p_order': 5,
        'sc_quadrature_rule': 'p',
        'sc_poly_rule': 'three_terms_recurrence',
        'cross_truncation': 0.7,
        'regression': False,
        'sampleFromStandardDist': True,
        'read_nodes_from_file': True,
        'parameters_file': '/dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights/KPU_d10_l7.asc',
        'mpi': True,
        'mpi_method': 'MpiPoolSolver',
        'num_cores': 112,
        'parallel_statistics': True,
        'compute_Sobol_t': True,
        'compute_Sobol_m': True,
        'save_all_simulations': True,
        'store_gpce_surrogate_in_stat_dict': True,
        'uncertain': 'all',
        'sampling_rule': 'random',
        'sc_poly_normed': True
    }
    
    # Update with any provided overrides
    defaults.update(kwargs)
    
    # Create mock object
    mock_args = Mock()
    for key, value in defaults.items():
        setattr(mock_args, key, value)
    
    return mock_args


def test_local_debugging_mode():
    """Test the local debugging configuration mode."""
    print("=" * 60)
    print("TESTING LOCAL DEBUGGING MODE")
    print("=" * 60)
    
    # This simulates what happens when local_debugging=True
    print("Creating HBV-SASK Monte Carlo configuration...")
    config = ConfigurationFactory.create_hbv_mc_configuration(
        mc_numevaluations=10000,
        sampling_rule="latin_hypercube"
    )
    
    print(f"Configuration created: {type(config).__name__}")
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"MC Evaluations: {config.mc_numevaluations}")
    print(f"Sampling Rule: {config.sampling_rule}")
    print(f"Output Directory: {config.outputResultDir}")
    print(f"Run Type: {config.run_type}")
    
    print("\nCreating HBV-SASK Stochastic Collocation configuration...")
    config_sc = ConfigurationFactory.create_hbv_sc_configuration(
        sc_q_order=5,
        sc_p_order=2
    )
    
    print(f"Configuration created: {type(config_sc).__name__}")
    print(f"Model: {config_sc.model}")
    print(f"UQ Method: {config_sc.uq_method}")
    print(f"SC Q Order: {config_sc.sc_q_order}")
    print(f"SC P Order: {config_sc.sc_p_order}")
    print(f"Output Directory: {config_sc.outputResultDir}")
    print(f"Run Type: {config_sc.run_type}")


def test_cluster_execution_mode():
    """Test the cluster execution configuration mode."""
    print("=" * 60)
    print("TESTING CLUSTER EXECUTION MODE")
    print("=" * 60)
    
    # Test 1: Standard SC configuration
    print("Test 1: Standard SC Configuration")
    print("-" * 40)
    
    mock_args = create_mock_uqsim_args(
        model='hbvsask',
        uq_method='sc',
        sc_q_order=5,
        sc_p_order=2,
        regression=True
    )
    
    config = ConfigurationFactory.from_uqsim_args(mock_args)
    
    print(f"Configuration created: {type(config).__name__}")
    print(f"Model: {config.model}")
    print(f"UQ Method: {config.uq_method}")
    print(f"SC Q Order: {config.sc_q_order}")
    print(f"SC P Order: {config.sc_p_order}")
    print(f"Run Type: {config.run_type}")
    
    # Test 2: PSP configuration (SC with regression)
    print("\nTest 2: PSP Configuration (SC with regression=False)")
    print("-" * 40)
    
    mock_args_psp = create_mock_uqsim_args(
        model='hbvsask',
        uq_method='sc',
        sc_q_order=5,
        sc_p_order=2,
        regression=False
    )
    
    config_psp = ConfigurationFactory.from_uqsim_args(mock_args_psp)
    
    print(f"Configuration created: {type(config_psp).__name__}")
    print(f"Model: {config_psp.model}")
    print(f"UQ Method: {config_psp.uq_method}")
    print(f"Regression: {config_psp.regression}")
    print(f"Run Type: {config_psp.run_type}")
    
    # Test 3: MC Regression configuration
    print("\nTest 3: MC Regression Configuration")
    print("-" * 40)
    
    mock_args_mc_reg = create_mock_uqsim_args(
        model='hbvsask',
        uq_method='mc',
        mc_numevaluations=10000,
        sampling_rule='latin_hypercube',
        regression=True
    )
    
    config_mc_reg = ConfigurationFactory.from_uqsim_args(mock_args_mc_reg)
    
    print(f"Configuration created: {type(config_mc_reg).__name__}")
    print(f"Model: {config_mc_reg.model}")
    print(f"UQ Method: {config_mc_reg.uq_method}")
    print(f"MC Evaluations: {config_mc_reg.mc_numevaluations}")
    print(f"Regression: {config_mc_reg.regression}")
    print(f"Run Type: {config_mc_reg.run_type}")
    
    # Test 4: Sparse Grid configuration (from bash script example)
    print("\nTest 4: Sparse Grid Configuration (from bash script)")
    print("-" * 40)
    
    mock_args_sparse = create_mock_uqsim_args(
        model='hbvsask',
        uq_method='sc',
        sc_q_order=7,
        sc_p_order=5,
        cross_truncation=0.7,
        parameters_file='/dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights/KPU_d10_l7.asc',
        read_nodes_from_file=True,
        regression=False
    )
    
    config_sparse = ConfigurationFactory.from_uqsim_args(mock_args_sparse)
    
    print(f"Configuration created: {type(config_sparse).__name__}")
    print(f"Model: {config_sparse.model}")
    print(f"UQ Method: {config_sparse.uq_method}")
    print(f"Parameters File: {config_sparse.parameters_file}")
    print(f"Read Nodes from File: {config_sparse.read_nodes_from_file}")
    print(f"Run Type: {config_sparse.run_type}")


def main():
    parser = argparse.ArgumentParser(description='Test configuration integration')
    parser.add_argument('--test-mode', choices=['local', 'cluster', 'both'], 
                       default='both', help='Which test mode to run')
    
    args = parser.parse_args()
    
    try:
        if args.test_mode in ['local', 'both']:
            test_local_debugging_mode()
        
        if args.test_mode in ['cluster', 'both']:
            if args.test_mode == 'both':
                print("\n\n")
            test_cluster_execution_mode()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
