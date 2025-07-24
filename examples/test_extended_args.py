#!/usr/bin/env python3
"""
Test script to verify the extended argument parser functionality.

This script tests the extended argument parser without running a full simulation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uqef
from uqef_dynamic.config import ExtendedUQSimArgumentParser, ConfigurationFactory

def test_extended_args_parsing():
    """Test that extended arguments are parsed correctly."""
    print("=== Testing Extended Argument Parser ===")
    
    # Create a UQSim instance
    uqsim = uqef.UQsim()
    
    # Extend the argument parser
    extended_parser = ExtendedUQSimArgumentParser(uqsim)
    
    # Test parsing with some extended arguments
    # We'll simulate command line arguments
    test_args = [
        'test_script.py',  # script name
        '--model', 'hbvsask',
        '--uq_method', 'sc',
        '--sc_p_order', '3',
        '--sc_q_order', '7',
        '--compute_kl_expansion_of_qoi',
        '--kl_expansion_order', '15',
        '--compute_generalized_sobol_indices',
        '--allow_conditioning_results_based_on_metric',
        '--condition_results_based_on_metric', 'NSE',
        '--condition_results_based_on_metric_value', '0.7',
        '--dict_stat_to_compute_json', '{"Var": true, "StdDev": true, "P10": true}',
        '--outputResultDir', '/tmp/test_output'
    ]
    
    # Temporarily replace sys.argv
    original_argv = sys.argv
    sys.argv = test_args
    
    try:
        # Parse extended arguments
        extended_args = extended_parser.parse_extended_args()
        
        print("✓ Extended arguments parsed successfully!")
        print(f"Extended arguments found: {list(extended_args.keys())}")
        
        # Verify specific arguments
        assert extended_args.get('kl_expansion_order') == 15, "KL expansion order not parsed correctly"
        assert extended_args.get('compute_kl_expansion_of_qoi') == True, "KL expansion flag not parsed correctly"
        assert extended_args.get('compute_generalized_sobol_indices') == True, "Generalized Sobol flag not parsed correctly"
        assert extended_args.get('condition_results_based_on_metric') == 'NSE', "Condition metric not parsed correctly"
        assert extended_args.get('condition_results_based_on_metric_value') == 0.7, "Condition value not parsed correctly"
        
        # Test JSON parsing
        expected_stats = {"Var": True, "StdDev": True, "P10": True}
        assert extended_args.get('dict_stat_to_compute') == expected_stats, "JSON statistics not parsed correctly"
        
        print("✓ All extended argument values verified!")
        
        # Test configuration creation
        config = ConfigurationFactory.from_uqsim_args(uqsim.args)
        ConfigurationFactory.apply_configuration_overrides(config, **extended_args)
        
        print("✓ Configuration created and extended arguments applied!")
        print(f"Final config KL expansion: {config.compute_kl_expansion_of_qoi}")
        print(f"Final config KL order: {config.kl_expansion_order}")
        print(f"Final config generalized Sobol: {config.compute_generalized_sobol_indices}")
        print(f"Final config conditional analysis: {config.allow_conditioning_results_based_on_metric}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
        
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def test_help_functionality():
    """Test the help functionality."""
    print("\n=== Testing Help Functionality ===")
    
    try:
        uqsim = uqef.UQsim()
        extended_parser = ExtendedUQSimArgumentParser(uqsim)
        help_text = extended_parser.get_extended_help()
        
        assert "Extended UQEF-Dynamic Arguments:" in help_text, "Help text missing header"
        assert "--compute_kl_expansion_of_qoi" in help_text, "Help text missing KL expansion argument"
        assert "--compute_generalized_sobol_indices" in help_text, "Help text missing generalized Sobol argument"
        
        print("✓ Help functionality works correctly!")
        print("Help text preview:")
        print(help_text[:500] + "...")
        
        return True
        
    except Exception as e:
        print(f"✗ Help test failed: {e}")
        return False

def test_json_parsing():
    """Test JSON string parsing functionality."""
    print("\n=== Testing JSON Parsing ===")
    
    try:
        uqsim = uqef.UQsim()
        extended_parser = ExtendedUQSimArgumentParser(uqsim)
        
        # Test valid JSON
        valid_json = '{"Var": true, "StdDev": false, "P10": true}'
        result = extended_parser._parse_json_string(valid_json)
        expected = {"Var": True, "StdDev": False, "P10": True}
        assert result == expected, f"JSON parsing failed: {result} != {expected}"
        
        # Test invalid JSON (should return empty dict)
        invalid_json = '{"Var": true, "StdDev": false'  # Missing closing brace
        result = extended_parser._parse_json_string(invalid_json)
        assert result == {}, f"Invalid JSON should return empty dict, got: {result}"
        
        print("✓ JSON parsing works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ JSON parsing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing UQEF-Dynamic Extended Argument Parser")
    print("=" * 50)
    
    tests = [
        test_extended_args_parsing,
        test_help_functionality,
        test_json_parsing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Extended argument parser is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
