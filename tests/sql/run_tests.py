#!/usr/bin/env python3
"""
Test runner for SQL factory tests
"""

import os
import sys
import subprocess

def run_tests():
    """Run all SQL factory tests"""
    
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    print("Running SQL Factory Tests...")
    print("=" * 50)
    
    try:
        # Run pytest on the sql test directory
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/sql/', 
            '-v',
            '--tb=short'
        ], capture_output=True, text=True, cwd=project_root)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        return result.returncode
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)