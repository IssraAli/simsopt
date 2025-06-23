#!/usr/bin/env python3
"""
Test script for the optimize_coils_simple function.
Loops through all input.* files in tests/test_files/ and tests the function with each surface.
"""

import os
import glob
import json
from pathlib import Path
from simsopt.geo import SurfaceRZFourier
from simsopt.util.permanent_magnet_helper_functions import optimize_coils_simple

def test_optimize_coils_simple():
    """Test the optimize_coils_simple function with all input files."""
    
    # Clean up previous test output files
    print("Cleaning up previous test output files...")
    test_output_dir = Path('./test_output')
    if test_output_dir.exists():
        import shutil
        shutil.rmtree(test_output_dir)
        print(f"  Removed existing test output directory: {test_output_dir}")
    else:
        print(f"  No existing test output directory found: {test_output_dir}")
    print()
    
    # Get the path to test_files directory
    test_files_dir = Path(__file__).parent.parent / "test_files"
    
    # Find all input.* files
    input_files = glob.glob(str(test_files_dir / "input.*"))
    
    print(f"Found {len(input_files)} input files to test:")
    for f in input_files:
        print(f"  {os.path.basename(f)}")
    print()
    
    successful_tests = 0
    failed_tests = 0
    skipped_files = 0
    
    for input_file in input_files:
        filename = os.path.basename(input_file)
            
        print(f"Testing with {filename}...")
        
        try:
            # Load surface from VMEC input file
            s = SurfaceRZFourier.from_vmec_input(input_file)
        except Exception as e:
            print(f"  Skipping {filename}: could not load as VMEC surface ({e})")
            skipped_files += 1
            print()
            continue
        
        try:
            # print(f"  Surface major radius: {s.get_rc(0, 0):.3f}")
            # print(f"  Surface minor radius component: {s.get_rc(1, 0):.3f}")
            
            # Determine target B-field based on major radius
            major_radius = s.get_rc(0, 0)
            if major_radius < 2.0:
                target_B = 1.0
                print(f"  Using target B-field: {target_B} T (small surface, R < 2.0)")
            else:
                target_B = 5.7
                print(f"  Using target B-field: {target_B} T (large surface, R >= 2.0)")
            
            # Test with 4 coils
            ncoils = 4
            print(f"  Testing with {ncoils} coils...")
            
            # Test with a smaller number of iterations for faster testing
            coils, results = optimize_coils_simple(
                s, 
                target_B=target_B,
                out_dir=f'./test_output/{filename.replace(".", "_")}_ncoils{ncoils}',
                max_iterations=10,
                max_iter_lag=5,
                ncoils=ncoils,
                order=4,
                nphi=8,
                ntheta=8,
                verbose=False
            )
            
            print("  ✓ Function completed successfully!")
            print(f"  Number of coils: {len(coils)}")
            print(f"  Initial B-field: {results['initial_B_field']:.3f} T")
            print(f"  Final B-field: {results['final_B_field']:.3f} T")
            print(f"  Target B-field: {results['target_B_field']:.3f} T")
            print(f"  Optimization time: {results['optimization_time']:.1f} seconds")
            print(f"  Final flux: {results['final_flux']:.2e}")
            print(f"  Output directory: {results['output_directory']}")
            
            # Save results and coils as JSON
            output_dir = Path(results['output_directory'])
            results_file = output_dir / 'optimization_results.json'
            coils_file = output_dir / 'coils_data.json'
            
            # Save results dictionary
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save coils data (convert to serializable format)
            coils_data = []
            for i, coil in enumerate(coils):
                coil_data = {
                    'index': i,
                    'current': float(coil.current.get_value()),
                    'curve_order': getattr(coil.curve, 'order', None),
                    'curve_type': type(coil.curve).__name__
                }
                coils_data.append(coil_data)
            
            with open(coils_file, 'w') as f:
                json.dump(coils_data, f, indent=2)
            
            print(f"  ✓ Results saved to {results_file}")
            print(f"  ✓ Coils data saved to {coils_file}")
            
            # Check that output files were created
            expected_files = [
                'coils_initial.vtu',
                'coils_optimized.vtu', 
                'surface_initial.vts',
                'surface_optimized.vts',
                'biot_savart_optimized.json',
                'optimization_results.json',
                'coils_data.json'
            ]
            
            print("  Checking output files:")
            all_files_present = True
            for expected_file in expected_files:
                file_path = output_dir / expected_file
                if file_path.exists():
                    print(f"    ✓ {expected_file}")
                else:
                    print(f"    ✗ {expected_file} (missing)")
                    all_files_present = False
            
            if not all_files_present:
                raise AssertionError(f"Some output files are missing for {filename}")
            
            print(f"  ✓ All tests passed for {filename}")
            successful_tests += 1
            
        except Exception as e:
            print(f"  ✗ Error with {filename}: {e}")
            failed_tests += 1
        
        print()
    
    print("Test Summary:")
    print(f"  Successful tests: {successful_tests}")
    print(f"  Failed tests: {failed_tests}")
    print(f"  Skipped files: {skipped_files}")
    print(f"  Total files: {len(input_files)}")
    
    if failed_tests > 0:
        raise AssertionError(f"{failed_tests} tests failed")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_optimize_coils_simple() 