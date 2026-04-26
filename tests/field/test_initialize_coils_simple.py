#!/usr/bin/env python3
"""
Test script for the initialize_coils_simple function.
Loops through all input.* files in tests/test_files/ and tests the function with each surface.
"""

import os
import glob
from pathlib import Path
from simsopt.geo import SurfaceRZFourier
from simsopt.util import initialize_coils_simple
from simsopt.util import calculate_modB_on_major_radius
from simsopt.field import BiotSavart


def get_curve_order(curve):
    # Traverse down base_curve attributes to find order
    visited = set()
    while True:
        if hasattr(curve, "order") and getattr(curve, "order") is not None:
            return curve.order
        elif hasattr(curve, "base_curve") and id(curve) not in visited:
            visited.add(id(curve))
            curve = curve.base_curve
        else:
            return None


def test_initialize_coils_simple():
    """Test the initialize_coils_simple function with all input files."""

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
            print(f"  Surface major radius: {s.get_rc(0, 0):.3f}")
            print(f"  Surface minor radius component: {s.get_rc(1, 0):.3f}")
            print(f"  NFP: {s.nfp}")

            # Test 1: Default target B-field (5.7 T)
            print("  Test 1: Default target B-field (5.7 T)")
            coils = initialize_coils_simple(s)

            print(f"  Successfully created {len(coils)} coils")

            # Basic checks
            assert len(coils) > 0, f"No coils created for {filename}"

            # Check that all coils have the expected properties
            for i, coil in enumerate(coils):
                assert hasattr(coil, "curve"), f"Coil {i} missing curve attribute"
                assert hasattr(coil, "current"), f"Coil {i} missing current attribute"
                order = get_curve_order(coil.curve)
                if order is not None:
                    assert order == 16, f"Coil {i} curve has order {order}, expected 16"

            # Verify B-field strength
            bs = BiotSavart(coils)
            B_achieved = calculate_modB_on_major_radius(bs, s)
            B_target = 5.7
            B_tolerance = 0.01 * B_target  # 1% tolerance

            print(f"  Target B-field: {B_target} T")
            print(f"  Achieved B-field: {B_achieved:.3f} T")
            print(f"  Difference: {abs(B_achieved - B_target):.3f} T")
            print(f"  Tolerance: {B_tolerance:.3f} T")

            assert abs(B_achieved - B_target) <= B_tolerance, (
                f"B-field verification failed: achieved {B_achieved:.3f} T, target {B_target} T, tolerance {B_tolerance:.3f} T"
            )

            print("  ✓ B-field verification passed for default target")

            # Test 2: Custom target B-field (3.0 T)
            print("  Test 2: Custom target B-field (3.0 T)")
            coils_custom = initialize_coils_simple(s, target_B=3.0)

            # Verify custom B-field strength
            bs_custom = BiotSavart(coils_custom)
            B_achieved_custom = calculate_modB_on_major_radius(bs_custom, s)
            B_target_custom = 3.0
            B_tolerance_custom = 0.01 * B_target_custom  # 1% tolerance

            print(f"  Target B-field: {B_target_custom} T")
            print(f"  Achieved B-field: {B_achieved_custom:.3f} T")
            print(f"  Difference: {abs(B_achieved_custom - B_target_custom):.3f} T")
            print(f"  Tolerance: {B_tolerance_custom:.3f} T")

            assert abs(B_achieved_custom - B_target_custom) <= B_tolerance_custom, (
                f"B-field verification failed: achieved {B_achieved_custom:.3f} T, target {B_target_custom} T, tolerance {B_tolerance_custom:.3f} T"
            )

            print("  ✓ B-field verification passed for custom target")
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
    test_initialize_coils_simple()
