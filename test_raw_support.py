#!/usr/bin/env python3
"""
Test script for 14-bit RAW file support in RCAN
This script demonstrates how to create and read 14-bit raw files.
"""

import os
import sys
import numpy as np

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

def create_sample_raw_file(filename, width, height, bit_depth=14):
    """
    Create a sample 14-bit raw file for testing.
    
    Args:
        filename: Output filename (should end with .raw)
        width: Image width in pixels
        height: Image height in pixels
        bit_depth: Bit depth (default: 14)
    """
    print(f"Creating sample raw file: {filename}")
    print(f"  Dimensions: {width}x{height}")
    print(f"  Bit depth: {bit_depth}")
    
    # Create a gradient pattern
    max_val = (1 << bit_depth) - 1  # 2^14 - 1 = 16383 for 14-bit
    
    # Create vertical gradient
    pattern = np.linspace(0, max_val, height, dtype=np.uint16)
    img = np.tile(pattern[:, np.newaxis], (1, width))
    
    # Save as raw binary file
    img.tofile(filename)
    print(f"  File size: {os.path.getsize(filename)} bytes")
    print(f"  Expected size: {width * height * 2} bytes")
    print(f"  Min value: {img.min()}, Max value: {img.max()}")


def test_read_raw_image(filename, width, height, bit_depth=14):
    """
    Test reading a raw file using the utility function.
    
    Args:
        filename: Raw file to read
        width: Image width
        height: Image height
        bit_depth: Bit depth (default: 14)
    """
    print(f"\nTesting read_raw_image function:")
    print(f"  Reading: {filename}")
    
    from utility import read_raw_image
    
    img = read_raw_image(filename, width, height, bit_depth)
    
    print(f"  Output shape: {img.shape}")
    print(f"  Output dtype: {img.dtype}")
    print(f"  Output range: [{img.min()}, {img.max()}]")
    print(f"  Expected range: [0, 255]")
    
    if img.min() >= 0 and img.max() <= 255:
        print("  ✓ Output range is correct!")
    else:
        print("  ✗ Output range is incorrect!")
        return False
    
    if img.shape == (height, width):
        print("  ✓ Output shape is correct!")
    else:
        print("  ✗ Output shape is incorrect!")
        return False
    
    return True


def test_parse_filename():
    """Test filename parsing function."""
    print("\nTesting parse_raw_filename function:")
    
    from utility import parse_raw_filename
    
    test_cases = [
        ("image_1920x1080.raw", (1920, 1080)),
        ("test_640x480.raw", (640, 480)),
        ("image_1920_1080.raw", (1920, 1080)),
        ("test_640_480.raw", (640, 480)),
        ("no_dimensions.raw", (None, None)),
    ]
    
    all_passed = True
    for filename, expected in test_cases:
        result = parse_raw_filename(filename)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {filename} -> {result} (expected {expected})")
        if result != expected:
            all_passed = False
    
    return all_passed


def main():
    """Main test function."""
    print("=" * 60)
    print("Testing 14-bit RAW file support for RCAN")
    print("=" * 60)
    
    # Create test directory
    test_dir = "/tmp/rcan_raw_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test 1: Parse filename
    print("\n[Test 1] Filename parsing")
    test1_passed = test_parse_filename()
    
    # Test 2: Create and read raw file with dimensions in filename
    print("\n[Test 2] Create and read raw file (dimensions in filename)")
    test_file = os.path.join(test_dir, "test_640x480.raw")
    create_sample_raw_file(test_file, 640, 480, 14)
    test2_passed = test_read_raw_image(test_file, 640, 480, 14)
    
    # Test 3: Different resolution
    print("\n[Test 3] Create and read raw file (different resolution)")
    test_file2 = os.path.join(test_dir, "test_1920x1080.raw")
    create_sample_raw_file(test_file2, 1920, 1080, 14)
    test3_passed = test_read_raw_image(test_file2, 1920, 1080, 14)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    print(f"  Test 1 (Filename parsing): {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Test 2 (Read 640x480 raw): {'PASS' if test2_passed else 'FAIL'}")
    print(f"  Test 3 (Read 1920x1080 raw): {'PASS' if test3_passed else 'FAIL'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 60)
    
    if all_passed:
        print("\n✓ 14-bit RAW file support is working correctly!")
        print(f"\nSample raw files created in: {test_dir}")
        print("You can use these files for testing with:")
        print(f"  python code/test.py --dir_demo {test_dir} --quant_mode float")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
