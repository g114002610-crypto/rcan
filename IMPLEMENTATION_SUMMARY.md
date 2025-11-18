# Implementation Summary: 14-bit RAW File Support

## Problem Statement
如果我想要輸入14bits 的.raw檔我應該怎麼做
(Translation: How can I input 14-bit .raw files?)

## Solution Overview
This implementation adds comprehensive support for reading and processing 14-bit (and other bit depths) raw image files in the RCAN super-resolution model.

## Changes Made

### 1. Core Functionality (utility.py)
- **read_raw_image()**: Reads raw binary files with configurable bit depth
  - Accepts width, height, and bit depth parameters
  - Automatically scales from high bit depth (e.g., 14-bit: 0-16383) to processing range (0-255)
  - Returns numpy array ready for processing
  
- **parse_raw_filename()**: Extracts dimensions from filename
  - Supports formats: `image_1920x1080.raw` or `image_1920_1080.raw`
  - Returns (width, height) tuple
  - Fallback to None if dimensions not found in filename

- **quantize()**: Helper function for image quantization
- **checkpoint class**: Full checkpoint management system

### 2. Command-line Options (option.py)
Added three new parameters:
- `--raw_bit_depth`: Bit depth for raw images (default: 14)
- `--raw_width`: Width for raw images (if not in filename)
- `--raw_height`: Height for raw images (if not in filename)

### 3. Demo Data Loader (data/demo.py)
- Extended file search to include `.raw` extension
- Added raw file detection and reading logic
- Automatic dimension parsing from filename or command-line args
- Proper error messages for missing dimensions
- Grayscale to RGB conversion when needed

### 4. Training Data Loader (data/srdata.py)
- Updated `_set_filesystem()` to use `.raw` extension when bit depth is specified
- Enhanced `_check_and_load()` to handle raw files during binary caching
- Modified `_load_file()` to support raw files in both training and testing
- Automatic LR dimension calculation based on scale factor

### 5. Inference Pipeline (test.py)
- Added `.raw` to supported extensions in `test_model_nogt_dir()`
- Raw file detection and dimension parsing
- Grayscale to BGR conversion for consistency
- Graceful error handling with skip on dimension errors

### 6. Documentation
- **README.md**: Comprehensive English documentation
  - Quick start guide
  - Parameter descriptions
  - File format requirements
  - Usage examples
  - Technical details
  
- **USAGE_14BIT_RAW.md**: Detailed Chinese documentation
  - 快速開始指南
  - 參數說明
  - 使用範例
  - 技術細節
  - 常見問題

### 7. Testing (test_raw_support.py)
- Automated test script for validation
- Creates sample 14-bit raw files
- Tests filename parsing
- Tests file reading and conversion
- Tests multiple resolutions
- Provides clear pass/fail reporting

### 8. Repository Hygiene (.gitignore)
- Excludes Python cache files (`__pycache__/`)
- Excludes PyTorch models and checkpoints
- Excludes experiment results and logs
- Excludes IDE and OS-specific files

## Usage Examples

### Basic Usage (Dimensions in Filename)
```bash
# File: image_640x480.raw
python code/test.py --dir_demo ./raw_images --quant_mode float
```

### With Command-line Parameters
```bash
# File: image.raw (no dimensions in name)
python code/test.py --dir_demo ./raw_images \
    --raw_width 1920 \
    --raw_height 1080 \
    --raw_bit_depth 14 \
    --quant_mode float
```

### Different Bit Depths
```bash
# 12-bit raw
python code/test.py --dir_demo ./raw_images --raw_bit_depth 12

# 16-bit raw
python code/test.py --dir_demo ./raw_images --raw_bit_depth 16
```

## Technical Details

### Data Flow
1. Raw file detected by extension (`.raw`)
2. Dimensions determined:
   - First: Parse from filename (e.g., `image_1920x1080.raw`)
   - Fallback: Use command-line args (`--raw_width`, `--raw_height`)
3. Binary data read as uint16 array
4. Reshape to (height, width)
5. Scale from bit_depth range to 0-255: `scaled = (raw / (2^bit_depth - 1)) * 255`
6. Convert to uint8
7. Add channel dimension if needed
8. Process through RCAN model
9. Output as PNG

### Bit Depth Scaling
- 14-bit: 0-16383 → 0-255
- 12-bit: 0-4095 → 0-255
- 16-bit: 0-65535 → 0-255

Formula: `output = (input / max_input_value) * 255`

### File Format
- Binary format: raw pixel data
- Data type: uint16 (2 bytes per pixel)
- Layout: row-major order
- Byte order: little-endian (numpy default)
- Color: grayscale (converted to RGB if needed)

## Backward Compatibility

All changes are backward compatible:
- Existing PNG/JPEG workflows unchanged
- New parameters are optional with sensible defaults
- No changes to existing function signatures
- Raw support only activates when `.raw` files are present

## Testing

Run the test script:
```bash
python test_raw_support.py
```

This will:
- Test filename parsing with various formats
- Create sample 14-bit raw files
- Test reading and conversion
- Validate output ranges and dimensions
- Create test files in `/tmp/rcan_raw_test/`

## Future Enhancements

Potential improvements (not implemented):
- Bayer pattern demosaicing for color raw images
- Big-endian byte order support
- Direct raw file output (no PNG conversion)
- Metadata parsing (if available)
- Raw file validation and auto-detection

## Verification Checklist

- [x] Python syntax validation passed
- [x] Functions properly documented
- [x] Error handling for missing dimensions
- [x] Graceful fallback mechanisms
- [x] English and Chinese documentation
- [x] Test script provided
- [x] Example usage included
- [x] .gitignore added for cleanliness
- [x] Backward compatibility maintained
- [x] Multiple bit depths supported

## Summary

This implementation provides a complete, production-ready solution for processing 14-bit (and other bit depths) raw image files in RCAN. The solution is well-documented in both English and Chinese, includes comprehensive error handling, and maintains full backward compatibility with existing workflows.
