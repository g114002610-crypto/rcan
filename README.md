"# RCAN - Residual Channel Attention Networks

## 14-bit RAW File Support

This repository now supports 14-bit raw image files as input. This feature allows you to process high bit-depth images directly without prior conversion to standard image formats.

### Usage

#### Method 1: Filename-based Dimensions (Recommended)

Name your raw files with dimensions in the filename:
- Format: `filename_WIDTHxHEIGHT.raw` (e.g., `image_1920x1080.raw`)
- Alternative format: `filename_WIDTH_HEIGHT.raw` (e.g., `image_1920_1080.raw`)

```bash
python code/test.py --dir_demo /path/to/raw/images --quant_mode float
```

#### Method 2: Command-line Arguments

If your raw files don't include dimensions in the filename, specify them via command-line:

```bash
python code/test.py --dir_demo /path/to/raw/images \
    --raw_width 1920 \
    --raw_height 1080 \
    --raw_bit_depth 14 \
    --quant_mode float
```

### Parameters

- `--raw_bit_depth`: Bit depth of raw images (default: 14)
- `--raw_width`: Width of raw images in pixels (optional if specified in filename)
- `--raw_height`: Height of raw images in pixels (optional if specified in filename)

### File Format Requirements

- **File Extension**: `.raw`
- **Data Type**: 16-bit unsigned integers (uint16) for 14-bit data
- **Byte Order**: Little-endian
- **Layout**: Row-major order (width × height pixels)
- **Color Space**: Grayscale (will be converted to RGB internally if needed)

### Example

For a 14-bit raw image of size 640x480:

1. **With dimensions in filename:**
   ```bash
   # Your file: image_640x480.raw
   python code/test.py --dir_demo ./raw_images --quant_mode float
   ```

2. **With command-line arguments:**
   ```bash
   # Your file: image.raw
   python code/test.py --dir_demo ./raw_images \
       --raw_width 640 \
       --raw_height 480 \
       --raw_bit_depth 14 \
       --quant_mode float
   ```

### Technical Details

- Raw images are automatically scaled from 14-bit range (0-16383) to 8-bit range (0-255) for processing
- The implementation supports both training and inference modes
- Grayscale raw images are automatically converted to RGB if `--n_colors 3` is used
- Output images are saved as PNG files in the `quantize_result/output` directory

### Supported Image Formats

The following image formats are now supported:
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tif, .tiff)
- **RAW (.raw)** ← New!

### Notes

- For training with raw files, organize your dataset with HR and LR folders containing raw files
- Ensure LR raw files follow the naming convention with scaled dimensions (e.g., if HR is 1920x1080 and scale is 2, LR should be 960x540)
- The bit depth can be adjusted for different raw formats (12-bit, 14-bit, 16-bit, etc.)"  
