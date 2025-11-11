def handle_14_bit_images(image):
    """Process 14-bit images and return the processed data."""
    # Example processing code for 14-bit images
    processed_image = image / 4  # Convert 14-bit to 8-bit for simplicity
    return processed_image


def save_14_bit_image(image, filename):
    """Save the 14-bit image to a file."""
    # Example saving function
    with open(filename, 'wb') as f:
        f.write(image.tobytes())
