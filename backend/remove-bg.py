#!/usr/bin/env python3
import os
import argparse
from PIL import Image, ImageDraw
import numpy as np
from rembg import remove


def remove_background_rembg(input_path, output_path):
    """
    Removes the background from an image using the rembg library.
    """
    try:
        with open(input_path, 'rb') as i:
            input_data = i.read()
            output_data = remove(input_data)
        with open(output_path, 'wb') as o:
            o.write(output_data)
        print(f"Background removed (rembg) and saved to '{output_path}'")
        return True
    except Exception as e:
        print(f"Error using rembg: {e}")
        return False


def remove_background_barebones(input_path, output_path, background_color_hex="#FFFFFF", tolerance=30):
    """
    Removes a (mostly) solid background color from an image.
    This is a very basic approach for demonstration.

    :param input_path: Path to the input image.
    :param output_path: Path to save the output image (will be PNG).
    :param background_color_hex: Hex string of the background color to target.
    :param tolerance: How close a pixel's color needs to be to the background_color to be removed.
    """
    try:
        img = Image.open(input_path).convert("RGBA")
        data = np.array(img)

        # Convert hex background_color to RGB tuple
        hex_color = background_color_hex.lstrip('#')
        bg_r, bg_g, bg_b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # r, g, b are uint8 arrays with shape (Width, Height)
        r, g, b = data[..., :3].T

        # Identify pixels close to the background color
        # Cast to np.int16 before subtraction to avoid uint8 overflow/underflow issues
        is_background = (
            (abs(r.astype(np.int16) - bg_r) < tolerance) &
            (abs(g.astype(np.int16) - bg_g) < tolerance) &
            (abs(b.astype(np.int16) - bg_b) < tolerance)
        )

        # Set alpha to 0 for background pixels
        data[..., -1][is_background.T] = 0

        processed_img = Image.fromarray(data)
        processed_img.save(output_path, "PNG")
        print(f"Background removed (barebones) and saved to '{output_path}'")
        return True
    except Exception as e:
        print(f"Error using barebones method: {e}")
        return False


def ensure_output_dir(output_path):
    """Ensures the output directory exists."""
    dir_to_create = os.path.dirname(output_path)
    if dir_to_create:
        os.makedirs(dir_to_create, exist_ok=True)


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Remove backgrounds from images.')
    parser.add_argument('input_image', help='Path to the input image')
    parser.add_argument(
        '-o', '--output', help='Path to save the output image (default: input_filename_nobg.png)')
    parser.add_argument('-m', '--method', choices=['rembg', 'barebones'], default='rembg',
                        help='Method to use for background removal (default: rembg)')
    parser.add_argument('-c', '--color', default='#FFFFFF',
                        help='Background color to remove for barebones method (hex format, default: #FFFFFF)')
    parser.add_argument('-t', '--tolerance', type=int, default=30,
                        help='Color tolerance for barebones method (0-255, default: 30)')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_image):
        print(f"Error: Input file '{args.input_image}' does not exist")
        return

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input_image))[0]
        output_dir = os.path.dirname(args.input_image)
        output_path = os.path.join(output_dir, f"{base_name}_nobg.png")

    # Ensure output directory exists
    ensure_output_dir(output_path)

    # Remove background using the selected method
    if args.method == 'rembg':
        remove_background_rembg(args.input_image, output_path)
    else:  # barebones
        remove_background_barebones(args.input_image, output_path,
                                    background_color_hex=args.color,
                                    tolerance=args.tolerance)


if __name__ == '__main__':
    main()
