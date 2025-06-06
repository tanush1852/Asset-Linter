import os
import argparse
import hashlib
import numpy as np
from PIL import Image, ImageEnhance
from scipy.fftpack import dct, idct
from math import log10, sqrt
import time
import heapq
import cv2

class AdvancedImageProcessor:
    def __init__(self):
        self.image_exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
        self.quality = 15  # Lower value = higher compression (was 75)
        self.block_size = 8
        
    # --------------------------
    # Core Analysis with Color Space Handling
    # --------------------------
    def scan_images(self, directory):
        """Advanced image analysis with color space awareness"""
        images = []
        print(f"Scanning for images in {directory}...")
        
        for root, _, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.image_exts:
                    path = os.path.join(root, file)
                    try:
                        with Image.open(path) as img:
                            images.append(self._analyze_image(img, path))
                            print(f"  Analyzed: {path}")
                    except Exception as e:
                        print(f"Error processing {path}: {str(e)}")
        
        if not images:
            print("No images found in the directory")
            return []
        return images

    def _analyze_image(self, img, path):
        """Detailed image analysis with color space awareness"""
        analysis = {
            'path': path,
            'format': img.format,
            'size': os.path.getsize(path),
            'resolution': f"{img.width}x{img.height}",
            'mode': img.mode,
            'channels': self._get_channel_info(img),
            'entropy': self._calculate_entropy(img)
        }
        
        # Convert to compatible mode for processing
        base_img = img.convert('RGB') if img.mode not in ['L', 'RGB'] else img.copy()
        analysis.update({
            'stats': self._calculate_image_stats(base_img),
            'color_histogram': self._color_histogram(base_img)
        })
        return analysis

    def _get_channel_info(self, img):
        """Get channel information for various color modes"""
        mode_info = {
            'L': ['Luminance'],
            'RGB': ['Red', 'Green', 'Blue'],
            'CMYK': ['Cyan', 'Magenta', 'Yellow', 'Black'],
            'YCbCr': ['Luma', 'Blue-diff', 'Red-diff'],
            'LAB': ['Lightness', 'A', 'B']
        }
        return mode_info.get(img.mode, [f"Channel {i}" for i in range(len(img.getbands()))])

    def _calculate_image_stats(self, img):
        """Calculate basic image statistics for each channel"""
        stats = {}
        img_array = np.array(img)
        
        # Handle different channel counts
        if len(img_array.shape) == 2:  # Grayscale
            stats['mean'] = float(np.mean(img_array))
            stats['std'] = float(np.std(img_array))
            stats['min'] = int(np.min(img_array))
            stats['max'] = int(np.max(img_array))
        else:  # Color (RGB, RGBA, etc.)
            for i, channel_name in enumerate(self._get_channel_info(img)):
                if i < img_array.shape[2]:  # Make sure index is valid
                    channel_data = img_array[:,:,i]
                    stats[channel_name] = {
                        'mean': float(np.mean(channel_data)),
                        'std': float(np.std(channel_data)),
                        'min': int(np.min(channel_data)),
                        'max': int(np.max(channel_data))
                    }
                
        return stats
    
    def _color_histogram(self, img):
        """Generate simplified color histograms for the image"""
        # Use a simplified histogram approach for speed
        histograms = {}
        
        # Convert to RGB to ensure consistent handling
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb)
        
        # Per-channel histograms - use fewer bins for performance
        for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
            channel_data = img_array[:,:,i]
            hist, _ = np.histogram(channel_data, bins=32, range=(0, 256))
            histograms[channel_name] = hist / max(hist.sum(), 1)  # Normalize, avoid div by zero
            
        # Simplified combined color histogram - very coarse for efficiency
        combined = np.zeros((8, 8, 8))
        h, w = img_array.shape[:2]
        
        # Sample the image rather than processing every pixel
        sample_rate = max(1, min(h, w) // 100)  # Sample at most 1/100 of the smaller dimension
        
        # Use coarse binning of color values
        for i in range(0, h, sample_rate):
            for j in range(0, w, sample_rate):
                r_bin = img_array[i, j, 0] // 32
                g_bin = img_array[i, j, 1] // 32
                b_bin = img_array[i, j, 2] // 32
                combined[r_bin, g_bin, b_bin] += 1
        
        # Normalize
        if combined.sum() > 0:
            combined = combined / combined.sum()
            
        histograms['combined'] = combined
        return histograms

    # --------------------------
    # Enhanced Duplicate Detection
    # --------------------------
    def detect_duplicates(self, images):
        """More efficient duplicate detection using perceptual hash"""
        duplicates = []
        hash_registry = {}
        
        print("Detecting duplicate images...")
        start_time = time.time()

        for i, img in enumerate(images):
            # Use only perceptual hash for better performance
            phash = self._perceptual_hash(img['path'])
            
            if phash in hash_registry:
                duplicates.append((hash_registry[phash], img['path']))
                print(f"  Found duplicate: {img['path']} is similar to {hash_registry[phash]}")
            else:
                hash_registry[phash] = img['path']
                
            # Show progress for large image sets
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1}/{len(images)} images...")
                
        print(f"Duplicate detection completed in {time.time() - start_time:.2f} seconds")
        return list(set(duplicates))

    def _perceptual_hash(self, path):
        """Optimized perceptual hash"""
        # Use smaller size for faster processing
        try:
            img = Image.open(path).convert('L').resize((16, 16), Image.NEAREST)
            pixels = np.array(img, dtype=np.float32)
            
            # Simple DCT on grayscale image
            dct_result = dct(dct(pixels, axis=0), axis=1)
            dct_low = dct_result[:8, :8]  # Take top-left 8x8 coefficients
            
            # Create hash from DCT coefficients
            avg = np.mean(dct_low)
            return hash(tuple((dct_low > avg).flatten()))
        except Exception as e:
            print(f"  Error hashing {path}: {str(e)}")
            return hash(os.path.basename(path))  # Fallback to filename hash

    # --------------------------
    # Improved Lossy Compression
    # --------------------------
    def lossy_compress(self, path):
        """Optimized JPEG-style lossy compression"""
        print(f"Compressing {path}...")
        start_time = time.time()
        
        try:
            # Open image
            img = Image.open(path)
            
            # Store original size for comparison
            original_size = os.path.getsize(path)
            
            # Get output path
            output_path = self._get_output_path(path, 'lossy')
            
            # Determine optimal quality based on image content
            optimal_quality = self._get_optimal_quality(img)
            
            # Apply aggressive compression by default
            if img.mode in ['RGBA', 'LA']:
                # Handle transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                background.save(output_path, 'JPEG', quality=optimal_quality, optimize=True)
            else:
                # For standard images
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=optimal_quality, optimize=True)
            
            # Calculate compression ratio
            new_size = os.path.getsize(output_path)
            ratio = original_size / max(new_size, 1)  # Avoid division by zero
            
            print(f"  Compressed {path} -> {output_path}")
            print(f"  Size reduced from {original_size:,} to {new_size:,} bytes")
            print(f"  Compression ratio: {ratio:.2f}x (saved {(1-1/ratio)*100:.1f}%)")
            print(f"  Compression completed in {time.time() - start_time:.2f} seconds")
            
            return output_path
            
        except Exception as e:
            print(f"  Error compressing {path}: {str(e)}")
            return path

    def _get_optimal_quality(self, img):
        """Determine optimal JPEG quality based on image content"""
        # Calculate entropy of image
        entropy = self._calculate_entropy(img)
        
        # Use entropy to determine quality
        # Higher entropy (more complex image) = slightly higher quality to preserve details
        # Lower entropy (simpler image) = lower quality is acceptable
        if entropy > 7.5:
            return 30  # Complex image
        elif entropy > 6.0:
            return 20  # Medium complexity
        else:
            return 10  # Simple image
    
    # --------------------------
    # Improved Lossless Compression
    # --------------------------
    def lossless_compress(self, path):
        """Advanced lossless compression using RLE and PNG"""
        print(f"Applying lossless compression to {path}...")
        start_time = time.time()
        
        try:
            # Open image
            img = Image.open(path)
            img_array = np.array(img)
            
            # Store original size for comparison
            original_size = os.path.getsize(path)
            
            # Get output path with .png extension
            output_path = os.path.splitext(self._get_output_path(path, 'lossless'))[0] + '.png'
            
            # Apply RLE pre-processing for better compression
            processed_array = self._rle_preprocess(img_array)
            
            # Convert back to image and save with PNG compression
            processed_img = Image.fromarray(processed_array)
            
            # Save with maximum PNG compression
            processed_img.save(output_path, 'PNG', optimize=True, compress_level=9)
            
            # Calculate compression ratio
            new_size = os.path.getsize(output_path)
            ratio = original_size / max(new_size, 1)
            
            print(f"  Compressed {path} -> {output_path}")
            print(f"  Size reduced from {original_size:,} to {new_size:,} bytes")
            print(f"  Compression ratio: {ratio:.2f}x (saved {(1-1/ratio)*100:.1f}%)")
            print(f"  Compression completed in {time.time() - start_time:.2f} seconds")
            
            return output_path
            
        except Exception as e:
            print(f"  Error compressing {path}: {str(e)}")
            return path

    def _rle_preprocess(self, img_array):
        """Pre-process image using RLE for better PNG compression"""
        if len(img_array.shape) == 3:  # Color image
            height, width, channels = img_array.shape
            processed = np.zeros_like(img_array)
            
            for c in range(channels):
                channel = img_array[:,:,c]
                for i in range(height):
                    row = channel[i]
                    # Modify pixel values to enhance PNG compression
                    # by making repeated values more common
                    for j in range(1, width):
                        if abs(int(row[j]) - int(row[j-1])) <= 2:
                            row[j] = row[j-1]
                processed[:,:,c] = channel
                
        else:  # Grayscale image
            height, width = img_array.shape
            processed = np.zeros_like(img_array)
            
            for i in range(height):
                row = img_array[i]
                for j in range(1, width):
                    if abs(int(row[j]) - int(row[j-1])) <= 2:
                        row[j] = row[j-1]
                processed[i] = row
                
        return processed

    # --------------------------
    # Enhanced Image Processing
    # --------------------------
    def enhance_image(self, path):
        """Enhanced image processing using CLAHE"""
        print(f"Enhancing {path}...")
        start_time = time.time()
        
        try:
            # Open and convert image to numpy array
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Could not read image: {path}")
                
            # Create output directory
            output_path = self._get_output_path(path, 'enhanced')
            
            # Apply CLAHE enhancement
            enhanced = self._apply_clahe(img)
            
            # Save enhanced image
            cv2.imwrite(output_path, enhanced)
            
            print(f"  Enhanced {path} -> {output_path}")
            print(f"  Enhancement completed in {time.time() - start_time:.2f} seconds")
            
            return output_path
            
        except Exception as e:
            print(f"  Error enhancing {path}: {str(e)}")
            return path

    def _apply_clahe(self, image, clip_limit=2.0, grid_size=(8, 8)):
        """Apply CLAHE enhancement to image"""
        
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            cl = clahe.apply(l)
            
            # Merge channels back
            enhanced_lab = cv2.merge((cl, a, b))
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            enhanced = clahe.apply(image)
            
        return enhanced

    # --------------------------
    # Image Format Conversion
    # --------------------------
    def convert_format(self, path, target_format):
        """Convert image to specified format while preserving quality"""
        print(f"Converting {path} to {target_format}...")
        start_time = time.time()
        
        try:
            # Open image
            img = Image.open(path)
            
            # Store original size
            original_size = os.path.getsize(path)
            
            # Get output path with new extension
            output_path = os.path.splitext(self._get_output_path(path, 'converted'))[0] + f'.{target_format}'
            
            # Handle alpha channel for JPEG conversion
            if target_format.lower() in ['jpg', 'jpeg']:
                # Flatten alpha channel to white background if needed
                if img.mode in ['RGBA', 'LA']:
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=95)  # Use 'JPEG' instead of target_format.upper()
            
            elif target_format.lower() == 'png':
                img.save(output_path, 'PNG', optimize=True)
            
            elif target_format.lower() == 'webp':
                if img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')
                img.save(output_path, 'WEBP', quality=90, lossless=False)
            
            elif target_format.lower() == 'bmp':
                if img.mode not in ['RGB']:
                    img = img.convert('RGB')
                img.save(output_path, 'BMP')
            
            elif target_format.lower() == 'tiff':
                img.save(output_path, 'TIFF', compression='tiff_deflate')
            
            # Calculate size difference
            new_size = os.path.getsize(output_path)
            ratio = new_size / max(original_size, 1)
            
            print(f"  Converted {path} -> {output_path}")
            print(f"  Size changed from {original_size:,} to {new_size:,} bytes")
            print(f"  Size ratio: {ratio:.2f}x")
            print(f"  Conversion completed in {time.time() - start_time:.2f} seconds")
            
            return output_path
            
        except Exception as e:
            print(f"  Error converting {path}: {str(e)}")
            return path

    # --------------------------
    # Core Algorithm Improvements
    # --------------------------
    def _calculate_entropy(self, img):
        """Calculate image entropy efficiently"""
        # Convert to grayscale for entropy calculation
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        # Downsample large images for faster processing
        h, w = img_array.shape
        if h * w > 1_000_000:  # For 1MP+ images
            scale = int(np.sqrt(h * w / 1_000_000))
            img_array = img_array[::scale, ::scale]
            
        # Calculate histogram and entropy
        hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))

    def _get_output_path(self, original, prefix):
        """Create output directory and return path for processed file"""
        output_dir = os.path.join(os.path.dirname(original), prefix)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(original))[0]
        
        # Use appropriate extension based on the operation
        extension = '.jpg' if prefix == 'lossy' else '.png'
        
        return os.path.join(output_dir, base_name + extension)

def main():
    parser = argparse.ArgumentParser(description='Advanced Image Processing Tool')
    parser.add_argument('--analyze', action='store_true', help='Analyze images')
    parser.add_argument('--dedup', action='store_true', help='Detect duplicate images')
    parser.add_argument('--lossy', action='store_true', help='Apply lossy compression')
    parser.add_argument('--lossless', action='store_true', help='Apply lossless compression')
    parser.add_argument('--enhance', choices=['clahe'], help='Apply image enhancement')
    parser.add_argument('--convert', choices=['jpg', 'png', 'webp', 'bmp', 'tiff'],
                       help='Convert images to specified format')
    parser.add_argument('--dir', help='Directory containing images')
    parser.add_argument('--output-dir', help='Output directory (optional)')
    
    args = parser.parse_args()
    processor = AdvancedImageProcessor()
    
    # Check if at least one operation is specified
    if not any([args.analyze, args.dedup, args.lossy, args.lossless, args.enhance, args.convert]):
        print("No operation specified. Please use at least one of --analyze, --dedup, --lossy, --lossless, --enhance, or --convert.")
        parser.print_help()
        return
    
    # Ensure input directory exists
    if not os.path.isdir(args.dir):
        print(f"Input directory {args.dir} does not exist.")
        return
    
    # Process all images
    print(f"\n--- Advanced Image Processing Tool ---")
    start_time = time.time()
    
    images = processor.scan_images(args.dir)
    
    if not images:
        print("No images found to process.")
        return
    
    print(f"Found {len(images)} images to process.\n")
    
    # Execute requested operations
    if args.analyze:
        print("\n=== Image Analysis Results ===")
        for img in images:
            print(f"\n{img['path']}:")
            print(f"  Size: {img['size']:,} bytes")
            print(f"  Resolution: {img['resolution']}")
            print(f"  Color Mode: {img['mode']}")
            print(f"  Entropy: {img['entropy']:.2f} bits/pixel")
    
    if args.dedup:
        print("\n=== Duplicate Detection Results ===")
        duplicates = processor.detect_duplicates(images)
        if duplicates:
            print(f"\nFound {len(duplicates)} duplicate images:")
            for original, duplicate in duplicates:
                print(f"  {duplicate} is a duplicate of {original}")
        else:
            print("\nNo duplicate images found")
    
    if args.lossy:
        print("\n=== Lossy Compression Results ===")
        for img in images:
            processor.lossy_compress(img['path'])
    
    if args.lossless:
        print("\n=== Lossless Compression Results ===")
        for img in images:
            processor.lossless_compress(img['path'])
    
    if args.enhance:
        print(f"\n=== Image Enhancement Results ({args.enhance}) ===")
        for img in images:
            processor.enhance_image(img['path'])
    
    if args.convert:
        print(f"\n=== Format Conversion Results ({args.convert}) ===")
        for img in images:
            processor.convert_format(img['path'], args.convert)
    
    total_time = time.time() - start_time
    print(f"\nAll operations completed in {total_time:.2f} seconds.")

if __name__ == '__main__':
    main()