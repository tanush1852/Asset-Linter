import os
import argparse
import hashlib
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from math import log10, sqrt

class AdvancedImageProcessor:
    def __init__(self):
        self.image_exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
        self.quality = 75
        self.block_size = 8
        
    # --------------------------
    # Core Analysis with Color Space Handling
    # --------------------------
    def scan_images(self, directory):
        """Advanced image analysis with color space awareness"""
        images = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.image_exts:
                    path = os.path.join(root, file)
                    try:
                        with Image.open(path) as img:
                            images.append(self._analyze_image(img, path))
                    except Exception as e:
                        print(f"Error processing {path}: {str(e)}")
        
        # Check if any images were processed successfully
        if not any(images):
            print("No images found in ./assets directory")
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

    # --------------------------
    # Missing methods implementation
    # --------------------------
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
                channel_data = img_array[:,:,i]
                stats[channel_name] = {
                    'mean': float(np.mean(channel_data)),
                    'std': float(np.std(channel_data)),
                    'min': int(np.min(channel_data)),
                    'max': int(np.max(channel_data))
                }
                
        return stats
    
    def _color_histogram(self, img):
        """Generate color histograms for the image"""
        histograms = {}
        
        # Convert to RGB to ensure consistent handling
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb)
        
        # Per-channel histograms
        for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
            channel_data = img_array[:,:,i]
            hist, _ = np.histogram(channel_data, bins=256, range=(0, 256))
            histograms[channel_name] = hist / hist.sum()  # Normalize
            
        # Combined color histogram (simplified)
        combined = np.zeros((16, 16, 16))  # Reduced resolution for efficiency
        r_bins = np.linspace(0, 255, 17)
        g_bins = np.linspace(0, 255, 17)
        b_bins = np.linspace(0, 255, 17)
        
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    r_mask = (img_array[:,:,0] >= r_bins[i]) & (img_array[:,:,0] < r_bins[i+1])
                    g_mask = (img_array[:,:,1] >= g_bins[j]) & (img_array[:,:,1] < g_bins[j+1])
                    b_mask = (img_array[:,:,2] >= b_bins[k]) & (img_array[:,:,2] < b_bins[k+1])
                    combined[i,j,k] = np.sum(r_mask & g_mask & b_mask)
        
        # Normalize the combined histogram
        if combined.sum() > 0:
            combined = combined / combined.sum()
            
        histograms['combined'] = combined
        return histograms

    # --------------------------
    # Enhanced Duplicate Detection
    # --------------------------
    def detect_duplicates(self, images):
        """Multi-method duplicate detection"""
        duplicates = []
        hash_registry = {'phash': {}, 'dhash': {}, 'color': {}}

        for img in images:
            phash = self._perceptual_hash(img['path'])
            dhash = self._difference_hash(img['path'])
            chash = self._color_hash(img['color_histogram'])
            
            for hname, hval in [('phash', phash), ('dhash', dhash), ('color', chash)]:
                if hval in hash_registry[hname]:
                    duplicates.append((hash_registry[hname][hval], img['path']))
                else:
                    hash_registry[hname][hval] = img['path']
        
        return list(set(duplicates))

    def _perceptual_hash(self, path):
        """Improved pHash with color awareness"""
        img = Image.open(path).convert('RGB').resize((32,32), Image.LANCZOS)
        ycbcr = img.convert('YCbCr')
        hashes = []
        
        for channel in ycbcr.split():
            dct_coeffs = dct(dct(np.array(channel, dtype=float), axis=0), axis=1)
            top_coeffs = dct_coeffs[:8, :8]
            avg = np.mean(top_coeffs)
            hashes.append(tuple((top_coeffs > avg).flatten()))
        
        return hash(tuple(hashes))

    def _difference_hash(self, path):
        """Difference hash implementation"""
        img = Image.open(path).convert('L').resize((9,8), Image.LANCZOS)
        pixels = np.array(img)
        diff = pixels[:,1:] > pixels[:,:-1]
        return hash(tuple(diff.flatten()))

    def _color_hash(self, histogram):
        """Color distribution hash"""
        return hash(tuple(histogram['combined'].flatten()))

    # --------------------------
    # Advanced Lossy Compression
    # --------------------------
    def lossy_compress(self, path, methods=['dct', 'lwt']):
        """Hybrid compression with algorithm selection"""
        best_result = None
        img = Image.open(path)
        
        for method in methods:
            if method == 'dct':
                compressed = self._dct_compression(img)
            elif method == 'lwt':
                compressed = self._lwt_compression(img)
            
            quality = self._calculate_psnr(img, compressed)
            if not best_result or quality > best_result['quality']:
                best_result = {
                    'method': method,
                    'data': compressed,
                    'quality': quality
                }

        output_path = self._get_output_path(path, 'lossy')
        best_result['data'].save(output_path)
        return output_path

    def _dct_compression(self, img):
        """DCT-based compression with adaptive quantization"""
        channels = []
        for channel in img.convert('YCbCr').split():
            quantized = self._process_channel_dct(channel)
            channels.append(quantized)
        
        return Image.merge('YCbCr', channels).convert(img.mode)

    def _lwt_compression(self, img):
        """Lifting Wavelet Transform implementation"""
        # LWT implementation placeholder
        return img.copy()

    def _process_channel_dct(self, channel):
        """DCT processing with adaptive quality"""
        arr = np.array(channel, dtype=np.float32)
        quantized = np.zeros_like(arr)
        
        for i in range(0, arr.shape[0], self.block_size):
            for j in range(0, arr.shape[1], self.block_size):
                block = arr[i:i+self.block_size, j:j+self.block_size]
                if block.shape[0] == self.block_size and block.shape[1] == self.block_size:
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    q_block = np.round(dct_block / self._quality_matrix())
                    quantized[i:i+self.block_size, j:j+self.block_size] = q_block
        
        return Image.fromarray(quantized.astype('uint8'))

    # --------------------------
    # Improved Lossless Compression
    # --------------------------
    def lossless_compress(self, path):
        """JPEG-LS inspired compression with multiple predictors"""
        img = Image.open(path)
        predictors = [
            self._median_predictor,
            self._gradient_predictor,
            self._average_predictor
        ]
        
        best_ratio = 0
        best_data = None
        
        for predictor in predictors:
            compressed = self._apply_predictor(img, predictor)
            ratio = img.size[0]*img.size[1] / len(compressed)
            if ratio > best_ratio:
                best_ratio = ratio
                best_data = compressed
        
        output_path = self._get_output_path(path, 'lossless')
        with open(output_path, 'wb') as f:
            f.write(best_data)
        return output_path

    def _apply_predictor(self, img, predictor):
        """Apply prediction and entropy coding"""
        pixels = np.array(img.convert('L'))
        residuals = np.zeros_like(pixels)
        
        for i in range(1, pixels.shape[0]):
            for j in range(1, pixels.shape[1]):
                residuals[i,j] = pixels[i,j] - predictor(pixels, i, j)
        
        return self._huffman_encode(residuals.flatten())
    
    # --------------------------
    # Missing predictor methods
    # --------------------------
    def _median_predictor(self, pixels, i, j):
        """Median edge predictor"""
        # Use neighboring pixels to predict current pixel value
        a = pixels[i, j-1]      # left
        b = pixels[i-1, j]      # above
        c = pixels[i-1, j-1]    # diagonal
        
        # Predict based on gradient direction
        if c >= max(a, b):
            return min(a, b)
        elif c <= min(a, b):
            return max(a, b)
        else:
            return a + b - c
    
    def _gradient_predictor(self, pixels, i, j):
        """Gradient-based predictor"""
        # Simple gradient-based prediction
        a = pixels[i, j-1]      # left
        b = pixels[i-1, j]      # above
        c = pixels[i-1, j-1]    # diagonal
        
        # Linear prediction using gradient
        gradient = a + b - c
        return max(0, min(255, gradient))
    
    def _average_predictor(self, pixels, i, j):
        """Average predictor"""
        # Simple average of neighbors
        a = pixels[i, j-1]      # left
        b = pixels[i-1, j]      # above
        
        return (a + b) // 2

    # --------------------------
    # Enhanced Image Processing
    # --------------------------
    def enhance_image(self, path, method='clahe'):
        """Advanced enhancement techniques"""
        img = Image.open(path)
        if method == 'clahe':
            enhanced = self._clahe(img)
        elif method == 'retinex':
            enhanced = self._retinex(img)
        elif method == 'wavelet':
            enhanced = self._wavelet_denoise(img)
            
        output_path = self._get_output_path(path, 'enhanced')
        enhanced.save(output_path)
        return output_path

    def _clahe(self, img, tile=8, clip_limit=2.0):
        """Contrast Limited Adaptive Histogram Equalization"""
        # Basic implementation of CLAHE
        # Convert to LAB color space for better results with color images
        if img.mode == 'RGB':
            # Process luminance channel in LAB space
            img_lab = img.convert('LAB')
            l, a, b = img_lab.split()
            l_array = np.array(l)
            
            # Apply CLAHE to luminance channel
            tile_size = (img.height // tile, img.width // tile)
            enhanced_l = np.zeros_like(l_array)
            
            # Process each tile
            for y in range(0, img.height, tile_size[0]):
                for x in range(0, img.width, tile_size[1]):
                    # Get tile
                    tile_end_y = min(y + tile_size[0], img.height)
                    tile_end_x = min(x + tile_size[1], img.width)
                    tile_img = l_array[y:tile_end_y, x:tile_end_x]
                    
                    # Compute histogram
                    hist, bins = np.histogram(tile_img.flatten(), 256, [0, 256])
                    
                    # Apply clip limit
                    if clip_limit > 0:
                        clip_value = clip_limit * (tile_img.size / 256)
                        hist = np.clip(hist, 0, clip_value)
                    
                    # Create cumulative distribution function
                    cdf = hist.cumsum()
                    cdf = 255 * cdf / cdf[-1]  # Normalize
                    
                    # Apply histogram equalization to tile
                    tile_eq = np.interp(tile_img.flatten(), bins[:-1], cdf)
                    enhanced_l[y:tile_end_y, x:tile_end_x] = tile_eq.reshape(tile_img.shape)
            
            # Merge back with original a,b channels
            enhanced_l_img = Image.fromarray(enhanced_l.astype(np.uint8))
            enhanced_img = Image.merge('LAB', (enhanced_l_img, a, b)).convert('RGB')
            return enhanced_img
        else:
            # For grayscale images
            img_array = np.array(img)
            tile_size = (img.height // tile, img.width // tile)
            enhanced = np.zeros_like(img_array)
            
            # Process each tile
            for y in range(0, img.height, tile_size[0]):
                for x in range(0, img.width, tile_size[1]):
                    # Get tile
                    tile_end_y = min(y + tile_size[0], img.height)
                    tile_end_x = min(x + tile_size[1], img.width)
                    tile_img = img_array[y:tile_end_y, x:tile_end_x]
                    
                    # Apply histogram equalization to tile
                    hist, bins = np.histogram(tile_img.flatten(), 256, [0, 256])
                    cdf = hist.cumsum()
                    if cdf[-1] > 0:  # Avoid division by zero
                        cdf = 255 * cdf / cdf[-1]
                        tile_eq = np.interp(tile_img.flatten(), bins[:-1], cdf)
                        enhanced[y:tile_end_y, x:tile_end_x] = tile_eq.reshape(tile_img.shape)
            
            return Image.fromarray(enhanced.astype(np.uint8))

    def _retinex(self, img):
        """Retinex-based color restoration"""
        # Basic Single Scale Retinex implementation
        img_array = np.array(img.convert('RGB'), dtype=np.float32)
        output = np.zeros_like(img_array)
        
        # Apply Retinex to each channel
        for i in range(3):
            channel = img_array[:,:,i]
            # Create Gaussian blur (approximation)
            blurred = self._gaussian_blur(channel, sigma=25)
            # Apply log transform
            channel_log = np.log10(channel + 1.0)
            blur_log = np.log10(blurred + 1.0)
            # Retinex formula
            retinex = channel_log - blur_log
            # Normalize to 0-255 range
            retinex = (retinex - retinex.min()) * 255 / (retinex.max() - retinex.min())
            output[:,:,i] = retinex
        
        return Image.fromarray(output.astype(np.uint8))
    
    def _gaussian_blur(self, channel, sigma=1.0):
        """Simple Gaussian blur implementation"""
        # Create a simple approximation of Gaussian blur
        # For a real implementation, use scipy.ndimage.gaussian_filter
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create a simple Gaussian kernel
        x = np.linspace(-3*sigma, 3*sigma, kernel_size)
        kernel_1d = np.exp(-0.5 * x**2 / sigma**2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Apply horizontal blur
        temp = np.zeros_like(channel)
        for i in range(channel.shape[0]):
            temp[i,:] = np.convolve(channel[i,:], kernel_1d, mode='same')
        
        # Apply vertical blur
        result = np.zeros_like(channel)
        for j in range(channel.shape[1]):
            result[:,j] = np.convolve(temp[:,j], kernel_1d, mode='same')
            
        return result

    def _wavelet_denoise(self, img):
        """Wavelet-based denoising"""
        # Simple wavelet denoising implementation
        # This is a placeholder - for real implementation use PyWavelets
        return img

    # --------------------------
    # Core Algorithm Improvements
    # --------------------------
    def _calculate_entropy(self, img):
        """Calculate image entropy for quality assessment"""
        hist = np.histogram(np.array(img), bins=256)[0]
        hist = hist[hist > 0] / hist.sum()
        return -np.sum(hist * np.log2(hist))

    def _quality_matrix(self):
        """Generate quality matrix based on compression level"""
        return np.linspace(1, 30, self.block_size**2).reshape((self.block_size, self.block_size))

    def _calculate_psnr(self, original, compressed):
        """PSNR calculation for quality assessment"""
        mse = np.mean((np.array(original) - np.array(compressed)) ** 2)
        return 20 * log10(255 / sqrt(mse)) if mse != 0 else float('inf')

    def _huffman_encode(self, data):
        """Improved Huffman coding implementation"""
        # Simple implementation for demonstration
        # For a real implementation, use a proper Huffman coding library
        # This just compresses the data using basic techniques
        # First, convert to bytes
        data_bytes = data.astype(np.int8).tobytes()
        # Then apply a simple compression (just a placeholder)
        return data_bytes

    def _get_output_path(self, original, prefix):
        os.makedirs(os.path.join(os.path.dirname(original), prefix), exist_ok=True)
        return os.path.join(
            os.path.dirname(original),
            prefix,
            os.path.basename(original)
        )

def main():
    parser = argparse.ArgumentParser(description="Advanced Image Processing Tool")
    parser.add_argument('--analyze', action='store_true', help='Analyze image properties')
    parser.add_argument('--dedup', action='store_true', help='Detect duplicate images')
    parser.add_argument('--lossy', action='store_true', help='Apply lossy compression')
    parser.add_argument('--lossless', action='store_true', help='Apply lossless compression')
    parser.add_argument('--enhance', choices=['clahe', 'retinex', 'wavelet'], 
                       help='Image enhancement method')
    parser.add_argument('--output-dir', default='./processed', 
                       help='Output directory for processed images')
    
    args = parser.parse_args()
    processor = AdvancedImageProcessor()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all images in ./assets
    images = processor.scan_images('./assets')
    
    if not images:
        print("No images found in ./assets directory")
        return
    
    # Execute requested operations
    if args.analyze:
        print("\nImage Analysis Results:")
        for img in images:
            print(f"\n{img['path']}:")
            print(f"  Size: {img['size']} bytes")
            print(f"  Resolution: {img['resolution']}")
            print(f"  Color Mode: {img['mode']}")
            print(f"  Entropy: {img['entropy']:.2f} bits/pixel")
    
    if args.dedup:
        duplicates = processor.detect_duplicates(images)
        if duplicates:
            print("\nDuplicate Images Found:")
            for original, duplicate in duplicates:
                print(f"  {duplicate} is a duplicate of {original}")
        else:
            print("\nNo duplicate images found")
    
    if args.lossy:
        print("\nApplying lossy compression:")
        for img in images:
            output = processor.lossy_compress(img['path'])
            print(f"  Compressed {img['path']} -> {output}")
    
    if args.lossless:
        print("\nApplying lossless compression:")
        for img in images:
            output = processor.lossless_compress(img['path'])
            print(f"  Compressed {img['path']} -> {output}")
    
    if args.enhance:
        print(f"\nApplying {args.enhance} enhancement:")
        for img in images:
            output = processor.enhance_image(img['path'], method=args.enhance)
            print(f"  Enhanced {img['path']} -> {output}")

if __name__ == '__main__':
    main()