import os
import hashlib
import argparse
import subprocess
import shutil
from PIL import Image
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()

class AssetForge:
    def __init__(self):
        self.image_exts = ['.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp', '.tiff']
        self.video_exts = ['.mp4', '.mov', '.avi', '.webm', '.mkv']
        self.console = Console()
        
    def scan_assets(self, directory):
        """Scan a directory for image and video assets"""
        assets = []

        for root, _, files in os.walk(directory):
            for file in files:
                path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if ext in self.image_exts:
                    try:
                        img = Image.open(path)
                        assets.append({
                            'type': 'image', 
                            'path': path, 
                            'format': img.format,
                            'size': os.path.getsize(path), 
                            'resolution': f"{img.width}x{img.height}"
                        })
                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to process image {path}: {e}[/yellow]")
                
                elif ext in self.video_exts:
                    try:
                        # Use ffprobe to get video info
                        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                              "-show_entries", "stream=width,height,codec_name:format=duration", 
                              "-of", "default=noprint_wrappers=1", path]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        info = {}
                        for line in result.stdout.splitlines():
                            if '=' in line:
                                key, value = line.split('=')
                                info[key] = value
                        
                        assets.append({
                            'type': 'video', 
                            'path': path, 
                            'format': info.get('codec_name', 'unknown'),
                            'size': os.path.getsize(path), 
                            'resolution': f"{info.get('width', '?')}x{info.get('height', '?')}",
                            'duration': float(info.get('duration', 0))
                        })
                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to process video {path}: {e}[/yellow]")
        
        return assets

    def show_assets(self, assets):
        """Display assets in a table"""
        image_count = sum(1 for asset in assets if asset['type'] == 'image')
        video_count = sum(1 for asset in assets if asset['type'] == 'video')
        total_size = sum(asset['size'] for asset in assets)
        
        self.console.print(Panel(f"Found [bold]{len(assets)}[/bold] assets ([green]{image_count}[/green] images, [blue]{video_count}[/blue] videos) - Total size: [yellow]{self._format_size(total_size)}[/yellow]"))
        
        table = Table(title="Asset Analysis")
        table.add_column("Type", justify="left", style="cyan")
        table.add_column("Path", justify="left")
        table.add_column("Format", justify="left")
        table.add_column("Size", justify="right")
        table.add_column("Resolution", justify="center")
        table.add_column("Duration", justify="center")
        
        for asset in assets:
            table.add_row(
                asset['type'], 
                os.path.basename(asset['path']), 
                asset['format'],
                self._format_size(asset['size']), 
                asset.get('resolution', '-'),
                f"{asset.get('duration', 0):.2f}s" if asset['type'] == 'video' else '-' 
            )
        self.console.print(table)

    def _format_size(self, size_bytes):
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024 or unit == 'GB':
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024

    # Lossy compression techniques
    def lossy_compress_image(self, path, level):
     """Compress image using lossy techniques with RGBA support"""
     img = Image.open(path)
     dir_name, file_name = os.path.split(path)
     name, ext = os.path.splitext(file_name)
     compressed_dir = os.path.join(dir_name, f"lossy_compressed_{level}")
     os.makedirs(compressed_dir, exist_ok=True)
    
    # Handle RGBA images by converting them to RGB with white background
     if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        img = background

    # Choose output format
     output_ext = '.jpg' if ext.lower() in ['.png', '.bmp', '.tiff'] else ext.lower()
     compressed_path = os.path.join(compressed_dir, f"{name}{output_ext}")
    
     quality = max(5, 100 - level)  # Ensure minimum quality
    
    # Save with appropriate settings
     if output_ext in ['.jpg', '.jpeg']:
        img.save(compressed_path, format='JPEG', quality=quality, optimize=True)
     elif output_ext == '.webp':
        img.save(compressed_path, format='WEBP', quality=quality)
     else:
        img.save(compressed_path, quality=quality, optimize=True)
        
     return compressed_path

    # Lossless compression techniques
    def lossless_compress_image(self, path):
        """Compress image using lossless techniques"""
        img = Image.open(path)
        dir_name, file_name = os.path.split(path)
        name, ext = os.path.splitext(file_name)
        compressed_dir = os.path.join(dir_name, "lossless_compressed")
        os.makedirs(compressed_dir, exist_ok=True)
        
        # Choose appropriate lossless format based on content
        if img.mode == 'RGBA' or 'transparency' in img.info:
            # Use PNG for images with transparency
            output_ext = '.png'
            compressed_path = os.path.join(compressed_dir, f"{name}{output_ext}")
            
            # Optimize PNG
            img.save(compressed_path, format='PNG', optimize=True, compress_level=9)
            
            # Further optimize with external tool if available
            try:
                optipng_path = shutil.which('optipng')
                if optipng_path:
                    subprocess.run([optipng_path, "-o7", compressed_path], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
                
        else:
            # Use WebP lossless for RGB images
            output_ext = '.webp'
            compressed_path = os.path.join(compressed_dir, f"{name}{output_ext}")
            img.save(compressed_path, format='WEBP', lossless=True, quality=100)
            
        return compressed_path

    def lossy_compress_video(self, path, level):
        """Compress video using lossy techniques"""
        dir_name, file_name = os.path.split(path)
        name, ext = os.path.splitext(file_name)
        compressed_dir = os.path.join(dir_name, f"lossy_compressed_video_{level}")
        os.makedirs(compressed_dir, exist_ok=True)
        compressed_path = os.path.join(compressed_dir, f"{name}{ext}")
        
        # Calculate CRF (Constant Rate Factor) based on level (0-100)
        # For H.264: 0 (lossless) to 51 (worst quality), 23 is default
        # Map our 0-100 level to 18-35 range for reasonable quality
        crf = 18 + (level / 100.0) * 17
        
        # Determine if we should also resize the video for higher compression levels
        resize_option = []
        if level > 70:
            resize_option = ["-vf", "scale=iw/2:ih/2"]  # Half the resolution
        elif level > 40:
            resize_option = ["-vf", "scale=iw*0.75:ih*0.75"]  # 75% of original resolution
            
        # Build FFmpeg command for lossy compression
        cmd = [
            "ffmpeg", "-i", path, 
            "-c:v", "libx264", "-crf", str(int(crf)),
            "-preset", "slow",  # Higher compression, slower encoding
            "-c:a", "aac", "-b:a", f"{128 if level < 50 else 96}k"
        ]
        
        # Add resize option if needed
        if resize_option:
            cmd.extend(resize_option)
            
        # Output file
        cmd.append(compressed_path)
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return compressed_path

    def lossless_compress_video(self, path):
        """Compress video using lossless techniques"""
        dir_name, file_name = os.path.split(path)
        name, ext = os.path.splitext(file_name)
        compressed_dir = os.path.join(dir_name, "lossless_compressed_video")
        os.makedirs(compressed_dir, exist_ok=True)
        compressed_path = os.path.join(compressed_dir, f"{name}_lossless.mkv")
        
        # Use FFV1 codec for lossless video compression
        cmd = [
            "ffmpeg", "-i", path,
            "-c:v", "ffv1", "-level", "3", "-g", "1", "-threads", "8",
            "-c:a", "flac",  # Lossless audio too
            compressed_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return compressed_path

    def convert_image(self, path, to_format):
        """Convert image to specified format"""
        img = Image.open(path)
        dir_name, file_name = os.path.split(path)
        name = os.path.splitext(file_name)[0]
        convert_dir = os.path.join(dir_name, f"converted_{to_format}")
        os.makedirs(convert_dir, exist_ok=True)
        
        new_path = os.path.join(convert_dir, f"{name}.{to_format}")
        
        # Handle special cases for each format
        if to_format.lower() == 'jpg' or to_format.lower() == 'jpeg':
            # Convert to RGB mode for JPEG (strips alpha)
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                background.save(new_path, 'JPEG', quality=95)
            else:
                img.convert('RGB').save(new_path, 'JPEG', quality=95)
                
        elif to_format.lower() == 'png':
            img.save(new_path, 'PNG', optimize=True)
            
        elif to_format.lower() == 'webp':
            img.save(new_path, 'WEBP', lossless=False, quality=90)
            
        elif to_format.lower() == 'avif':
            # AVIF support may require additional libraries
            try:
                img.save(new_path, 'AVIF', quality=90)
            except Exception:
                # Fallback to using external tools if available
                temp_png = f"{os.path.splitext(new_path)[0]}_temp.png"
                img.save(temp_png, 'PNG')
                
                try:
                    subprocess.run(["avifenc", "-s", "6", temp_png, new_path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    os.remove(temp_png)
                except Exception:
                    console.print(f"[red]Failed to convert to AVIF: {path}[/red]")
                    return path
        else:
            # Generic case
            img.save(new_path, to_format.upper())
            
        return new_path

    def convert_video(self, path, to_format):
        """Convert video to specified format"""
        dir_name, file_name = os.path.split(path)
        name = os.path.splitext(file_name)[0]
        convert_dir = os.path.join(dir_name, f"converted_{to_format}")
        os.makedirs(convert_dir, exist_ok=True)
        
        new_path = os.path.join(convert_dir, f"{name}.{to_format}")
        
        # Format-specific encoding parameters
        codec_options = []
        
        if to_format.lower() == 'mp4':
            codec_options = ["-c:v", "libx264", "-crf", "23", "-preset", "medium", 
                            "-c:a", "aac", "-b:a", "128k"]
        elif to_format.lower() == 'webm':
            codec_options = ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", 
                            "-c:a", "libopus", "-b:a", "96k"]
        elif to_format.lower() == 'av1':
            # AV1 encoding, slow but efficient
            codec_options = ["-c:v", "libaom-av1", "-crf", "30", "-b:v", "0",
                            "-strict", "experimental", "-c:a", "libopus", "-b:a", "128k"]
        elif to_format.lower() == 'hevc' or to_format.lower() == 'h265':
            codec_options = ["-c:v", "libx265", "-crf", "28", "-preset", "medium",
                            "-c:a", "aac", "-b:a", "128k"]
            to_format = "mp4"  # Use mp4 container for HEVC
            new_path = os.path.join(convert_dir, f"{name}.{to_format}")
        elif to_format.lower() == 'gif':
            # For GIF, we need to use a special palette approach for better quality
            palette_path = os.path.join(convert_dir, f"{name}_palette.png")
            
            # Generate palette
            subprocess.run([
                "ffmpeg", "-i", path, 
                "-vf", "fps=10,scale=320:-1:flags=lanczos,palettegen", 
                palette_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Create GIF using palette
            subprocess.run([
                "ffmpeg", "-i", path, "-i", palette_path,
                "-filter_complex", "fps=10,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse", 
                new_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Clean up palette
            os.remove(palette_path)
            return new_path
        
        # Run the conversion
        cmd = ["ffmpeg", "-i", path] + codec_options + [new_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return new_path

    def detect_duplicates(self, assets):
        """Detect duplicate files using checksums"""
        console.print("[bold]Analyzing for duplicates...[/bold]")
        seen = {}
        duplicates = []
        
        for asset in track(assets, description="Calculating file hashes"):
            try:
                with open(asset['path'], 'rb') as f:
                    # For large files, read chunks to save memory
                    if asset['size'] > 50 * 1024 * 1024:  # 50 MB
                        # For large files, use a fast hash of the first 1MB + middle 1MB + last 1MB
                        hasher = hashlib.md5()
                        
                        # First 1MB
                        hasher.update(f.read(1024 * 1024))
                        
                        # Middle 1MB
                        f.seek(max(0, asset['size'] // 2 - 512 * 1024))
                        hasher.update(f.read(1024 * 1024))
                        
                        # Last 1MB
                        f.seek(max(0, asset['size'] - 1024 * 1024))
                        hasher.update(f.read(1024 * 1024))
                        
                        file_hash = hasher.hexdigest()
                    else:
                        # For smaller files, hash the entire content
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    
                    if file_hash in seen:
                        duplicates.append((seen[file_hash], asset['path']))
                    else:
                        seen[file_hash] = asset['path']
            except Exception as e:
                console.print(f"[yellow]Warning: Could not hash {asset['path']}: {e}[/yellow]")
        
        return duplicates

    def remove_background(self, path):
        """Remove image background using alpha matting"""
        try:
            # Try to use rembg if available (best results)
            try:
                from rembg import remove
                img = Image.open(path)
                out = remove(img)
                
                dir_name, file_name = os.path.split(path)
                name = os.path.splitext(file_name)[0]
                nobg_dir = os.path.join(dir_name, "nobg")
                os.makedirs(nobg_dir, exist_ok=True)
                
                new_path = os.path.join(nobg_dir, f"{name}_nobg.png")
                out.save(new_path)
                return new_path
            except ImportError:
                # Fallback to external tools if available
                dir_name, file_name = os.path.split(path)
                name = os.path.splitext(file_name)[0]
                nobg_dir = os.path.join(dir_name, "nobg")
                os.makedirs(nobg_dir, exist_ok=True)
                
                new_path = os.path.join(nobg_dir, f"{name}_nobg.png")
                
                # Try using imagemagick
                if shutil.which('convert'):
                    subprocess.run([
                        "convert", path, 
                        "-fuzz", "10%", 
                        "-transparent", "white", 
                        new_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return new_path
                else:
                    console.print("[yellow]Warning: Background removal requires rembg or ImageMagick.[/yellow]")
                    return path
        except Exception as e:
            console.print(f"[red]Error removing background: {e}[/red]")
            return path

    def apply_batch_operations(self, assets, args):
        """Apply operations to multiple assets"""
        results = {
            'compressed_lossy': 0,
            'compressed_lossless': 0,
            'converted': 0,
            'bg_removed': 0
        }
        
        # Process images
        image_assets = [a for a in assets if a['type'] == 'image']
        video_assets = [a for a in assets if a['type'] == 'video']
        
        # Lossy compression
        if args.lossy:
            level = args.lossy_level if args.lossy_level is not None else 50
            console.print(f"[bold]Applying lossy compression (level {level})...[/bold]")
            
            for asset in track(image_assets, description="Compressing images with lossy techniques"):
                self.lossy_compress_image(asset['path'], level)
                results['compressed_lossy'] += 1
                
            if args.include_video:
                for asset in track(video_assets, description="Compressing videos with lossy techniques"):
                    self.lossy_compress_video(asset['path'], level)
                    results['compressed_lossy'] += 1
                    
        # Lossless compression
        if args.lossless:
            console.print("[bold]Applying lossless compression...[/bold]")
            
            for asset in track(image_assets, description="Compressing images with lossless techniques"):
                self.lossless_compress_image(asset['path'])
                results['compressed_lossless'] += 1
                
            if args.include_video:
                for asset in track(video_assets, description="Compressing videos with lossless techniques"):
                    self.lossless_compress_video(asset['path'])
                    results['compressed_lossless'] += 1
                    
        # Format conversion
        if args.convert_to:
            console.print(f"[bold]Converting assets to {args.convert_to} format...[/bold]")
            
            for asset in track(assets, description=f"Converting to {args.convert_to}"):
                if asset['type'] == 'image':
                    self.convert_image(asset['path'], args.convert_to)
                    results['converted'] += 1
                elif asset['type'] == 'video' and args.include_video:
                    self.convert_video(asset['path'], args.convert_to)
                    results['converted'] += 1
                    
        # Background removal
        if args.remove_bg:
            console.print("[bold]Removing backgrounds from images...[/bold]")
            
            for asset in track(image_assets, description="Removing backgrounds"):
                self.remove_background(asset['path'])
                results['bg_removed'] += 1
                
        return results


def main():
    parser = argparse.ArgumentParser(
        description="AssetForge - Advanced CLI Asset Modifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze assets in a directory
  python assetforge.py --dir ./assets --analyze
  
  # Apply lossy compression (level 70) to all images
  python assetforge.py --dir ./assets --lossy --lossy-level 70
  
  # Apply lossless compression to all images
  python assetforge.py --dir ./assets --lossless
  
  # Convert all images to WebP format
  python assetforge.py --dir ./assets --convert-to webp
  
  # Find duplicate assets
  python assetforge.py --dir ./assets --dedup
  
  # Apply multiple operations at once
  python assetforge.py --dir ./assets --analyze --lossy --convert-to webp --dedup
        """
    )
    
    parser.add_argument('--dir', required=True, help='Directory to scan')
    parser.add_argument('--analyze', action='store_true', help='Analyze assets')
    
    # Compression options
    compression_group = parser.add_argument_group('Compression Options')
    compression_group.add_argument('--lossy', action='store_true', 
                                 help='Apply lossy compression techniques')
    compression_group.add_argument('--lossy-level', type=int, default=50, 
                                 help='Compression level for lossy (0-100, default: 50)')
    compression_group.add_argument('--lossless', action='store_true', 
                                 help='Apply lossless compression techniques')
    
    # Conversion options
    conversion_group = parser.add_argument_group('Conversion Options')
    conversion_group.add_argument('--convert-to', 
                                help='Convert to specified format (jpg, png, webp, avif, mp4, webm)')
    
    # Other operations
    operation_group = parser.add_argument_group('Additional Operations')
    operation_group.add_argument('--dedup', action='store_true', 
                               help='Detect and list duplicate assets')
    operation_group.add_argument('--remove-bg', action='store_true', 
                               help='Remove backgrounds from images')
    
    # Additional flags
    parser.add_argument('--include-video', action='store_true', 
                      help='Include video files in operations (may be slow)')
    parser.add_argument('--output-dir', 
                      help='Custom output directory for processed files')

    args = parser.parse_args()
    
    # Create AssetForge instance
    asset_forge = AssetForge()
    
    # Start scanning
    console.print(f"[bold green]Scanning directory: {args.dir}[/bold green]")
    assets = asset_forge.scan_assets(args.dir)
    
    if not assets:
        console.print("[red]No assets found in the specified directory.[/red]")
        return
    
    # Show analysis if requested or if no specific operation is selected
    if args.analyze or not any([args.lossy, args.lossless, args.convert_to, args.dedup, args.remove_bg]):
        asset_forge.show_assets(assets)
    
    # Check for duplicates
    if args.dedup:
        duplicates = asset_forge.detect_duplicates(assets)
        if duplicates:
            console.print(f"[red]Found {len(duplicates)} duplicate files:[/red]")
            table = Table(title="Duplicate Assets")
            table.add_column("Original", style="green")
            table.add_column("Duplicate", style="red")
            
            for original, duplicate in duplicates:
                table.add_row(original, duplicate)
            console.print(table)
            
            if Confirm.ask("Would you like to list redundant files that can be safely deleted?"):
                for original, duplicate in duplicates:
                    console.print(f"rm '{duplicate}'")
        else:
            console.print("[green]No duplicate assets found.[/green]")
    
    # Apply batch operations
    operations_results = None
    if any([args.lossy, args.lossless, args.convert_to, args.remove_bg]):
        operations_results = asset_forge.apply_batch_operations(assets, args)
        
        # Show summary of operations
        if operations_results:
            console.print("\n[bold green]Operations completed:[/bold green]")
            for op, count in operations_results.items():
                if count > 0:
                    console.print(f"- {op.replace('_', ' ').title()}: {count} file(s)")

if __name__ == '__main__':
    main()