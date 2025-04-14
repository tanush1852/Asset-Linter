import os
import hashlib
import argparse
from PIL import Image
from rembg import remove
import ffmpeg
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import track

console = Console()

def scan_assets(directory):
    image_exts = ['.jpg', '.jpeg', '.png', '.webp', '.avif']
    video_exts = ['.mp4', '.mov', '.avi', '.webm']
    assets = []

    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in image_exts:
                try:
                    img = Image.open(path)
                    assets.append({
                        'type': 'image', 'path': path, 'format': img.format,
                        'size': os.path.getsize(path), 'resolution': f"{img.width}x{img.height}"
})

                except:
                    pass
            elif ext in video_exts:
                try:
                    probe = ffmpeg.probe(path)
                    video_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
                    assets.append({
                        'type': 'video', 'path': path, 'format': video_stream['codec_name'],
                        'size': os.path.getsize(path), 'resolution': f"{video_stream['width']}x{video_stream['height']}",
                        'duration': float(probe['format']['duration'])
                    })
                except:
                    pass
    return assets

def show_assets(assets):
    table = Table(title="Asset Analysis")
    table.add_column("Type", justify="left")
    table.add_column("Path", justify="left")
    table.add_column("Format", justify="left")
    table.add_column("Size (KB)", justify="right")
    table.add_column("Resolution", justify="center")
    table.add_column("Duration", justify="center")
    
    for asset in assets:
        table.add_row(
            asset['type'], asset['path'], asset['format'],
            f"{asset['size'] // 1024}", asset.get('resolution', '-'),
            str(round(asset.get('duration', 0), 2)) if asset['type'] == 'video' else '-' 
        )
    console.print(table)

def compress_image(path, level):
    img = Image.open(path)
  

    dir_name, file_name = os.path.split(path)
    name, ext = os.path.splitext(file_name)
    compressed_dir = os.path.join(dir_name, f"compressed_{level}")
    os.makedirs(compressed_dir, exist_ok=True)
    compressed_path = os.path.join(compressed_dir, f"{name}_compressed{level}{ext}")

    img.save(compressed_path, optimize=True, quality=100 - level)
    return compressed_path

def convert_image(path, to_format):
    img = Image.open(path)
    base = os.path.splitext(path)[0]
    new_path = f"{base}.{to_format}"
    img.save(new_path, format=to_format.upper())
    return new_path

def convert_video(path, to_format):
    base = os.path.splitext(path)[0]
    new_path = f"{base}.{to_format}"
    ffmpeg.input(path).output(new_path).run(overwrite_output=True)
    return new_path

def detect_duplicates(assets):
    seen = {}
    duplicates = []
    for asset in assets:
        with open(asset['path'], 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash in seen:
                duplicates.append((seen[file_hash], asset['path']))
            else:
                seen[file_hash] = asset['path']
    return duplicates

def remove_background(path):
    img = Image.open(path)
    out = remove(img)
    new_path = path.replace('.', '_nobg.')
    out.save(new_path)
    return new_path

def main():
    parser = argparse.ArgumentParser(description="AssetLinter CLI")
    parser.add_argument('--dir', required=True, help='Directory to scan')
    parser.add_argument('--analyze', action='store_true', help='Analyze assets')
    parser.add_argument('--optimize', type=int, help='Compress images by %')
    parser.add_argument('--convert-img', help='Convert images to specified format (e.g., webp)')
    parser.add_argument('--convert-vid', help='Convert videos to specified format (e.g., avif)')
    parser.add_argument('--dedup', action='store_true', help='Detect duplicates')
    parser.add_argument('--remove-bg', action='store_true', help='Remove image backgrounds')

    args = parser.parse_args()
    assets = scan_assets(args.dir)

    if args.analyze:
        show_assets(assets)

    if args.optimize:
        for asset in track(assets, description="Optimizing..."):
            if asset['type'] == 'image':
                compress_image(asset['path'], args.optimize)

    if args.convert_img:
        for asset in track(assets, description="Converting images..."):
            if asset['type'] == 'image':
                convert_image(asset['path'], args.convert_img)

    if args.convert_vid:
        for asset in track(assets, description="Converting videos..."):
            if asset['type'] == 'video':
                convert_video(asset['path'], args.convert_vid)

    if args.dedup:
        duplicates = detect_duplicates(assets)
        if duplicates:
            console.print("[red]Duplicates Found:[/red]")
            for original, duplicate in duplicates:
                console.print(f"{original} <--> {duplicate}")
        else:
            console.print("[green]No duplicates found.")

    if args.remove_bg:
        for asset in track(assets, description="Removing backgrounds..."):
            if asset['type'] == 'image':
                remove_background(asset['path'])

if __name__ == '__main__':
    main()
