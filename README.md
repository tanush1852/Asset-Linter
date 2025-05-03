# Asset-Linter

A utility tool for scanning, optimizing, and managing image and video assets in your project.

## Overview

`assetforge.py` provides comprehensive functionality to analyze and optimize your image and video assets. The tool offers various operations such as compression, format conversion, duplicate detection, and background removal.

## Installation

### Prerequisites

```bash
pip install Pillow rich ffmpeg-python rembg
```

### Setup

1. Navigate to your backend folder:
   ```bash
   cd backend
   ```

2. Create an assets folder for your images:
   ```bash
   mkdir assets
   ```

3. Add your images to the assets folder

## Usage

### 1. Scan & Analyze Assets

Display a summary of all assets in the directory:

```bash
python assetforge.py --dir ./assets --analyze
```

### 2. Lossy Compression

Compress images with quality reduction for smaller file size:

```bash
# Default compression (level 50)
python assetforge.py --dir ./assets --lossy

# Custom compression level (0-100, higher = more compression)
python assetforge.py --dir ./assets --lossy --lossy-level 70
```

**Output:** Creates a `lossy_compressed_[LEVEL]` folder with compressed files.

### 3. Lossless Compression

Compress images without quality loss:

```bash
python assetforge.py --dir ./assets --lossless
```

**Output:** Creates a `lossless_compressed` folder with optimized files.

### 4. Convert Formats

Convert assets to a specified format:

```bash
# Convert images to WebP
python assetforge.py --dir ./assets --convert-to webp

# Convert videos to MP4 (if needed)
python assetforge.py --dir ./assets --convert-to mp4 --include-video
```

**Supported Formats:**
- Images: jpg, png, webp, avif
- Videos: mp4, webm, avi, mov

### 5. Detect Duplicates

Find duplicate files based on content hash:

```bash
python assetforge.py --dir ./assets --dedup
```

### 6. Remove Background

Remove image backgrounds (requires rembg):

```bash
python assetforge.py --dir ./assets --remove-bg
```

### 7. Full Pipeline (Multiple Operations)

Run multiple operations at once:

```bash
# Example: Analyze + Lossy Compress + Convert to WebP + Detect Duplicates
python assetforge.py --dir ./assets --analyze --lossy --convert-to webp --dedup
```

## Notes

- The video processing functionality exists in the code but can be mostly ignored for image-only workflows.
- Background removal functionality may not work .
