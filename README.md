# Asset-Linter

So Asset-Forge.py is the new file  with the updated code in the backend folder, I have included functions for photos as well as videos but mostly we will scrape out the video part.
Pre installation: pip install Pillow rich ffmpeg-python rembg
cd backend
First create a new folder inside the backend folder , name it as assets folder and add all your images in that folder.
Below are the list of commands :

1. Scan & Analyze Assets
Displays a summary of all assets (images/videos) in the directory.
python assetforge.py --dir ./assets --analyze

2. Lossy Compression
Compresses images/videos with quality reduction (smaller file size).
# Default compression (level 50)
python assetforge.py --dir ./assets --lossy

# Custom compression level (0-100, higher = more compression)
python assetforge.py --dir ./assets --lossy --lossy-level 70
Output:

Creates a lossy_compressed_[LEVEL] folder with compressed files.

3. Lossless Compression
Compresses images/videos without quality loss (optimized file size).

python assetforge.py --dir ./assets --lossless
Output:

Creates a lossless_compressed folder with optimized files.


4. Convert Formats
Converts all assets to a specified format (e.g., webp, avif, mp4, webm).
# Convert images to WebP
python assetforge.py --dir ./assets --convert-to webp

# Convert videos to MP4
python assetforge.py --dir ./assets --convert-to mp4 --include-video
Supported Formats:

Images: jpg, png, webp, avif

Videos: mp4, webm, avi, mov


5. Detect Duplicates
Finds duplicate files (based on content hash).

python assetforge.py --dir ./assets --dedup


6. Remove Background[MIGHT NOT WORK]
Removes image backgrounds (requires rembg).
python assetforge.py --dir ./assets --remove-bg

7. Full Pipeline (Multiple Operations)
Run multiple operations at once:
# Analyze + Lossy Compress + Convert to WebP + Detect Duplicates
python assetforge.py --dir ./assets --analyze --lossy --convert-to webp --dedup
