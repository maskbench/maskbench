import os
import subprocess
import argparse
from pathlib import Path

def convert_video(input_path, output_path):
    """Convert video to web-compatible format using H.264 codec"""
    command = [
        'ffmpeg',
        '-i', input_path,  # Input file
        '-c:v', 'libx264',  # Use H.264 codec
        '-preset', 'medium',  # Encoding preset (trade-off between speed and compression)
        '-movflags', '+faststart',  # Enable fast start for web playback
        '-y',  # Overwrite output file if it exists
        output_path
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Successfully converted {input_path}")
        # Replace original file with converted one
        os.remove(input_path)
        os.rename(output_path, input_path)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e}")

def convert_all_videos_in_directory(directory):
    """Convert all MP4 videos in the given directory"""
    directory = Path(directory)
    for video_file in directory.glob("*.mp4"):
        temp_output = str(video_file.parent / f"{video_file.stem}_converted{video_file.suffix}")
        convert_video(str(video_file), temp_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert videos in a directory to web-compatible format using H.264 codec')
    parser.add_argument('directory', type=str, help='Directory containing the videos to convert')
    args = parser.parse_args()

    # Verify the directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        exit(1)

    print(f"Converting videos in {args.directory}")
    convert_all_videos_in_directory(args.directory)