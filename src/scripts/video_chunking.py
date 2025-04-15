import argparse
import os

from moviepy import VideoFileClip


def main():
    parser = argparse.ArgumentParser(description="Chunk a video into smaller segments.")
    parser.add_argument("--video_path", type=str, help="The path to the input video file relative to the dataset directory configured in the .env file.")
    parser.add_argument("--output_folder", type=str, default="", help="Path to the output folder for video chunks.")
    parser.add_argument("--chunk_length", type=float, default=10, help="Duration of each video chunk in seconds (default: 10).")
    parser.add_argument("--slide", type=float, default=0.01, help="Slide duration between chunks in seconds (default: 0.01).")
    
    args = parser.parse_args()

    chunk_length = args.chunk_length
    slide = args.slide

    # These are locations within the docker container (mounted directories)
    video_path = os.path.join("/datasets", args.video_path)
    video_file_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = "/output"

    with VideoFileClip(video_path) as video:
        if not video:
            raise ValueError(f"Could not open video file: {video_path}")
        duration = video.duration
        step = chunk_length - slide

        os.makedirs(out_dir, exist_ok=True)

        i = 0
        chunk_num = 1
        while i < duration:
            start = i
            end = min(start + chunk_length, duration)

            chunk = video.subclipped(start, end)
            chunk.write_videofile(os.path.join(out_dir, f"{video_file_name}_chunk{chunk_num}.mp4"))

            i += step
            chunk_num += 1

if __name__ == "__main__":
    main()
