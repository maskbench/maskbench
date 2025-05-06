from video_chunker import VideoChunker


def main():
    video_chunker = VideoChunker(chunk_length=120, slide=0)
    video_path = "ted-talks/0080_song.mp4"
    chunks = video_chunker.chunk_video(video_path)


if __name__ == "__main__":
    main()