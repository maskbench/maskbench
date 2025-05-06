import os

from moviepy import VideoFileClip


class VideoChunker:
    def __init__(self, chunk_length: int, slide: int = 0):
        """
        Initialize the VideoChunker with a specified chunk length.

        Args:
            chunk_length (int): The length of each video chunk in seconds.
            slide (int): The overlap duration of chunks in seconds. Default is 0.
        """
        self.chunk_length = chunk_length
        self.slide = slide


    def chunk_video(self, video_path: str) -> list:
        """
        Chunk the video into smaller segments of specified length.

        Args:
            video_path (str): The path to the video file relative to the datasets directory.

        Returns:
            list: A list video file chunks.
        """

        # These are locations within the docker container (mounted directories)
        video_path = os.path.join("/datasets", video_path)
        video_file_name = os.path.splitext(os.path.basename(video_path))[0]
        chunks = []

        with VideoFileClip(video_path) as video:
            if not video:
                raise ValueError(f"Could not open video file: {video_path}")
            duration = video.duration
            step = self.chunk_length - self.slide

            i = 0
            chunk_num = 1
            while i < duration:
                start = i
                end = min(start + self.chunk_length, duration)

                chunk = video.subclipped(start, end)
                chunk.filename = f"{video_file_name}_chunk{chunk_num}.mp4"
                chunks.append(chunk)

                i += step
                chunk_num += 1

        return chunks