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


    def chunk_video(self, video_path: str, output_path:str) -> list:
        """
        Chunk the video into smaller segments of specified length.

        Args:
            video_path (str): The path to the video file relative to the datasets directory.

        Returns:
            list: A list video file chunks.
        """

        # These are locations within the docker container (mounted directories)
        # video_path = os.path.join("/datasets", video_path)
        os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
        chunks = []

        with VideoFileClip(video_path, audio=False) as video: # we need to look into audio. writing chunk will change accordingly 
            if not video:
                raise ValueError(f"Could not open video file: {video_path}")
            duration = video.duration
            step = self.chunk_length - self.slide

            i = 0
            chunk_num = 1
            while i < duration:
                start = i
                end = min(start + self.chunk_length, duration)

                if end - start > 0:
                    chunk = video.subclipped(start, end).without_audio()
                    chunk.filename = f"chunk_{chunk_num}.mp4"
                    chunk_path = os.path.join(output_path, chunk.filename)
                    if not chunk.audio:
                        chunk.write_videofile(chunk_path, codec="libx264", audio=False)
                    else:
                        chunk.write_videofile(chunk_path, codec="libx264", audio_codec="aac")  
                    chunks.append(chunk_path)
                else: 
                    break

                i += step
                chunk_num += 1

        return chunks