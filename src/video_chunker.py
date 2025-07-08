import os
import cv2


class VideoChunker:
    def __init__(self, chunk_length: int):
        """
        Initialize the VideoChunker with a specified chunk length.

        Args:
            chunk_length (int): The length of each video chunk in seconds.
            slide (int): The overlap duration of chunks in seconds. Default is 0.
        """
        self.chunk_length = chunk_length

    def chunk_video_using_opencv(self, video_path: str, output_path: str) -> list:
        """
        Chunk the video into smaller segments of specified length using OpenCV.
        For videos shorter than chunk_length, the entire video will be kept as one chunk.
        For videos longer than chunk_length, they will be split into chunks of chunk_length seconds.

        Args:
            video_path (str): The path to the video file relative to the datasets directory.
            output_path (str): The directory to save the video chunks.

        Returns:
            list: A list of video file chunks.
        """
        os.makedirs(output_path, exist_ok=True)
        chunk_paths = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        num_chunk_frames = int(self.chunk_length * fps)

        frames_written_in_current_chunk = 0
        chunk_num = 1
        video_writer = None
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if video_writer is None:
                chunk_path = os.path.join(output_path, f"chunk_{chunk_num}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(chunk_path, fourcc, fps, (width, height))
                chunk_paths.append(chunk_path)

            video_writer.write(frame)
            frames_written_in_current_chunk += 1

            if frames_written_in_current_chunk >= num_chunk_frames:
                video_writer.release()
                video_writer = None
                frames_written_in_current_chunk = 0
                chunk_num += 1

            frame_idx += 1

        if video_writer is not None:
            video_writer.release()

        cap.release()
        return chunk_paths
