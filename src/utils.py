import os
import cv2


def get_video_metadata(video_path: str) -> dict:
    """
    Get metadata of a video capture object.

    Args:
        cap (str): The path to the video.

    Returns:
        dict: A dictionary containing the video's metadata such as width, height, fps, duration, and name.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Video capture is not opened.")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    metadata = {"width": width, "height": height, "fps": fps, "duration": duration, "name": video_name}

    return cap, metadata
