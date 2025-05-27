import cv2

def get_video_metadata(video: str | cv2.VideoCapture) -> dict:
    """
    Get metadata of a video capture object.
    
    Args:
        cap (str | cv2.VideoCapture): The path to the video or the video capture object.
        
    Returns:
        dict: A dictionary containing the video's metadata such as width, height, fps, and duration.
    """
    if isinstance(video, str):
        cap = cv2.VideoCapture(video)
    elif isinstance(video, cv2.VideoCapture):
        cap = video

    if not cap.isOpened():
        raise ValueError("Video capture is not opened.")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration
    }
    
    return cap, metadata