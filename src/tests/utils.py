import numpy as np
from typing import List

from inference.pose_result import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult


def create_example_video_pose_result(keypoints_data, video_name="example_video", fps: int = 30):
    """Helper function to create a VideoPoseResult from keypoints data.
    
    Args:
        keypoints_data: List of frames, where each frame is a list of persons,
                       and each person is a list of (x,y) coordinates
    """
    frames = []
    for frame_idx, frame_data in enumerate(keypoints_data):
        persons = []
        for person_data in frame_data:
            keypoints = [
                PoseKeypoint(x=float(x), y=float(y))
                for x, y in person_data
            ]
            persons.append(PersonPoseResult(keypoints=keypoints))
        frames.append(FramePoseResult(persons=persons, frame_idx=frame_idx))
    
    return VideoPoseResult(
        fps=fps,
        frame_width=1920,
        frame_height=1080,
        frames=frames,
        video_name=video_name
    )