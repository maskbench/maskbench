import numpy as np
from typing import List

from inference.pose_result import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult


def create_test_pose_result(poses: List[List[List[List[float]]]], video_name = "test_video"):
    frames = []
    for frame_idx, frame in enumerate(poses):
        persons = []
        for person_data in frame:
            keypoints = [
                PoseKeypoint(x=x, y=y)
                for x, y in person_data
            ]
            persons.append(PersonPoseResult(keypoints=keypoints))
        frames.append(FramePoseResult(persons=persons, frame_idx=frame_idx))
    return VideoPoseResult(frames=frames, fps=30, frame_width=1920, frame_height=1080, video_name=video_name)