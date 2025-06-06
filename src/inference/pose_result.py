from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PoseKeypoint:
    x: float
    y: float
    confidence: Optional[float] = None

@dataclass
class PersonPoseResult:
    keypoints: List[PoseKeypoint]  # Fixed length per pose estimator (e.g., 17 for COCO)
    id: Optional[int] = None # for tracking across frames

@dataclass
class FramePoseResult:
    persons: List[PersonPoseResult]
    frame_idx: int


class VideoPoseResult:
    def __init__(self, fps: int, frame_width: int, frame_height: int, frames: List[FramePoseResult]):
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frames = frames