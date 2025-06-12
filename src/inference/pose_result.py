from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import numpy.ma as ma


@dataclass
class PoseKeypoint:
    x: float
    y: float
    confidence: Optional[float] = None


@dataclass
class PersonPoseResult:
    keypoints: List[PoseKeypoint]  # Fixed length per pose estimator (e.g., 17 for COCO)
    id: Optional[int] = None  # for tracking across frames


@dataclass
class FramePoseResult:
    persons: List[PersonPoseResult]
    frame_idx: int


class VideoPoseResult:
    def __init__(
        self,
        fps: int,
        frame_width: int,
        frame_height: int,
        frames: List[FramePoseResult],
        video_name: str,
    ):
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frames = frames
        self.video_name = video_name

    def to_numpy_ma(self) -> np.ndarray:
        """
        Convert the video pose results to a masked array.
        
        Returns:
            Masked array with shape (num_frames, max_persons, num_keypoints, 2)
            where 2 represents x and y coordinates. Values are masked for:
            - Frames with fewer persons than max_persons
            - Missing keypoints
        """
        if not self.frames:
            print("Warning: No frames in video pose result.")
            return ma.array(np.zeros((0, 0, 0, 2)))
            
        # Get dimensions
        num_frames = len(self.frames)
        max_persons = max(len(frame.persons) for frame in self.frames)
        num_keypoints = len(self.frames[0].persons[0].keypoints) if self.frames[0].persons else 0
        
        if max_persons == 0 or num_keypoints == 0:
            print("Warning: No persons or keypoints found in video pose result.")
            return ma.array(np.zeros((num_frames, 0, 0, 2)))
        
        # Initialize arrays - all values masked by default
        values = np.zeros((num_frames, max_persons, num_keypoints, 2))
        mask = np.ones_like(values, dtype=bool)  # True means masked
        
        for frame_idx, frame in enumerate(self.frames):
            # Only fill and unmask values for persons that exist
            for person_idx, person in enumerate(frame.persons):
                for kpt_idx, keypoint in enumerate(person.keypoints):
                    values[frame_idx, person_idx, kpt_idx, 0] = keypoint.x
                    values[frame_idx, person_idx, kpt_idx, 1] = keypoint.y
                    mask[frame_idx, person_idx, kpt_idx] = False  # Unmask only existing values
        
        return ma.array(values, mask=mask)
