from dataclasses import asdict, dataclass
from typing import List, Optional
import numpy as np
import numpy.ma as ma

np.set_printoptions(threshold=np.inf)


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
    """
    This class is the main output of the pose estimation models.
    It contains the pose estimation results for a video.
    It is a nested object that contains a `FramePoseResult`object for each frame in the video.
    Within each frame pose result, there is a list of `PersonPoseResult` objects, one for each person in the frame.
    Every `PersonPoseResult` contains a list of `PoseKeypoint` objects, one for each keypoint in the model output format, with the x, y coordinates and a confidence score.
    """
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

    def __info__(self, num_of_sample_frames: int = 3) -> dict:
        return {
            "video_name": self.video_name,
            "fps": self.fps,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "num_frames": len(self.frames),
            "sample_frames": self.frames[:num_of_sample_frames] if len(self.frames) > num_of_sample_frames else self.frames,
        }
    
    def to_numpy_ma(self) -> np.ndarray:
        """
        Convert the video pose results from a nested object to a masked array.
        This method is useful for evaluation and plotting in order to work
        with arrays rather than nested objects.
        
        Returns:
            Masked array with shape (num_frames, max_persons, num_keypoints, 2)
            where 2 represents x and y coordinates. Max_persons is the maximum number
            of detected persons in the entire video. Values are masked for frames with 
            fewer persons than max_persons, which means that these values are not included
            in computations (e.g. evaluation or plotting).
        """
        if not self.frames:
            print("Warning: No frames in video pose result.")
            return ma.array(np.zeros((0, 0, 0, 2)))
            
        # Get dimensions
        num_frames = len(self.frames)
        max_persons = max(len(frame.persons) for frame in self.frames)
        num_keypoints = max(
            len(person.keypoints)
            for frame in self.frames
            for person in frame.persons
        ) if any(frame.persons for frame in self.frames) else 0
        
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

    def to_json(self) -> dict:
        return {
            "fps": self.fps,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "frames": [asdict(frame) for frame in self.frames],
            "video_name": self.video_name,
        }

    def __str__(self):
        array = self.to_numpy_ma()
        return f"VideoPoseResult(fps={self.fps}, frame_width={self.frame_width}, frame_height={self.frame_height}, video_name={self.video_name}), frame_values: \n{array}"