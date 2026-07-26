import math
import os
from abc import ABC
from typing import Dict, List
from pathlib import Path

from .video_sample import VideoSample
from pose_result_class import VideoPoseResult
from keypoint_pairs import COCO_KEYPOINT_PAIRS


class Dataset(ABC):
    def __init__(self, name: str, video_folder: str, gt_folder: str = None, config: dict = None):
        self.name = name
        self.config = config
        self.video_folder = Path(video_folder)
        self.gt_folder = Path(gt_folder) if gt_folder else None  # Optional - None if dataset has no ground truth
        self.video_samples = self.load_videos()

    def load_videos(self) -> List[VideoSample]:
        """
        Default implementation to load video samples from the videos folder.
        Expects videos to be directly in the videos folder.
        """
        video_extensions = (".avi", ".mp4")
        samples = []

        if not self.video_folder.exists():
            raise ValueError(f"Videos folder not found at {self.video_folder}")

        video_files = []
        for ext in video_extensions:
            video_files.extend(self.video_folder.rglob(f"*{ext}"))

        for filename in video_files:
                samples.append(VideoSample(filename))

        return samples

    def get_gt_pose_results(self) -> Dict[str, VideoPoseResult]:
        """
        Default implementation to load ground truth pose results from the gt_folder.
        Expects one JSON file per video with the same name as the video file.
        The format of the ground truth files should be consistent with `VideoPoseResult` structure, otherwise overwrite
        this method in a subclass and implement your own logic to load the ground truth data.
        The returned dictionary should map video names to ground truth `VideoPoseResult` objects.
        Returns empty dict if no gt_folder is specified or doesn't exist.
        """
        if self.gt_folder is None or not self.gt_folder.exists():
            return {}

        gt_pose_results = {}
        for sample in self.video_samples:
            video_name = sample.video_path.stem
            json_path = self.gt_folder / f"{video_name}.json"

            if not json_path.exists():
                raise ValueError(f"Ground truth JSON file missing for video `{video_name}`.")
            
            gt_pose_results[video_name] = VideoPoseResult.from_json(json_path, video_name)

        return gt_pose_results

    def get_single_pose_result(self, video_name: str) -> VideoPoseResult:
        """
        Default implementation to load a single ground truth pose result from the gt_folder.
        Expects one JSON file per video with the same name as the video file.
        The format of the ground truth files should be consistent with `VideoPoseResult` structure, otherwise overwrite
        this method in a subclass and implement your own logic to load the ground truth data.
        Returns None if no gt_folder is specified or doesn't exist.
        """
        if self.gt_folder is None or not self.gt_folder.exists():
            return None

        json_path = self.gt_folder / f"{video_name}.json"

        if not json_path.exists():
            return None
        
        return VideoPoseResult.from_json(json_path, video_name)

    def get_gt_keypoint_pairs(self) -> None | List[tuple]:
        """
        Default implementation to return COCO keypoint pairs if gt_folder is specified and exists,
        otherwise returns None.
        """
        if self.gt_folder is not None and self.gt_folder.exists():
            return COCO_KEYPOINT_PAIRS
        return None

    def __iter__(self):
        return iter(self.video_samples)

    def __len__(self):
        return math.ceil(len(self.video_samples))
