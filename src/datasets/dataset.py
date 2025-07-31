import math
from abc import ABC, abstractmethod
from typing import Dict, List

from .video_sample import VideoSample
from inference import VideoPoseResult


class Dataset(ABC):
    def __init__(self, name: str, dataset_folder: str, config: dict = None):
        self.name = name
        self.dataset_folder = dataset_folder
        self.config = config
        self.samples = self._load_samples()

    @abstractmethod
    def _load_samples(self) -> List[VideoSample]:
        """
        Load video samples from the dataset folder.
        This method should be overridden by subclasses to implement specific loading logic.
        It should return one VideoSample for each video in the dataset.
        """
        raise NotImplementedError()

    def get_gt_pose_results(self) -> Dict[str, VideoPoseResult]:
        """
        Load ground truth pose results from the dataset folder.
        This method should be overridden by subclasses if ground truth data is available.
        The returned dictionary should map video names to ground truth`VideoPoseResult` objects.
        """
        return {}

    def get_gt_keypoint_pairs(self) -> None | List[tuple]:
        """
        Load ground truth keypoint pairs from the dataset folder.
        This method should be overridden by subclasses if ground truth data is available.
        """
        return None

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return math.ceil(len(self.samples))
