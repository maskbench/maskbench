import math
from abc import ABC, abstractmethod
from typing import List

from .video_sample import VideoSample
from inference import VideoPoseResult


class Dataset(ABC):
    def __init__(self, dataset_folder: str):
        self.dataset_folder = dataset_folder

    @abstractmethod
    def _load_samples(self) -> List[VideoSample]:
        """
        Load video samples from the dataset folder.
        This method should be overridden by subclasses to implement specific loading logic.
        """
        raise NotImplementedError()

    def get_gt_pose_results(self) -> List[VideoPoseResult]:
        """
        Load ground truth pose results from the dataset folder.
        This method should be overridden by subclasses if ground truth data is available.
        The order of the results should match the order of the samples returned by _load_samples.
        """
        return []

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return math.ceil(len(self.samples))
