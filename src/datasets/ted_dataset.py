import os
from typing import List

from .dataset import Dataset
from .video_sample import VideoSample


class TedDataset(Dataset):
    def __init__(self, name: str, dataset_folder: str, config: dict = None):
        super().__init__(name, dataset_folder, config)
        self.samples = self._load_samples()

    def _load_samples(self) -> List[VideoSample]:
        video_extensions = (".avi", ".mp4")
        samples = []

        for filename in os.listdir(self.dataset_folder):
            video_path = os.path.join(self.dataset_folder, filename)
            if filename.endswith(video_extensions):
                samples.append(VideoSample(video_path))

        return samples
