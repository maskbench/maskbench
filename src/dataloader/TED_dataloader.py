from dataloader.dataset_loader import DatasetLoader
from dataloader.video_sample import VideoSample 
from pathlib import Path
from typing import List
import os


class TedDataloader(DatasetLoader):
    def __init__(self, dataset_folder: str, shuffle: bool = False, batch_size: int = 1):
        super().__init__(dataset_folder, shuffle, batch_size)
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[VideoSample]:
        # assuming videos and ground truths are named the same
        # Collect video files
        video_extensions = (".avi", ".mp4")
        samples = []

        for filename in os.listdir(self.dataset_folder):
            video_path = os.path.join(self.dataset_folder, filename)
            if filename.endswith(video_extensions):
                samples.append(VideoSample(video_path))

        return samples