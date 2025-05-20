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
        videos_list = [os.path.join(self.dataset_folder, f) 
        for f in os.listdir(self.dataset_folder) 
        if f.lower().endswith((".avi", ".mp4"))]
        
        ground_truth_list = [os.path.join(self.dataset_folder, f)
         for f in os.listdir(self.dataset_folder) 
         if f.lower().endswith((".eaf", ".json"))]
        
        if not len(videos_list) == len(ground_truth_list):
            print("# of Videos != # of Ground Truths")

        samples = []
        for video, ground_truth in zip(sorted(videos_list), sorted(ground_truth_list)):
            print("adding", video, ground_truth)
            samples.append(
                VideoSample([video], ground_truth)
            )

        return samples