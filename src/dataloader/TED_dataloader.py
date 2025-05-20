from dataloader.dataset_loader import DatasetLoader
from dataloader.video_sample import VideoSample 
import glob
from pathlib import Path
from typing import List
import os


class TED_DATALOADER(DatasetLoader):
    def __init__(self, config: dict):
        super().__init__(config)
        self.samples = self._load_samples()
        
        
    def _load_samples(self) -> List[VideoSample]:
        # assuming videos and ground truths are named the same
        videos_path = os.path.join(
            self.config.get("dataset_folder"), self.config.get("video_folder"), f"*{self.config.get("video_extension")}"
        )
        ground_truth_path = os.path.join(
            self.config.get("dataset_folder"), self.config.get("ground_truth_folder"), f"*{self.config.get("ground_truth_extension")}"
        )
        
        videos_list = glob.glob(videos_path)
        ground_truth_list = glob.glob(ground_truth_path)

        if not len(videos_list) == len(ground_truth_list):
            print("# of Videos != # of Ground Truths")

        samples = []
        for video, ground_truth in zip(sorted(videos_list), sorted(ground_truth_list)):
            print("adding", video, ground_truth)
            samples.append(
                VideoSample([video], ground_truth)
            )

        return samples