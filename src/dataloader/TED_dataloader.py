from dataloader.dataset_loader import DatasetLoader
from dataloader.video_sample import VideoSample 
import glob
from pathlib import Path
from typing import List
import os


class TED_DATALOADER(DatasetLoader):
    def __init__(self, dataset_folder: Path, batch_size: int = 1, shuffle: bool = True):
        self.ground_truth_folder = ""
        self.ground_truth_ext = ".eaf"
        self.videos_folder = ""
        self.video_ext = ".mp4"
        super().__init__(dataset_folder, batch_size, shuffle)
        
        
    def _load_samples(self, dataset_folder: Path) -> List[VideoSample]:
        # assuming videos and ground truths are named the same
        videos_list = glob.glob(f"{dataset_folder}/{self.videos_folder}/*{self.video_ext}")
        ground_truth_list = glob.glob(f"{dataset_folder}/{self.ground_truth_folder}/*{self.ground_truth_ext}")

        print(videos_list)
        print(ground_truth_list)

        if not len(videos_list) == len(ground_truth_list):
            print("# of Videos != # of Ground Truths")

        samples = []
        for video, ground_truth in zip(sorted(videos_list), sorted(ground_truth_list)):
            print("adding", video, ground_truth)
            samples.append(
                VideoSample([video], ground_truth)
            )

        return samples