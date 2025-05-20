from abc import ABC, abstractmethod
import random
from pathlib import Path
from typing import List
from dataloader.video_sample import VideoSample 

class DatasetLoader(ABC):
    def __init__(self, config: dict):
        self.config = config 

    @abstractmethod
    def _load_samples(self, dataset_folder: Path) -> List[VideoSample]:
        raise NotImplementedError()

    def __iter__(self):
        indices = list(range(len(self.samples)))
        if self.config.get("shuffle"):
            random.shuffle(indices)
        for i in range(0, len(indices), self.config.get("batch_size")):
            batch = [self.samples[j] for j in indices[i:i + self.config.get("batch_size")]]
            yield batch

    def __len__(self):
        return math.ceil(len(self.samples) / self.config.get("batch_size"))