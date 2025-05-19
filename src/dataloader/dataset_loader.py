from abc import ABC, abstractmethod
import random
from pathlib import Path
from typing import List
from dataloader.video_sample import VideoSample 

class DatasetLoader(ABC):
    def __init__(self, dataset_folder: Path, batch_size: int = 1, shuffle: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples = self._load_samples(dataset_folder)

    @abstractmethod
    def _load_samples(self, dataset_folder: Path) -> List[VideoSample]:
        raise NotImplementedError()

    def __iter__(self):
        indices = list(range(len(self.samples)))
        if self.shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch = [self.samples[j] for j in indices[i:i + self.batch_size]]
            yield batch

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)