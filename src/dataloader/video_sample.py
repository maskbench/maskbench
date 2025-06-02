import os
import uuid 
from typing import List, Optional 
from pathlib import Path

from evaluation.pose_result import VideoPoseResult

class VideoSample:
    def __init__(self, video_path: Path, gt_pose_path: Optional[Path] = None, metadata: Optional[dict] = None):
        self.id = str(uuid.uuid4())
        self.path = video_path
        self.gt_pose_path = gt_pose_path
        self.metadata = metadata
        self.pose_results = {}

    def get_info(self):
        return [self.path, self.gt_pose_path]

    def get_filename(self):
        return os.path.splitext(os.path.basename(self.path))[0]

    def add_result(self, video_pose_result: VideoPoseResult, model_name: str):
        self.pose_results[model_name] = video_pose_result
        