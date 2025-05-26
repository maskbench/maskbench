import uuid 
from typing import List, Optional 
from pathlib import Path

class VideoSample:
    def __init__(self, video_paths: List[Path], gt_pose_path: Optional[Path] = None, metadata: Optional[dict] = None):
        self.id = str(uuid.uuid4())
        self.video_paths = video_paths # this is needed for datasets like BioCV, where a single movement consists of 8 different camera views
        self.gt_pose_path = gt_pose_path
        self.metadata = metadata

    def get_video_path(self):
        return self.video_paths
    
    def get_info(self):
        return [self.video_paths, self.gt_pose_path]
        