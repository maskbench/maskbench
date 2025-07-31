import os
import uuid
from typing import Optional
from pathlib import Path


class VideoSample:
    def __init__(
        self,
        video_path: Path,
        metadata: Optional[dict] = None,
    ):
        self.id = str(uuid.uuid4())
        self.path = video_path
        self.metadata = metadata

    def get_filename(self):
        return os.path.splitext(os.path.basename(self.path))[0]
