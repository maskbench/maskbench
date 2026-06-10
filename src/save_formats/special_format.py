from abc import ABC, abstractmethod
from typing import Dict, List

from pose_result_class import VideoPoseResult
from checkpointer import Checkpointer

class SpecialFormat(ABC):
    """Base class for all special save formats in MaskBench."""
    
    def __init__(self, name: str, checkpointer: Checkpointer):
        """
        Initialize a special save format.
        
        Args:
            name: Unique name of the metric
            checkpointer: Checkpointer instance for saving results
        """
        self.name = name
        self.checkpointer = checkpointer

    @abstractmethod
    def create(
        self,
        video_pose_results: Dict[str, Dict[str, List[VideoPoseResult]]],
    ) -> None:
        """
        Create the necessary directories and files for saving pose results in a special format. 
        
        Args:
            video_pose_results: Pose estimation results for all videos
            
        Returns:
            None
        """
        pass