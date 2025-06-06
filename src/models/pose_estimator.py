from abc import ABC, abstractmethod
import cv2

from inference.pose_result import VideoPoseResult

class PoseEstimator(ABC):
    def __init__(self, name: str, config: dict = None):
        """
        Initialize the PoseEstimator with a name and configuration.
        Args:
            name (str): The name of the estimator (e.g. "yolo-pose-v8", "mediapipe", "openpose", ...).
            config (dict): Configuration dictionary for the pose estimator. This can include arbitrary parameters for the model that are necessary for inference (e.g. "confidence_threshold", "weights_file_name", ...).
        """

        self.name = name
        self.config = config if config else {}

    @abstractmethod
    def estimate_pose(self, video_path: str) -> VideoPoseResult:
        """
        Abstract method to estimate the pose of a video using the specific pose estimation model.
        This method should be implemented by subclasses.
        Args:
            video_path (str): The full path to the input video file. 
        Returns:
            list: A list of tensors containing the keypoints for each frame.
        """
        pass
    
    @abstractmethod
    def get_keypoint_pairs(self) -> list:
        pass