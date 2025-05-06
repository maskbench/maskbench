from abc import ABC, abstractmethod
import cv2

class PoseEstimator(ABC):
    def __init__(self, model_name: str, config: dict = None):
        """
        Initialize the PoseEstimator with a model name and configuration.
        Args:
            model_name (str): The name of the model (e.g. "yolo-pose-v8", "mediapipe", "openpose", ...).
            config (dict): Configuration dictionary for the model. This can include arbitrary parameters for the model that are necessary for inference (e.g. "confidence_threshold", "weights_file_name", ...).
        """

        self.model_name = model_name
        self.config = config if config else {}

    @abstractmethod
    def estimate_pose(self, video_path: str) -> list:
        """
        Abstract method to estimate the pose of a video using the specific pose estimation model.
        This method should be implemented by subclasses.
        Args:
            video_path (str): The full path to the input video file. 
        Returns:
            list: A list of tensors containing the keypoints for each frame.
        """
        pass