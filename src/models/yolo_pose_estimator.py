import os
from pathlib import Path
import cv2

from ultralytics import YOLO, settings

from models.pose_estimator import PoseEstimator


class YoloPoseEstimator(PoseEstimator):
    def __init__(self, model_name: str, config: dict):
        """
        Initialize the YoloPoseEstimator with a model name and configuration.
        Args:
            model_name (str): The name of the model (e.g. "yolo-pose-v8", "yolo-pose-v11").
            config (dict): Configuration dictionary for the model. It must contain the key "weights" with the path to the weights file relative to the weights folder. Note that MaskBench does not download the weights for you. Please visit https://docs.ultralytics.com/tasks/pose/ to download the weights.
        """

        super().__init__(model_name, config)

        weights_file_name = self.config.get("weights")
        weights_file_path = os.path.join("/weights", weights_file_name)
        if not os.path.exists(weights_file_path):
            raise ValueError(f"Could not find weights file under {weights_file_path}. Please download the weights from https://docs.ultralytics.com/tasks/pose/ and place them in the weights folder.")

        settings.update({"weights_dir": "/weights"})
        self.model = YOLO(weights_file_name)
        

    def estimate_pose(self, video_path: str) -> list:
        """
        Estimate the pose of a video using YOLO pose estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            list: A list of tensors containing the keypoints for each frame.
        """

        confidence = self.config.get("confidence_threshold", 0.85)
        results = self.model.track(video_path, conf=confidence, stream=True)

        keypoints_tensor_list = []

        for result in results:
            keypoints_tensor_list.append(result.keypoints.xy)

        # TODO: we migth refactor this to use an xarray with a frame, person, keypoint and dimension
        # or create a dedicated PoseData class that also stores the layout of the keypoints
        return keypoints_tensor_list