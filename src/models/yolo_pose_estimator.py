import os
from pathlib import Path
import cv2
from ultralytics import YOLO, settings
import torch

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

        weights_file = self.config.get("weights")
        pre_built_weights_file_path = os.path.join("/weights/pre_built", weights_file)
        user_weights_file_path = os.path.join("/weights/user_weights", weights_file)
          
        if os.path.exists(pre_built_weights_file_path):
            weights_file_path = pre_built_weights_file_path
        elif os.path.exists(user_weights_file_path):
            weights_file_path = user_weights_file_path 
        else:
            raise ValueError(f"Could not find weights file under {weights_file}. Please download the weights from https://docs.ultralytics.com/tasks/pose/ and place them in the weights folder.")

        self.model = YOLO(weights_file_path)
        # only for dev
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("yolo is using", device)
        self.model.to(device)

    def get_pair_points(self):
        return [(15, 13), (16, 14),(13, 11),(12, 14),(11, 12),(11, 5),(12, 6),(5, 6),(5, 7),(6, 8),(7, 9),(8, 10),(0, 1),(0, 2),(1, 3),(2, 4)]

    def estimate_pose(self, video_path: str) -> list:
        """
        Estimate the pose of a video using YOLO pose estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            list: A list of tensors containing the keypoints for each frame.
        """

        confidence = self.config.get("confidence_threshold", 0.85)
        results = self.model.track(video_path, conf=confidence, stream=True, verbose=False)

        keypoints_tensor_list = []

        for result in results:
            if result.keypoints: # if no keypoints detected
                keypoints_tensor_list.append(result.keypoints.xy.int().tolist()) # convert floats to int and make it list so its serializable
            else: 
                keypoints_tensor_list.append([])

        # TODO: we migth refactor this to use an xarray with a frame, person, keypoint and dimension
        # or create a dedicated PoseData class that also stores the layout of the keypoints
        return keypoints_tensor_list