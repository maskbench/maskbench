import os
import time
from typing import Dict, List

from .pose_result import VideoPoseResult

class InferenceEngine():
    def __init__(self, dataset: dict, pose_estimators: list):
        self.dataset = dataset
        self.pose_estimators = pose_estimators
        self.base_output_path = "/output"
        self.estimator_point_pairs = dict()

    def estimate_pose_keypoints(self) -> Dict[str, List[VideoPoseResult]]:
        results = {}
        for estimator in self.pose_estimators:
            results[estimator.name] = []

            for video in self.dataset:
                print(f"Running estimator '{estimator.name}' on video {video.path}")

                start_time = time.time()
                video_pose_result = estimator.estimate_pose(video.path) 
                results[estimator.name].append(video_pose_result)
                end_time = time.time()

                print(f"Inference time: '{estimator.name}': {end_time - start_time}")

        return results



            
 

