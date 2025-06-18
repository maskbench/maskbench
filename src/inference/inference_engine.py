import time
from typing import Dict, List
from dataclasses import asdict
from .pose_result import VideoPoseResult
import json
import os

class InferenceEngine:
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
                try:
                    video_pose_result = estimator.estimate_pose(video.path)
                    results[estimator.name].append(video_pose_result)
                    
                    video_name, _ = os.path.splitext(video.path) # remove extension
                    json_folderpath = os.path.join(self.base_output_path, os.path.basename(video_name))
                    os.makedirs(json_folderpath, exist_ok=True)
                    
                    json_filepath = os.path.join(json_folderpath, f"{estimator.name}.json")
                    serialized_video_pose_result = {
                        "fps": video_pose_result.fps,
                        "frame_width": video_pose_result.frame_width,
                        "frame_height": video_pose_result.frame_height,
                        "frames": [asdict(frame) for frame in video_pose_result.frames]
                    }
                    
                    with open(json_filepath, "w+") as f:
                        json.dump(serialized_video_pose_result, f, indent=2)
            
                except Exception as e:
                    print(f"Faced Exception: {e} on Video: {video}")
                    
                end_time = time.time()

                print(f"Inference time: '{estimator.name}': {end_time - start_time}")

        return results
