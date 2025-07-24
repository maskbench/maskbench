import time
from .pose_result import VideoPoseResult
from typing import Dict, List
from checkpointer import Checkpointer

class InferenceEngine:
    def __init__(self, dataset: dict, pose_estimators: list, checkpointer: Checkpointer):
        self.dataset = dataset
        self.pose_estimators = pose_estimators
        self.estimator_point_pairs = dict()
        self.checkpointer = checkpointer

    def estimate_pose_keypoints(self) -> Dict[str, Dict[str, List[VideoPoseResult]]]:
        results = {}
        if self.checkpointer.load_checkpoint:
            print(f"Loading results from checkpoint {self.checkpointer.checkpoint_dir}")
            results = self.checkpointer.load_pose_results()
            
        for estimator in self.pose_estimators:
            if estimator.name not in results:
                results[estimator.name] = {} 
            
            for video in self.dataset:
                if video.get_filename() in results[estimator.name]:
                    continue 

                print(f"Running estimator '{estimator.name}' on video {video.path}")
                start_time = time.time()
                try:
                    video_pose_result = estimator.estimate_pose(video.path)
                    results[estimator.name][video.get_filename()] = video_pose_result
                    self.checkpointer.save_video_pose_result(video_pose_result, estimator.name)
                    self.checkpointer.save_inference_time(estimator.name, video.get_filename(), time.time() - start_time)
                except Exception as e:
                    raise Exception(f"Faced Exception: {e} on Video: {video.get_filename()} with Estimator: {estimator.name}")


        return results 