import time
from .pose_result import VideoPoseResult
from typing import Dict, List
from checkpointer import Checkpointer
import logging

class InferenceEngine:
    """Class responsible for running the pose estimators on the videos and saving the results in the `poses` folder."""
    
    def __init__(self, dataset: dict, pose_estimators: list, checkpointer: Checkpointer):
        self.dataset = dataset
        self.pose_estimators = pose_estimators
        self.estimator_point_pairs = dict()
        self.checkpointer = checkpointer

    def estimate_pose_keypoints(self) -> Dict[str, Dict[str, VideoPoseResult]]:
        """
        Run the pose estimators on the videos and save the results in the `poses` folder.
        If a checkpoint name is provided in the configuration file, the inference engine will load the results from the checkpoint and skip the inference for the videos that already have results.
        This allows to resume the inference process in case it fails or to skip the inference entirely and only evaluate the metrics.

        Returns:
            Dictionary mapping pose estimator names to video names and `VideoPoseResult` objects.
        """
        results = {}
        if self.checkpointer.load_checkpoint:
            print(f"Loading results from checkpoint {self.checkpointer.checkpoint_dir}")
            results = self.checkpointer.load_pose_results(pose_estimator_names=list(map(lambda x: x.name, self.pose_estimators)))
            
        for estimator in self.pose_estimators:
            if estimator.name not in results: # if the estimator has no results yet
                results[estimator.name] = {} 
            
            for video in self.dataset:
                if video.get_filename() in results[estimator.name]:
                    print(f"Skipping already processed video {video.get_filename()} for estimator {estimator.name}")
                    continue 

                print(f"Running estimator '{estimator.name}' on video {video.path}")
                start_time = time.time()
                try:
                    video_pose_result = estimator.estimate_pose(video.path)
                    results[estimator.name][video.get_filename()] = video_pose_result
                    self.checkpointer.save_video_pose_result(video_pose_result, estimator.name)
                    self.checkpointer.save_inference_time(estimator.name, video.get_filename(), time.time() - start_time)
                except Exception as e:
                    print(f"Error processing video {video.get_filename()} with estimator {estimator.name}: {e}")
                    logging.error(f"Faced Exception: {e} on Video: {video.get_filename()} with Estimator: {estimator.name}")
                    continue

        return results 