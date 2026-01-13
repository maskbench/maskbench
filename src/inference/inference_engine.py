import time
from .pose_result import VideoPoseResult
from typing import Dict, List
from checkpointer import Checkpointer
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

class InferenceEngine:
    """Class responsible for running the pose estimators on the videos and saving the results in the `poses` folder."""
    
    def __init__(self, dataset: dict, pose_estimators: list, checkpointer: Checkpointer):
        self.dataset = dataset
        self.pose_estimators = pose_estimators
        self.estimator_point_pairs = dict()
        self.checkpointer = checkpointer
        self.results = dict()
    
    def run_parallel_tasks(self, max_workers: int = None) -> Dict:
        num_estimator = len(self.pose_estimators)
        if max_workers is None:
            max_workers = num_estimator

        print('=' * 50)
        print(f"Running {num_estimator} pose estimators with max_workers={max_workers}")
        print(f"Using {mp.cpu_count()} CPU cores")
        print(f"Total videos to process: {len(self.dataset)}")
        print('=' * 50)

        if self.checkpointer.load_checkpoint:
            print(f"Loading results from checkpoint {self.checkpointer.checkpoint_dir}")
            self.results = self.checkpointer.load_pose_results(pose_estimator_names=list(map(lambda x: x.name, self.pose_estimators)))
        
        for estimator in self.pose_estimators: # if user adds a new model, initialize its results dict
            if estimator.name not in self.results:
                self.results[estimator.name] = {} 

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_estimator = {
                executor.submit(self.estimate_pose_keypoints, estimator): estimator
                for estimator in self.pose_estimators
            }
            for future in as_completed(future_to_estimator):
                estimator = future_to_estimator[future]
                try:
                    estimator_results = future.result()
                    self.results[estimator_results['estimator']] = estimator_results['results']
                except Exception as e:
                    print(f"Estimator {estimator.name} generated an exception: {e}")
        return self.results

    def estimate_pose_keypoints(self, estimator) -> Dict:
        """
        Run the pose estimators on the videos and save the results in the `poses` folder.
        If a checkpoint name is provided in the configuration file, the inference engine will load the results from the checkpoint and skip the inference for the videos that already have results.
        This allows to resume the inference process in case it fails or to skip the inference entirely and only evaluate the metrics.

        Returns:
            Dictionary mapping pose estimator names to video names and `VideoPoseResult` objects.
        """
           
        estimator_results = {}
            
        for video in self.dataset:
            if video.get_filename() in self.results[estimator.name]:
                continue # if results already exist, skip inference

            print(f"Running estimator '{estimator.name}' on video {video.path}")
            start_time = time.time()
            try:
                video_pose_result = estimator.estimate_pose(video.path)
                estimator_results[video.get_filename()] = video_pose_result
                self.checkpointer.save_video_pose_result(video_pose_result, estimator.name)
                self.checkpointer.save_inference_time(estimator.name, video.get_filename(), time.time() - start_time)
            except Exception as e:
                raise Exception(f"Faced Exception: {e} on Video: {video.get_filename()} with Estimator: {estimator.name}")

        return {'estimator': estimator.name, 'results': estimator_results}