import logging
from typing import Dict, List
from evaluation.metrics import Metric
from metric_result_class import MetricResult
from checkpointer import Checkpointer
from datasets import Dataset
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

class Evaluator:
    """Main evaluator class that orchestrates the evaluation process."""
    
    def __init__(self, metrics: List[Metric], checkpointer: Checkpointer, dataset: Dataset):
        self.metrics = {metric.name: metric for metric in metrics}
        self.checkpointer = checkpointer
        self.dataset = dataset

    def evaluate(
        self,
        model_list: List[str] = None,
    ) -> Dict[str, Dict[str, Dict[str, MetricResult]]]:
        """
        Run evaluation for all metrics on all models and videos.
        
        Args:
            models_video_pose_results: Dictionary mapping model names to video names and `VideoPoseResult` objects.
            gt_video_pose_results: Optional dictionary mapping video names to ground truth `VideoPoseResult` objects.
            
        Returns:
            Dictionary mapping metric names to models to video names to `MetricResult` objects.
        """
        results = {}
        max_workers = max(mp.cpu_count() - 1, 1)  # Use all available CPU cores for parallel processing

        for metric_name, metric in self.metrics.items():
            print(f"Computing metric: {metric_name}")
            model_results_dict = {}
            
            for model_name in model_list:
                video_metric_results = {}

                progress = tqdm.tqdm(total=len(self.dataset), desc=f"Evaluating {metric_name} for model {model_name}", unit="video")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_video = {
                        executor.submit(self.evaluate_video, video.get_filename(), model_name, metric): video
                        for video in self.dataset
                    }
                    for future in as_completed(future_to_video):
                        video_name = future_to_video[future].get_filename()
                        result = future.result()
                        if result is not None:
                            video_metric_results[video_name] = result
                            self.checkpointer.save_metric_result(metric_name, model_name, video_name, result)
                        progress.update(1)
                model_results_dict[model_name] = video_metric_results  
            
            results[metric_name] = model_results_dict
            
        return results

    def evaluate_video(self, video_name: str, model_name: str, metric: Metric) -> Dict[str, MetricResult]:
        video_pose_result = None
        if self.checkpointer.exists_evaluation_result(metric.name, model_name, video_name):
            print(f'Skipping evaluation for {video_name} using {model_name} for metric {metric.name} as results already exist.')
            return self.checkpointer.load_evaluation_result(metric.name, model_name, video_name)
        
        if not self.checkpointer.exists(model_name, video_name):
            print(f"No pose results found for video {video_name} using estimator {model_name}. Skipping.")
            logging.error(f"No pose results found for video {video_name} using estimator {model_name}. Skipping Evaluation for this video.")
            return None
        video_pose_result = self.checkpointer.load_pose_result(model_name, video_name)

        gt_pose_result = self.dataset.get_single_pose_result(video_name) # Can be None
        result = metric.compute(video_pose_result, gt_pose_result, model_name)
        return result