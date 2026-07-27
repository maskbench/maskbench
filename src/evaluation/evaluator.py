import logging
from typing import Dict, List
from evaluation.metrics import Metric
from metric_result_class import MetricResult
from checkpointer import Checkpointer
from datasets import Dataset


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
        
        for metric_name, metric in self.metrics.items():
            print(f"Computing metric: {metric_name}")
            model_results_dict = {}
            
            for model_name in model_list:
                video_metric_results = {}

                for video in self.dataset:
                    video_name = video.get_filename()
                    video_pose_result = None
                    if not self.checkpointer.exists(model_name, video_name):
                        print(f"No pose results found for video {video_name} using estimator {model_name}. Skipping.")
                        logging.error(f"No pose results found for video {video_name} using estimator {model_name}. Skipping Evaluation for this video.")
                        continue
                    video_pose_result = self.checkpointer.load_pose_result(model_name, video_name)

                    gt_pose_result = self.dataset.get_single_pose_result(video_name) # Can be None
                    result = metric.compute(video_pose_result, gt_pose_result, model_name)
                    if result is not None:
                        video_metric_results[video_name] = result

                model_results_dict[model_name] = video_metric_results  
            
            results[metric_name] = model_results_dict
            
        return results