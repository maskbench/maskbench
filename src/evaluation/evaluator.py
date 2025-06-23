from typing import Dict, List
from evaluation.metrics import MetricResult, Metric
from inference.pose_result import VideoPoseResult


class Evaluator:
    """Main evaluator class that orchestrates the evaluation process."""
    
    def __init__(self, metrics: List[Metric]):
        """
        Initialize the evaluator.
        
        Args:
            metrics: List of instantiated metric objects to use for evaluation
        """
        self.metrics = {metric.name: metric for metric in metrics}
    
    def evaluate(
        self,
        models_video_pose_results: Dict[str, List[VideoPoseResult]],
        gt_video_pose_results: List[VideoPoseResult] = None
    ) -> Dict[str, Dict[str, MetricResult]]:
        """
        Run evaluation for all metrics on all models and videos.
        
        Args:
            models_video_pose_results: Dictionary mapping model names to their video results
            gt_video_pose_results: Optional list of ground truth results
            
        Returns:
            Dictionary mapping metric names to models to MetricResults.
        """
        results = {}
        
        for metric_name, metric in self.metrics.items():
            print(f"Computing metric: {metric_name}")
            
            model_results_dict = {}
            for model_name, video_pose_results in models_video_pose_results.items():
                
                if not gt_video_pose_results:
                    gt_video_pose_results = [None] * len(video_pose_results)
                
                video_metric_results = {}
                for video_result, gt_result in zip(video_pose_results, gt_video_pose_results):
                    video_metric_results[video_result.video_name] = metric.compute(video_result, gt_result, model_name)

                model_results_dict[model_name] = video_metric_results
            
            results[metric_name] = model_results_dict
            
        return results