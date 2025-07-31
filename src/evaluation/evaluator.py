from typing import Dict, List
from evaluation.metrics import MetricResult, Metric
from inference.pose_result import VideoPoseResult


class Evaluator:
    """Main evaluator class that orchestrates the evaluation process."""
    
    def __init__(self, metrics: List[Metric]):
        self.metrics = {metric.name: metric for metric in metrics}
    
    def evaluate(
        self,
        models_video_pose_results: Dict[str, Dict[str, VideoPoseResult]],
        gt_video_pose_results: Dict[str, VideoPoseResult] = None
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
            for model_name, video_pose_results in models_video_pose_results.items():
                
                if not gt_video_pose_results:
                    gt_video_pose_results = {video_name: None for video_name in video_pose_results.keys()}
                
                video_metric_results = {}
                for video_name, video_result in video_pose_results.items():
                    gt_result = gt_video_pose_results[video_name]
                    video_metric_results[video_name] = metric.compute(video_result, gt_result, model_name)

                model_results_dict[model_name] = video_metric_results
            
            results[metric_name] = model_results_dict
            
        return results