from typing import Dict, Optional, Any


from inference.pose_result import VideoPoseResult
from .euclidean_distance import EuclideanDistanceMetric
from .metric import Metric
from .metric_result import MetricResult


class RMSEMetric(EuclideanDistanceMetric):
    """
    Root Mean Square Error metric.
    Args:
        config: Configuration for the RMSE metric. It must contain the following fields:
            - normalize_by: The normalization strategy to use. Can be "bbox", "head" or "torso". Default is "bbox".
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="RMSE", config=config)
    
    def compute(
        self,
        video_result: VideoPoseResult,
        gt_video_result: Optional[VideoPoseResult] = None,
        model_name: Optional[str] = None
    ) -> MetricResult:
        """
        Compute the RMSE metric for a video result.
        Args:
            video_result: VideoPoseResult object containing the predicted poses.
            gt_video_result: VideoPoseResult object containing the ground truth poses.
            model_name: Name of the model being evaluated.
        Returns:
            MetricResult object containing the RMSE metric values for each frame.
        """
        euclidean_distance_result = super().compute(video_result, gt_video_result, model_name)
        return euclidean_distance_result.aggregate(dims=["person", "keypoint"], method="rmse")