import numpy as np
import numpy.ma as ma
from typing import Dict, Optional, Any

from inference.pose_result import VideoPoseResult
from .metric import Metric
from .metric_result import FRAME_AXIS, MetricResult
from .euclidean_distance import EuclideanDistanceMetric


class PCKMetric(EuclideanDistanceMetric):
    """
    PCK (Percentage of Correct Keypoints) metric.
    Args:
        config: Configuration for the PCK metric. It must contain the following fields:
            - threshold: The threshold for the PCK metric.
            - normalize_by: The normalization strategy to use. Can be "bbox", "head" or "torso". Default is "bbox".
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="PCK", config=config)

        if config["threshold"] is None:
            raise ValueError("Threshold is required for PCK computation.")

        self.threshold = config["threshold"]
    
    def compute(
        self,
        video_result: VideoPoseResult,
        gt_video_result: Optional[VideoPoseResult] = None,
        model_name: Optional[str] = None
    ) -> MetricResult:
        """
        Compute the PCK metric for a video result.
        Args:
            video_result: VideoPoseResult object containing the predicted poses.
            gt_video_result: VideoPoseResult object containing the ground truth poses.
            model_name: Name of the model being evaluated.
        Returns:
            MetricResult object containing the PCK metric values for each frame for the video.
        """
        euclidean_distance = super().compute(video_result, gt_video_result, model_name)
        values = (euclidean_distance.values < self.threshold).sum(axis=(1, 2)) / euclidean_distance.values[0].size

        return MetricResult(
            values=values,
            axis_names=[FRAME_AXIS],
            metric_name=self.name,
            video_name=video_result.video_name,
            model_name=model_name,
        )
