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

        if config.get("threshold") is None:
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
        Keypoints are considered correct if their distance to the ground truth is less than the threshold.
        Does not take into account invalid keypoints (i.e.) where ground truth is (0,0) but prediction is unequal (0,0).

        Args:
            video_result: VideoPoseResult object containing the predicted poses.
            gt_video_result: VideoPoseResult object containing the ground truth poses.
            model_name: Name of the model being evaluated.
        Returns:
            MetricResult object containing the PCK metric values for each frame for the video.
        """
        euclidean_distances = super().compute(video_result, gt_video_result, model_name).values
        valid_distances = ma.masked_array(euclidean_distances, mask=(euclidean_distances == np.nan))
        correct_keypoints = (valid_distances < self.threshold)
        num_valid_distances = (~valid_distances.mask).sum(axis=(1, 2))
        values = correct_keypoints.sum(axis=(1, 2)) / num_valid_distances

        return MetricResult(
            values=values,
            axis_names=[FRAME_AXIS],
            metric_name=self.name,
            video_name=video_result.video_name,
            model_name=model_name,
        )
