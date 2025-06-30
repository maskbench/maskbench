from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from inference.pose_result import VideoPoseResult
from evaluation.metrics.metric_result import MetricResult, FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS


class Metric(ABC):
    """Base class for all metrics in MaskBench."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a metric.
        
        Args:
            name: Unique name of the metric
            config: Optional configuration dictionary for the metric
        """
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    def compute(
        self,
        video_result: VideoPoseResult,
        gt_video_result: Optional[VideoPoseResult] = None,
        model_name: Optional[str] = None
    ) -> MetricResult:
        """
        Compute the metric for a video.
        
        Args:
            video_result: Pose estimation results for the video
            gt_video_result: Optional ground truth pose results
            model_name: Name of the model being evaluated
            
        Returns:
            MetricResult containing the metric values for the video
        """
        pass


class DummyMetric(Metric):
    """
    A simple metric that just returns the raw pose data.
    Useful for testing and as an example of how to implement a metric.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="DummyMetric", config=config)
    
    def compute(
        self,
        video_result: VideoPoseResult,
        gt_video_result: Optional[VideoPoseResult] = None,
        model_name: Optional[str] = None
    ) -> MetricResult:
        """
        Simply convert the pose data to a MetricResult without any computation.
        """
        # Convert pose data to masked array and take only x coordinates
        values = video_result.to_numpy_ma()[:, :, :, 0]

        return MetricResult(
            values=values,
            axis_names=[FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS],
            metric_name=self.name,
            video_name=video_result.video_name,
            model_name=model_name,
        ) 