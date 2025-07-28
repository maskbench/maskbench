import numpy as np
import numpy.ma as ma
from typing import Dict, Optional, Any

from inference.pose_result import VideoPoseResult
from .metric import Metric
from .metric_result import COORDINATE_AXIS, FRAME_AXIS, KEYPOINT_AXIS, PERSON_AXIS, MetricResult
from .acceleration import AccelerationMetric


class JerkMetric(Metric):
    """
    Jerk metric.
    
    Required config parameters:
        - time_unit: str, either "frame" or "second" - specifies whether to compute jerk 
          in pixels/frame³ or pixels/second³. Defaults to "frame".
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="Jerk", config=config)
        self.acceleration_metric = AccelerationMetric(config)
        time_unit = config.get("time_unit", "frame") if config else "frame"
        if time_unit not in ["second", "frame"]:
            raise ValueError("time_unit must be either 'second' or 'frame'")
        self.time_unit = time_unit
    
    def compute(
        self,
        video_result: VideoPoseResult,
        gt_video_result: Optional[VideoPoseResult] = None,
        model_name: Optional[str] = None
    ) -> MetricResult:
        """
        Compute the jerk metric for a video result.
        Args:
            video_result: VideoPoseResult object containing the predicted poses.
            gt_video_result: This is not used for this metric.
            model_name: Name of the model being evaluated.
        Returns:
            MetricResult object containing the jerk metric values for each frame, person and keypoint.
            The jerk is calculated as the difference between consecutive acceleration values.
            For a given VideoPoseResult with T frames, the MetricResult will have T-3 frames.
            Every time a keypoint is missing in one of four consecutive frames, the jerk is NaN.
            If the number of frames is less than 4, the MetricResult will have 1 frame with NaN values.
            
            The time_unit parameter in config controls the calculation:
            - time_unit="second": jerk is computed per second (pixels/second³) by dividing by the time delta between frames
            - time_unit="frame": jerk is computed per frame (pixels/frame³)
        """
        pred_poses = video_result.to_numpy_ma()  # shape: (frames, persons, keypoints, 2)

        if pred_poses.shape[0] <= 3:
            print("Warning: Jerk metric requires at least 4 frames to compute. Returning empty MetricResult.")
            return MetricResult(
                values=np.nan * np.ones((1, pred_poses.shape[1], pred_poses.shape[2], 2)),
                axis_names=[FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS, COORDINATE_AXIS],
                metric_name=self.name,
                video_name=video_result.video_name,
                model_name=model_name,
            )

        acceleration_result = self.acceleration_metric.compute(video_result, gt_video_result, model_name)
        
        jerk = ma.diff(acceleration_result.values, axis=0)  # shape: (frames-3, persons, keypoints, 2)
        
        if self.time_unit == "second":
            fps = video_result.fps
            timedelta = 1 / fps
            jerk = jerk / timedelta
            
        jerk.data[jerk.mask] = np.nan

        return MetricResult(
            values=jerk,
            axis_names=[FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS, COORDINATE_AXIS],
            metric_name=self.name,
            video_name=video_result.video_name,
            model_name=model_name,
            unit=f"pixels/{self.time_unit}³"
        )