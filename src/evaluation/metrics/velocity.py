import numpy as np
import numpy.ma as ma
from typing import Dict, Optional, Any


from evaluation.utils import DISTANCE_FILL_VALUE, calculate_bbox_sizes_for_persons_in_frame
from inference.pose_result import VideoPoseResult
from .metric import Metric
from .metric_result import COORDINATE_AXIS, FRAME_AXIS, KEYPOINT_AXIS, PERSON_AXIS, MetricResult


class VelocityMetric(Metric):
    """
    Velocity metric.
    
    Config parameters:
        - time_unit: str, either "frame" or "second" - specifies whether to compute velocity 
          in pixels/frame or pixels/second. Defaults to "frame".
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="Velocity", config=config)
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
        Compute the velocity metric for a video result.
        Args:
            video_result: VideoPoseResult object containing the predicted poses.
            gt_video_result: This is not used for this metric.
            model_name: Name of the model being evaluated.
        Returns:
            MetricResult object containing the velocity metric values for each frame, person and keypoint.
            The velocity is calculated as the difference between the current frame's position and the previous frame's position.
            For a given VideoPoseResult with T frames, the MetricResult will have T-1 frames.
            Every time a keypoint is missing in one of two consecutive frames, the velocity is NaN.
            If the number of frames is less than 2, the MetricResult will have 1 frame with NaN values.
            
            The time_unit parameter in config controls the calculation:
            - time_unit="second": velocity is computed per second (pixels/second) by dividing by the time delta between frames
            - time_unit="frame": velocity is computed per frame (pixels/frame)
        """
        pred_poses = video_result.to_numpy_ma()  # shape: (frames, persons, keypoints, 2)

        if pred_poses.shape[0] <= 1:
            print("Warning: Velocity metric requires at least 2 frames to compute. Returning empty MetricResult.")
            return MetricResult(
                values=np.nan * np.ones((1, pred_poses.shape[1], pred_poses.shape[2], 2)),
                axis_names=[FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS, COORDINATE_AXIS],
                metric_name=self.name,
                video_name=video_result.video_name,
                model_name=model_name,
            )

        for frame_idx in range(pred_poses.shape[0] - 1):
            current_frame_poses = pred_poses[frame_idx]
            next_frame_poses = pred_poses[frame_idx + 1]
            sorted_next_frame_poses = self._match_person_indices(next_frame_poses, current_frame_poses)
            sorted_next_frame_poses.data[sorted_next_frame_poses.mask] = np.nan
            pred_poses[frame_idx + 1] = sorted_next_frame_poses

        # Mask all (0, 0) keypoints in addition to the existing mask
        zero_points_mask = np.repeat((pred_poses == 0).all(axis=-1)[..., np.newaxis], 2, axis=-1)
        pred_poses.mask |= zero_points_mask

        velocity = ma.diff(pred_poses, axis=0)  # shape: (frames-1, persons, keypoints, 2)
        
        if self.time_unit == "second":
            fps = video_result.fps
            timedelta = 1 / fps
            velocity = velocity / timedelta

        velocity.data[velocity.mask] = np.nan

        return MetricResult(
            values=velocity,
            axis_names=[FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS, COORDINATE_AXIS],
            metric_name=self.name,
            video_name=video_result.video_name,
            model_name=model_name,
            unit=f"pixels/{self.time_unit}"
        )