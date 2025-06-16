import numpy as np
import numpy.ma as ma
from typing import Dict, Optional, Any

from .metric import Metric
from .metric_result import MetricResult, FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS
from inference.pose_result import VideoPoseResult


class MPJPEMetric(Metric):
    """
    Mean Per Joint Position Error (MPJPE) metric.
    
    Computes the Euclidean distance between predicted and ground truth keypoints
    for each joint.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="MPJPE", config=config)
    
    def compute(
        self,
        video_result: VideoPoseResult,
        gt_video_result: Optional[VideoPoseResult] = None,
        model_name: Optional[str] = None
    ) -> MetricResult:
        """
        Compute MPJPE between predicted and ground truth poses.
        
        Args:
            video_result: Predicted pose results
            gt_video_result: Ground truth pose results
            model_name: Name of the model being evaluated
            
        Returns:
            MetricResult containing per-joint position errors with shape
            (num_frames, num_persons, num_keypoints)
        """
        if gt_video_result is None:
            raise ValueError("Ground truth video result is required for MPJPE computation")
            
        pred_poses = video_result.to_numpy_ma()  # shape: (frames, persons, keypoints, 2)
        gt_poses = gt_video_result.to_numpy_ma()  # shape: (frames, persons, keypoints, 2)
        
        if pred_poses.shape != gt_poses.shape or pred_poses.mask.any() != gt_poses.mask.any():
            raise ValueError(
                f"Shape mismatch between prediction ({pred_poses.shape}) "
                f"and ground truth ({gt_poses.shape}). Probably the number of detected persons is different."
                f"This is not supported yet."
            )
        
        # Compute Euclidean distance for each joint
        squared_diff = (pred_poses - gt_poses) ** 2
        distances = np.sqrt(squared_diff[:, :, :, 0] + squared_diff[:, :, :, 1])
        
        # Create masked array with same mask as input
        # We mask if either prediction or ground truth is masked
        mask = pred_poses.mask[:, :, :, 0] | gt_poses.mask[:, :, :, 0]
        values = ma.array(distances, mask=mask)
        
        return MetricResult(
            values=values,
            axis_names=[FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS],
            metric_name=self.name,
            video_name=video_result.video_name,
            model_name=model_name,
        )