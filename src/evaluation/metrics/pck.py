import numpy as np
import numpy.ma as ma
from typing import Dict, Optional, Any

from .metric import Metric
from .metric_result import FRAME_AXIS, MetricResult
from inference.pose_result import VideoPoseResult


class PCKMetric(Metric):
    """
    PCK (Percentage of Correct Keypoints) metric.
    Args:
        config: Configuration for the PCK metric. It must contain the following fields:
            - threshold: The threshold for the PCK metric.
            - normalize_by: The normalization strategy to use. Can be "bbox", "head" or "torso". Default is "bbox".
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="PCK", config=config)

        if config is None:
            raise ValueError("Config is required for PCK computation")
        if config["normalize_by"] not in ["bbox", "head", "torso"]:
            raise ValueError("Invalid normalization strategy. Must be one of 'bbox', 'head' or 'torso'")
        if config["normalize_by"] == "head" or config["normalize_by"] == "torso":
            raise NotImplementedError("Head and torso normalization is not implemented yet.")

        self.normalize_by = config["normalize_by"]
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
        if gt_video_result is None:
            raise ValueError("Ground truth video result is required for PCK computation")
            
        pred_poses = video_result.to_numpy_ma()  # shape: (frames, persons, keypoints, 2)
        gt_poses = gt_video_result.to_numpy_ma()  # shape: (frames, persons, keypoints, 2)

        values = []
        for frame_idx in range(pred_poses.shape[0]):
            pred_poses_frame = pred_poses[frame_idx] # shape: (M, K, 2)
            gt_poses_frame = gt_poses[frame_idx] # shape: (N, K, 2)

            pred_poses_frame = self._sort_predictions_by_ground_truth(pred_poses_frame, gt_poses_frame)
            
            norm_factors = None # shape: (N,)
            if self.normalize_by == "bbox":
                norm_factors = self.calculate_bbox_sizes_for_persons_in_frame(gt_poses_frame)
            else:
                raise NotImplementedError("Normalization by head or torso is not implemented yet.")
            norm_factors = np.array([norm_factors, norm_factors]).reshape(gt_poses_frame.shape[0], 2)

            distances = self._calculate_distances_for_frame(pred_poses_frame, gt_poses_frame, norm_factors)
            percentage_correct = (distances < self.threshold).sum() / distances.size
            values.append(percentage_correct)

        return MetricResult(
            values=np.array(values),
            axis_names=[FRAME_AXIS],
            metric_name=self.name,
            video_name=video_result.video_name,
            model_name=model_name,
        )

    def _calculate_distances_for_frame(self, pred_poses: ma.MaskedArray, gt_poses: ma.MaskedArray, norm_factors: np.ndarray) -> ma.MaskedArray:
        """
        Calculate PCK for a single frame.
        Slightly modified version of the original function from mmpose.
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/evaluation/functional/keypoint_eval.py#L10
        
        Args:
            pred_poses: Predicted poses array of shape (M, K, 2) where M is number of persons
                        and each pose has K keypoints with x,y coordinates. Note that M can be different from N.
            gt_poses: Ground truth poses array of shape (N, K, 2) where N is number of persons
                        and each pose has K keypoints with x,y coordinates.
            norm_factors: Normalization factors for each person of shape (N,).
        """
        N, K, _ = gt_poses.shape
        M, _, _ = pred_poses.shape

        distances = np.full((N, K), -1, dtype=np.float32)
        norm_factors[np.where(norm_factors <= 0)] = 1e6
        
        # Initialize diff array with infinity for all ground truth persons
        diff = np.full_like(gt_poses, np.inf, dtype=np.float32)
        
        # Calculate diff only for persons that exist in predictions and ground truth
        n_pred = min(M, N)
        if n_pred > 0:
            diff[:n_pred] = gt_poses[:n_pred] - pred_poses[:n_pred]

        diff_norm = diff / norm_factors[:, None, :]
        distances = np.linalg.norm(diff_norm, axis=-1)
        return distances


    def calculate_bbox_sizes_for_persons_in_frame(self, gt_poses: np.ndarray) -> np.ndarray:
        """Calculate bounding box sizes for each person in a frame.
        
        Args:
            gt_poses: Ground truth poses array of shape (M, K, 2) where M is number of persons
                     and each pose has K keypoints with x,y coordinates.
                     
        Returns:
            np.ndarray: Array of shape (M,) containing the bounding box size for each person,
                       calculated as the maximum of width and height of the bounding box.
        """
        min_coords = gt_poses.min(axis=1)  # Shape: (M, 2)
        max_coords = gt_poses.max(axis=1)  # Shape: (M, 2)
        
        widths = max_coords[..., 0] - min_coords[..., 0]  # Shape: (M,)
        heights = max_coords[..., 1] - min_coords[..., 1]  # Shape: (M,)
        
        return np.maximum(widths, heights)
        