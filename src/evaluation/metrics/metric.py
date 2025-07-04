from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

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

    def _match_person_indices(self, pred_poses: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Match the predictions to the reference (e.g. ground truth or previous frame) for a single frame.
        This is useful for metrics that are order-dependent, such as PCK, acceleration or RMSE.
        It uses the Hungarian algorithm to find the best match between the predictions and the reference.

        Args:
            pred_poses: Predicted poses array of shape (M, K, 2) where M is number of persons
            reference: Reference poses array of shape (N, K, 2) where N is number of persons
            
        Returns:
            Sorted predictions array of shape (max(M,N), K, 2) where:
            - First N positions contain predictions matched to reference (or infinity if no match)
            - Remaining M-N positions (if M>N) contain unmatched predictions
        """
        M, K, _ = pred_poses.shape
        N, _, _ = reference.shape
        
        # If no predictions, return array of infinities of reference shape
        if M == 0:
            return np.full_like(reference, np.inf)
            
        # If no reference, return predictions as is
        if N == 0:
            return pred_poses
            
        # Calculate cost matrix based on Euclidian distance between each prediction and reference
        cost_matrix = np.zeros((M, N))

        mean_pred_poses = np.nanmean(pred_poses, axis=1)
        mean_ref_poses = np.nanmean(reference, axis=1)
        
        # Fill the valid part of cost matrix
        for i in range(M):
            for j in range(N):
                cost_matrix[i, j] = np.linalg.norm(mean_pred_poses[i] - mean_ref_poses[j])
        # Remove rows where all entries are nan, which might happen if the shape N or M is 
        # greater than the maximum number of persons in the reference or predictions.
        valid_rows = ~np.all(np.isnan(cost_matrix), axis=1)
        cost_matrix = cost_matrix[valid_rows]
                
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create output array that can hold all predictions
        max_persons = max(M, N)
        sorted_preds = np.full((max_persons, K, 2), np.inf)
        
        # First, fill the matched predictions in reference order
        used_pred_indices = set()
        for pred_idx, gt_idx in zip(row_ind, col_ind):
            if pred_idx < M and gt_idx < N:  # Only use valid matches
                sorted_preds[gt_idx] = pred_poses[pred_idx]
                used_pred_indices.add(pred_idx)
        
        # Then append any unused predictions at the end
        extra_idx = len(used_pred_indices)  # Start after used predictions
        for pred_idx in range(0, M):
            if pred_idx not in used_pred_indices:
                sorted_preds[extra_idx] = pred_poses[pred_idx]
                extra_idx += 1
                
        return np.array(sorted_preds)


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