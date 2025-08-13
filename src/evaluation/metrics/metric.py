from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import numpy as np
import numpy.ma as ma
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

    def _match_person_indices(self, poses_to_match: ma.MaskedArray, reference: ma.MaskedArray) -> ma.MaskedArray:
        """
        Match the predictions to the reference (e.g. ground truth or previous frame) for a single frame.
        This is useful for metrics that are order-dependent, such as PCK, acceleration or RMSE.
        It uses the Hungarian algorithm to find the best match between the predictions and the reference.
        If there are no predictions, it returns an array of infinities of reference shape. Infinities are used instead of nans
        to have an infinitely large mean center of a pose, which will not be matched to any reference (for example in a previous frame for kinematic metrics).

        Args:
            poses_to_match: Predicted poses array of shape (M, K, 2) where M is number of persons
            reference: Reference poses array of shape (N, K, 2) where N is number of persons
            
        Returns:
            Sorted predictions array of shape (max(M,N), K, 2) where:
            - First N positions contain predictions matched to reference (or infinity if no match)
            - Remaining M-N positions (if M>N) contain unmatched predictions
        """
        M, K, _ = poses_to_match.shape
        N, _, _ = reference.shape
        
        # If no predictions, return array of infinities of reference shape
        if M == 0:
            return np.full_like(reference, np.inf)
            
        # If no reference, return predictions as is
        if N == 0:
            return poses_to_match
            

        # Calculate mean positions
        mean_poses_to_match = np.nanmean(poses_to_match, axis=1)
        mean_ref_poses = np.nanmean(reference, axis=1)

        valid_M = M
        valid_N = N
        
        # Limit the cost matrix only to the valid persons (i.e. where the person is not completely masked)
        # Therefore, we need to create a mapping of the original person indices to the valid person indices
        poses_to_match_index_mapping = []
        reference_index_mapping = []
        is_poses_to_match_masked_array = isinstance(poses_to_match, ma.MaskedArray)
        is_reference_masked_array = isinstance(reference, ma.MaskedArray)
        for i in range(M):
            if is_poses_to_match_masked_array and poses_to_match.mask.all(axis=(1,2))[i]: # Reduce the number of valid poses to match (if the person is completely masked)
                valid_M -= 1
            else:
                poses_to_match_index_mapping.append(i)
        for i in range(N):
            if is_reference_masked_array and reference.mask.all(axis=(1,2))[i]: # Reduce the number of valid references (if the person is completely masked)
                valid_N -= 1
            else:
                reference_index_mapping.append(i)

        # Calculate cost matrix based on Euclidian distance between each prediction (valid_M) in the rows and references (valid_N) in the columns
        cost_matrix = np.zeros((valid_M, valid_N))
        for i in range(valid_M):
            for j in range(valid_N):
                pos_to_match_idx = poses_to_match_index_mapping[i]
                ref_idx = reference_index_mapping[j]
                cost_matrix[i, j] = np.linalg.norm(mean_poses_to_match[pos_to_match_idx] - mean_ref_poses[ref_idx])
        # Remove rows where all entries are nan, which might happen if the shape N or M is 
        # greater than the maximum number of persons in the reference or predictions.
        valid_rows = ~np.all(np.isnan(cost_matrix), axis=1)
        cost_matrix = cost_matrix[valid_rows]
                
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mapped_row_ind = [poses_to_match_index_mapping[i] for i in row_ind]
        mapped_col_ind = [reference_index_mapping[i] for i in col_ind]
        
        # Create output array that can hold all predictions
        max_persons = max(M, N)
        sorted_poses_to_match = np.full((max_persons, K, 2), np.inf)
        
        # First, fill the matched predictions in reference order and their dedicated mapped index
        used_pred_indices = set()
        for pred_idx, gt_idx in zip(mapped_row_ind, mapped_col_ind):
            if pred_idx < M and gt_idx < N:  # Only use valid matches
                sorted_poses_to_match[gt_idx] = poses_to_match[pred_idx]
                used_pred_indices.add(pred_idx)
        
        # Then append any unused predictions at the end (i.e. additional persons)
        if valid_M > valid_N:
            extra_idx = len(used_pred_indices)  # Start after used predictions
            for pred_idx in range(0, M):
                if pred_idx not in used_pred_indices:
                    sorted_poses_to_match[extra_idx] = poses_to_match[pred_idx]
                    extra_idx += 1
                
        # Create masked array where persons that are all 0 or inf are masked
        masked_poses = ma.array(sorted_poses_to_match)
        person_mask = (
            (masked_poses == 0).all(axis=(1,2)) | 
            (np.isinf(masked_poses)).all(axis=(1,2))
        )
        masked_poses[person_mask] = ma.masked
        return masked_poses


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