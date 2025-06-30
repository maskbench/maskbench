import numpy as np
import numpy.ma as ma
from typing import Dict, Optional, Any


from evaluation.utils import DISTANCE_FILL_VALUE, calculate_bbox_sizes_for_persons_in_frame
from inference.pose_result import VideoPoseResult
from .metric import Metric
from .metric_result import FRAME_AXIS, KEYPOINT_AXIS, PERSON_AXIS, MetricResult


class EuclideanDistanceMetric(Metric):
    """
    Euclidean Distance metric.
    Args:
        config: Configuration for the Euclidean Distance metric. It must contain the following fields:
            - threshold: The threshold for the metric.
            - normalize_by: The normalization strategy to use. Can be "bbox", "head" or "torso". Default is "bbox".
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="EuclideanDistance", config=config)

        if config is None:
            raise ValueError("Config is required for Euclidean Distance computation")
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
        Compute the Euclidean distance metric for a video result.
        Args:
            video_result: VideoPoseResult object containing the predicted poses.
            gt_video_result: VideoPoseResult object containing the ground truth poses.
            model_name: Name of the model being evaluated.
        Returns:
            MetricResult object containing the Euclidean distance metric values for each frame, person and keypoint.
            All distances are normalized by the person's bounding box, head or torso size.
            For missing keypoints:
                1) If the ground truth is (0,0), the distance is set to np.nan

                2) If the ground truth is not (0,0) and the prediction is (0,0) or if a person is entirely undetected, 
                the distance is set to a predetermined fill value in order to not affect the aggregation calculation too much.
        """
        if gt_video_result is None:
            raise ValueError("Ground truth video result is required for Euclidean distance computation")
            
        pred_poses = video_result.to_numpy_ma()  # shape: (frames, persons, keypoints, 2)
        gt_poses = gt_video_result.to_numpy_ma()  # shape: (frames, persons, keypoints, 2)

        frame_values = []
        for frame_idx in range(pred_poses.shape[0]):
            gt_poses_frame = gt_poses[frame_idx].data # shape: (N, K, 2), work on ndarrays without mask
            pred_poses_frame = pred_poses[frame_idx].data # shape: (M, K, 2), work on ndarrays without mask
            pred_poses_frame = self._match_person_indices(pred_poses_frame, gt_poses_frame)

            person_norm_factors = None # shape: (N,)
            if self.normalize_by == "bbox":
                person_norm_factors = calculate_bbox_sizes_for_persons_in_frame(gt_poses_frame)
            else:
                raise NotImplementedError("Normalization by head or torso is not implemented yet.")

            norm_factors = np.array([person_norm_factors, person_norm_factors]).reshape(gt_poses_frame.shape[0], 2)

            distances = self._calculate_euclidean_distances_for_frame(pred_poses_frame, gt_poses_frame, norm_factors)
            frame_values.append(distances)

        frame_values = np.array(frame_values)
        
        # Create mask for missing keypoints. Keypoints are excluded in these cases and values are set to np.nan
        # 1. Keypoints missing in both GT and predictions (both are (0,0))
        # 2. Keypoints missing in GT but present in predictions (GT is (0,0))
        gt_zeros = np.all(gt_poses == 0, axis=-1)  # Shape: (N, K)
        frame_values[gt_zeros] = np.nan
        frame_values = ma.array(np.array(frame_values), mask=gt_zeros) # where a person is undetected

        return MetricResult(
            values=frame_values,
            axis_names=[FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS],
            metric_name=self.name,
            video_name=video_result.video_name,
            model_name=model_name,
        )

    def _calculate_euclidean_distances_for_frame(self, pred_poses: np.ndarray, gt_poses: np.ndarray, norm_factors: np.ndarray) -> ma.MaskedArray:
        """
        Calculate Euclidean distances for a single frame.
        Slightly modified version of the original function from mmpose.
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/evaluation/functional/keypoint_eval.py#L10
        
        Args:
            pred_poses: Predicted poses array of shape (M, K, 2) where M is number of persons
                        and each pose has K keypoints with x,y coordinates. Note that M must be greater than or equal to N.
            gt_poses: Ground truth poses array of shape (N, K, 2) where N is number of persons
                        and each pose has K keypoints with x,y coordinates.
            norm_factors: Normalization factors for each person of shape (N, 2).
        Returns:
            Array of shape (N, K) where N is number of persons and K is number of keypoints.
        """
        N, K, _ = gt_poses.shape
        M, _, _ = pred_poses.shape
        if M < N:
            raise ValueError("Number of predicted persons must be greater than or equal to number of ground truth persons")

        norm_factors[np.where(norm_factors <= 0)] = 1e6
        # Calculate diff only for persons that exist in predictions and ground truth
        diff = gt_poses[:N] - pred_poses[:N]
        diff_norm = diff / norm_factors[:, None, :]
        distances = np.linalg.norm(diff_norm, axis=-1)
        
        # Cases where the prediction has missing data compared to the ground truth
        is_no_person_detected = np.all(np.all(np.isinf(pred_poses[:N]), axis=-1), axis=-1)  # Shape: (N,)
        has_prediction_zero_point = np.all(pred_poses[:N] == 0, axis=-1)  # Shape: (N, K)
        distances[is_no_person_detected] = DISTANCE_FILL_VALUE # where an entire person is undetected
        distances[has_prediction_zero_point] = DISTANCE_FILL_VALUE # where a person is detected but has a missing keypoint (like (0,0))

        return distances
