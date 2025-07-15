
from typing import Dict
import numpy as np
import numpy.ma as ma

from .metrics import MetricResult

# This is the value that will be used to fill the distance matrix for keypoints
# that are not visible in the prediction, but are visible in the ground truth.
DISTANCE_FILL_VALUE = 1.0

def calculate_bbox_sizes_for_persons_in_frame(gt_poses: np.ndarray) -> np.ndarray:
    """Calculate bounding box sizes for each person in a frame.
    
    Args:
        gt_poses: Ground truth poses array of shape (M, K, 2) where M is number of persons
                    and each pose has K keypoints with x,y coordinates.
                    
    Returns:
        np.ndarray: Array of shape (M,) containing the bounding box size for each person,
                    calculated as the maximum of width and height of the bounding box.
    """
    # Create mask for keypoints that are (0,0) to ignore them in the bounding box calculation
    zero_mask = np.all(gt_poses == 0, axis=-1)  # Shape: (M, K)
    
    # Get min/max coords ignoring (0,0) keypoints
    masked_gt_poses = ma.array(gt_poses, mask=np.dstack([zero_mask, zero_mask]))  # Shape: (M, K, 2)
    min_coords = masked_gt_poses.min(axis=1)  # Shape: (M, 2) 
    max_coords = masked_gt_poses.max(axis=1)  # Shape: (M, 2)
    
    widths = max_coords[..., 0] - min_coords[..., 0]  # Shape: (M,)
    heights = max_coords[..., 1] - min_coords[..., 1]  # Shape: (M,)
    
    return np.maximum(widths, heights)


def aggregate_results_over_all_videos(metric_results: Dict[str, Dict[str, Dict[str, MetricResult]]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate the results by averaging over the different datasets.
    Returns:
        Dict[str, Dict[str, float]]: Aggregated results for each metric and pose estimator.
    """
    aggregated_results = {}
    for metric_name, pose_estimator_results in metric_results.items():
        for pose_estimator_name, video_results in pose_estimator_results.items():
            aggregated_results[metric_name] = {}
            
            aggregated_video_results = []
            for video_name, result in video_results.items():
                aggregated_video_results.append(result.aggregate_all())
            aggregated_results[metric_name][pose_estimator_name] = np.round(np.mean(aggregated_video_results), decimals=2)

    return aggregated_results

