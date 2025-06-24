
import numpy as np
import numpy.ma as ma

DISTANCE_FILL_VALUE = 2.0

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

