import numpy as np
import numpy.ma as ma
import pytest

from evaluation.metrics.mpjpe import MPJPEMetric
from tests.utils import create_test_pose_result


class TestMPJPEMetric:
    """Test suite for the MPJPEMetric class."""
    
    @pytest.fixture
    def metric(self):
        """Fixture providing a fresh MPJPEMetric instance for each test."""
        return MPJPEMetric()
    
    def test_basic_computation(self, metric):
        """Test basic MPJPE computation with valid inputs."""
        pred_poses = [
            [[[1.0, 1.0], [2.0, 2.0]],  # frame 0, person 1, 2 keypoints
             [[3.0, 3.0], [4.0, 4.0]]], # frame 0, person 2, 2 keypoints
            [[[5.0, 5.0], [6.0, 6.0]]]  # frame 1, person 1, 2 keypoints
        ]
        gt_poses = [
            [[[0.0, 0.0], [2.0, 2.0]],  # frame 0, person 1, 2 keypoints
             [[3.0, 3.0], [4.0, 4.0]]], # frame 0, person 2, 2 keypoints
            [[[5.0, 5.0], [7.0, 7.0]]]  # frame 1, person 1, 2 keypoints
        ]
        
        pred_result = create_test_pose_result(pred_poses, "test_video")
        gt_result = create_test_pose_result(gt_poses, "ground_truth")
        print(pred_result.to_numpy_ma().shape)
        print(gt_result.to_numpy_ma().shape)
        
        result = metric.compute(pred_result, gt_result, model_name="test_model")
        
        expected_errors = ma.array([
            [[np.sqrt(2), 0.0],    # frame 0, person 1, 2 keypoints
             [0.0, 0.0]],          # frame 0, person 2, 2 keypoints
            [[0.0, np.sqrt(2)],    # frame 1, person 1, 2 keypoints
             [0.0, 0.0]],          # frame 1, person 2, 2 keypoints (not detected, invalid, masked)
        ], mask=[
            [[False, False],       # frame 0, person 1, 2 keypoints
             [False, False]],      # frame 0, person 2, 2 keypoints
            [[False, False],       # frame 1, person 1, 2 keypoints
             [True, True]],        # frame 1, person 2, 2 keypoints (masked)
        ])
        
        np.testing.assert_array_almost_equal(result.values, expected_errors)
    
    def test_missing_gt(self, metric):
        """Test that MPJPE raises error when ground truth is missing."""
        pred_poses = np.array([[[[1.0, 1.0], [2.0, 2.0]]]])
        pred_result = create_test_pose_result(pred_poses)
        
        with pytest.raises(ValueError, match="Ground truth video result is required"):
            metric.compute(pred_result, None)
    
    def test_shape_mismatch(self, metric):
        """Test that MPJPE raises error when number of detected persons don't match."""
        pred_poses = np.array([
            [[[1.0, 1.0], [2.0, 2.0]],  # person 1
             [[3.0, 3.0], [4.0, 4.0]]]  # person 2
        ])  # shape: (1 frame, 2 persons, 2 keypoints, 2 coords)
        
        gt_poses = np.array([
            [[[0.0, 0.0], [2.0, 2.0]]]  # only 1 person
        ])  # shape: (1 frame, 1 person, 2 keypoints, 2 coords)
        
        pred_result = create_test_pose_result(pred_poses)
        gt_result = create_test_pose_result(gt_poses)
        
        with pytest.raises(ValueError, match="Shape mismatch.*number of detected persons is different"):
            metric.compute(pred_result, gt_result)
    
    def test_different_masks(self, metric):
        """Test MPJPE computation with different masks between prediction and ground truth."""
        pred_poses = [
            [[]],                        # frame 1: no person detected
            [[[5.0, 5.0], [6.0, 6.0]]]   # frame 2: person 1 (valid)
        ]
        gt_poses = [
            [[[0.0, 0.0], [2.0, 2.0]]],  # frame 1: person 1 (valid)
            [[[5.0, 5.0], [7.0, 7.0]]]   # frame 2: person 1 (valid)
        ]
        
        # Create prediction result with frame 1 masked
        pred_result = create_test_pose_result(pred_poses, "test_video")
        print(pred_result.to_numpy_ma())
        gt_result = create_test_pose_result(gt_poses, "ground_truth")
        
        with pytest.raises(ValueError, match="Shape mismatch.*number of detected persons is different"):
            metric.compute(pred_result, gt_result)
        
    