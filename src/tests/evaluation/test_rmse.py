import unittest
import numpy as np
import numpy.ma as ma
import pytest

from evaluation.metrics import EuclideanDistanceMetric
from evaluation.utils import DISTANCE_FILL_VALUE
from tests.utils import create_example_video_pose_result

def compute_euclidean_distance_metric(gt_data, pred_data, threshold=0.1, normalize_by="bbox"):
    """Compute the Euclidean Distance metric."""
    gt_video_result = create_example_video_pose_result(gt_data, "ground_truth")
    pred_video_result = create_example_video_pose_result(pred_data, "prediction")

    euclidean_distance_config = {
        "threshold": threshold,
        "normalize_by": normalize_by
    }
    euclidean_distance_metric = EuclideanDistanceMetric(config=euclidean_distance_config)
    distances = euclidean_distance_metric.compute(
        video_result=pred_video_result,
        gt_video_result=gt_video_result,
        model_name="example_model"
    )
    return distances

class TestRmseEuclideanDistance(unittest.TestCase):
    """Test suite for the Euclidean distance and RMSE metric."""

    def test_basic_computation(self):
        """Test basic Euclidean distance computation, where one person is detected correctly and one is not."""
        gt_data = [
            [  # Frame 0
                [ # Person 0
                    (100, 100), (200, 200), (300, 300)
                ],
                [ # Person 1
                    (400, 400), (500, 500), (600, 600)
                ]
            ],
        ]

        # Create prediction data - with some offset from ground truth
        pred_data = [
            [  # Frame 0
                [ # Person 0
                    (110, 110), (210, 210), (310, 310),
                ],
                [ # Person 1
                    (420, 420), (520, 520), (620, 620)
                ]
            ],
        ]

        result_distances = compute_euclidean_distance_metric(gt_data, pred_data)
        np.testing.assert_array_almost_equal(
            result_distances.values,
            np.array([[
                [0.0707, 0.0707, 0.0707], 
                [0.1414, 0.1414, 0.1414]
            ]]),
            decimal=4
        )
        result_rmse = result_distances.aggregate(dims=["person", "keypoint"], method="rmse")
        np.testing.assert_array_almost_equal(
            result_rmse.values,
            np.array([0.1118]),
            decimal=4
        )

    def test_basic_computation_with_multiple_frames(self):
        """Test basic Euclidean distance computation, where a single person is detected correctly in each frame."""
        gt_data = [
            [  # Frame 0
                [ # Person 0
                    (100, 100), (200, 200), (300, 300)
                ],
            ],
            [  # Frame 1
                [ # Person 1
                    (400, 400), (500, 500), (600, 600)
                ],
            ],
        ]
        pred_data = [
            [  # Frame 0
                [ # Person 0
                    (110, 110), (210, 210), (310, 310)
                ],
            ],
            [  # Frame 1
                [ # Person 1
                    (410, 410), (510, 510), (610, 610)
                ],
            ],
        ]

        result_distances = compute_euclidean_distance_metric(gt_data, pred_data)
        np.testing.assert_array_almost_equal(
            result_distances.values,
            np.array([
                [[0.0707, 0.0707, 0.0707]], 
                [[0.0707, 0.0707, 0.0707]]
            ]),
            decimal=4
        )
        result_rmse = result_distances.aggregate(dims=["person", "keypoint"], method="rmse")
        np.testing.assert_array_almost_equal(
            result_rmse.values,
            np.array([0.0707, 0.0707]),
            decimal=4
        )

    def test_correct_detection_wrong_order(self):
        """
        Test that the Euclidean distance metric handles two correct person detections which are in the wrong order compared to the ground truth.
        """
        gt_data = [
            [  # Frame 0
                [ # Person 0
                    (100, 100), (200, 200), (300, 300)
                ],
                [ # Person 1
                    (400, 400), (500, 500), (600, 600)
                ]
            ],
        ]

        # Create prediction data - with some offset from ground truth
        pred_data = [
            [  # Frame 0
                [ # Person 0 (Person 1 in ground truth)
                    (410, 410), (510, 510), (610, 610)
                ],
                [ # Person 1 (Person 0 in ground truth)
                    (110, 110), (210, 210), (310, 310)
                ],
            ],
        ]

        result_distances = compute_euclidean_distance_metric(gt_data, pred_data)
        np.testing.assert_array_almost_equal(
            result_distances.values,
            np.array([[
                [0.0707, 0.0707, 0.0707], 
                [0.0707, 0.0707, 0.0707]
            ]]),
            decimal=4
        )
        result_rmse = result_distances.aggregate(dims=["person", "keypoint"], method="rmse")
        np.testing.assert_array_almost_equal(
            result_rmse.values,
            np.array([0.0707]),
            decimal=4
        )

    def test_missing_person_in_prediction(self):
        """
        Test that the Euclidean distance metric handles missing persons in the prediction.
        Every missing person in the prediction receives an infinite distance, 
        thereby resulting in a distance greater than the threshold
        """
        gt_data = [
            [  # Frame 0
                [ # Person 0
                    (100, 100), (200, 200), (300, 300)
                ],
                [ # Person 1
                    (400, 400), (500, 500), (600, 600)
                ]
            ], 
        ]

        pred_data = [
            [  # Frame 0
                [ # Person 0
                    (110, 110), (210, 210), (310, 310),
                ],
            ], 
        ]

        result_distances = compute_euclidean_distance_metric(gt_data, pred_data)
        np.testing.assert_array_almost_equal(
            result_distances.values,
            np.array([[
                [0.0707, 0.0707, 0.0707], 
                [DISTANCE_FILL_VALUE, DISTANCE_FILL_VALUE, DISTANCE_FILL_VALUE]
            ]]),
            decimal=4
        )
        result_rmse = result_distances.aggregate(dims=["person", "keypoint"], method="rmse")
        np.testing.assert_array_almost_equal(
            result_rmse.values,
            np.array([1.41509]),
            decimal=4
        )

    def test_additional_person_in_prediction(self):
        """
        Test that the Euclidean distance metric handles additional persons in the prediction.
        Every additional person in the prediction is ignored.
        """
        gt_data = [
            [  # Frame 0
                [ # Person 0
                    (100, 100), (200, 200), (300, 300)
                ],
            ],
        ]

        pred_data = [
            [  # Frame 0    
                [ # Person 0
                    (110, 110), (210, 210), (310, 310),
                ],
                [ # Person 1
                    (420, 420), (520, 520), (620, 620)
                ]
            ],
        ]

        result_distances = compute_euclidean_distance_metric(gt_data, pred_data)
        np.testing.assert_array_almost_equal(
            result_distances.values,
            np.array([[[0.0707, 0.0707, 0.0707]]]),
            decimal=4
        )
        result_rmse = result_distances.aggregate(dims=["person", "keypoint"], method="rmse")
        np.testing.assert_array_almost_equal(
            result_rmse.values,
            np.array([0.0707]),
            decimal=4
        )

    def test_missing_person_in_wrong_prediction_order(self):
        """
        In this test, one person is missing in the prediction. The remaining person is at index 0 in the
        prediction, but at index 1 in the ground truth.
        The metric must still match this correctly.
        """
        gt_data = [
            [  # Frame 0
                [ # Person 0
                    (100, 100), (200, 200), (300, 300)
                ],
                [ # Person 1
                    (400, 400), (500, 500), (600, 600)
                ],
            ], 
        ]

        pred_data = [
            [  # Frame 0
                [ # Person 0
                    (410, 410), (510, 510), (610, 610)
                ],
            ], 
        ]

        result_distances = compute_euclidean_distance_metric(gt_data, pred_data)
        np.testing.assert_array_almost_equal(
            result_distances.values,
            np.array([[
                [DISTANCE_FILL_VALUE, DISTANCE_FILL_VALUE, DISTANCE_FILL_VALUE],
                [0.0707, 0.0707, 0.0707] 
            ]]),
            decimal=4
        )
        result_rmse = result_distances.aggregate(dims=["person", "keypoint"], method="rmse")
        np.testing.assert_array_almost_equal(
            result_rmse.values,
            np.array([1.4151]),
            decimal=4
        )

    def test_missing_keypoint_in_gt_and_prediction(self):
        """
        Test that the Euclidean distance metric handles missing keypoints in the ground truth and prediction.
        Missing keypoints are ignored if they are not present in both the ground truth and prediction.
        """
        gt_data = [
            [  # Frame 0
                [ # Person 0
                    (100, 100), (0, 0), (300, 300)
                ],
            ],
        ]

        pred_data = [
            [  # Frame 0
                [ # Person 0
                    (110, 110), (0, 0), (310, 310)
                ],
            ],
        ]

        result_distances = compute_euclidean_distance_metric(gt_data, pred_data)
        np.testing.assert_array_almost_equal(
            result_distances.values,
            np.array([[[0.0707, np.nan, 0.0707]]]),
            decimal=4
        )
        result_rmse = result_distances.aggregate(dims=["person", "keypoint"], method="rmse")
        np.testing.assert_array_almost_equal(
            result_rmse.values,
            np.array([0.0707]),
            decimal=4
        )

    def test_missing_keypoint_in_prediction(self):
        """
        Test that the Euclidean distance metric handles missing keypoints in the prediction.
        Missing keypoints are ignored if they are not present in both the ground truth and prediction.
        """
        gt_data = [
            [  # Frame 0
                [ # Person 0
                    (100, 100), (200, 200), (300, 300)
                ],
            ],
        ]

        pred_data = [
            [  # Frame 0
                [ # Person 0
                    (110, 110), (0, 0), (310, 310)
                ],
            ],
        ]

        result_distances = compute_euclidean_distance_metric(gt_data, pred_data)
        np.testing.assert_array_almost_equal(
            result_distances.values,
            np.array([[[0.0707, DISTANCE_FILL_VALUE, 0.0707]]]),
            decimal=4
        )
        print("Result distances mask:\n", result_distances.values.mask)

        result_rmse = result_distances.aggregate(dims=["person", "keypoint"], method="rmse")
        np.testing.assert_array_almost_equal(
            result_rmse.values,
            np.array([1.1561]),
            decimal=4
        )

    def test_missing_keypoint_in_gt(self):
        """
        Test that the Euclidean distance metric handles missing keypoints in the ground truth and not in the prediction.
        Missing keypoints are ignored if they are not present in both the ground truth and prediction.
        """
        gt_data = [
            [  # Frame 0
                [ # Person 0
                    (100, 100), (0, 0), (300, 300)
                ],
            ],
        ]

        pred_data = [
            [  # Frame 0
                [ # Person 0
                    (110, 110), (210, 210), (310, 310)
                ],
            ],
        ]

        result_distances = compute_euclidean_distance_metric(gt_data, pred_data)
        np.testing.assert_array_almost_equal(
            result_distances.values,
            np.array([[[0.0707, np.nan, 0.0707]]]),
            decimal=4
        )
        result_rmse = result_distances.aggregate(dims=["person", "keypoint"], method="rmse")
        np.testing.assert_array_almost_equal(
            result_rmse.values,
            np.array([0.0707]),
            decimal=4
        )