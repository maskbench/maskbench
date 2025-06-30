import unittest
import numpy as np
import numpy.ma as ma
import pytest

from evaluation.metrics import AccelerationMetric
from tests.utils import create_example_video_pose_result

def compute_acceleration_metric(pred_data, fps: int = 30):
    """Compute the acceleration metric."""
    pred_video_result = create_example_video_pose_result(pred_data, "prediction", fps=fps)

    acceleration_metric = AccelerationMetric()
    result = acceleration_metric.compute(
        video_result=pred_video_result,
        model_name="example_model"
    )
    return result

class TestAccelerationMetric(unittest.TestCase):
    """Test suite for the AccelerationMetric class."""

    def test_zero_acceleration(self):
        """Test zero acceleration for a single person moving with constant velocity."""
        pred_data = [
            [  # Frame 0
                [(100, 100), (200, 200), (300, 300)],
            ],
            [  # Frame 1
                [(110, 110), (210, 210), (310, 310)],
            ],
            [  # Frame 2
                [(120, 120), (220, 220), (320, 320)],
            ],
            [  # Frame 3
                [(130, 130), (230, 230), (330, 330)],
            ],
        ]

        result = compute_acceleration_metric(pred_data)
        # Since the person moves with constant velocity (10,10) per frame the acceleration should be close to zero
        # Note: we get T-2 frames for acceleration where T is number of input frames
        expected_shape = (2, 1, 3)  # (frames, persons, keypoints)
        self.assertEqual(result.values.shape, expected_shape)
        np.testing.assert_array_equal(result.values, np.zeros(expected_shape))

    def test_non_zero_acceleration(self):
        """Test specific non-zero accelerations for each keypoint."""
        pred_data = [
            [  # Frame 0
                [(100, 100), (200, 200), (300, 300)],
            ],
            [  # Frame 1
                [(110, 110), (220, 220), (340, 340)],
            ],
            [  # Frame 2
                [(130, 130), (260, 260), (420, 420)],
            ],
            [  # Frame 3
                [(170, 170), (340, 340), (580, 580)],
            ],
        ]

        result = compute_acceleration_metric(pred_data, fps=1)
        expected_shape = (2, 1, 3)  # (frames, persons, keypoints)
        self.assertEqual(result.values.shape, expected_shape)
        
        # Expected accelerations for each keypoint over two frames
        expected_accelerations = np.array([
            [[14, 28, 56]],
            [[28, 56, 113]]
        ])
        np.testing.assert_array_almost_equal(result.values, expected_accelerations, decimal=0)

    def test_not_enough_frames(self):
        """
        Test acceleration computation with missing frames.
        In case the number of frames is less than 3, the acceleration should be NaN.
        """
        pred_data = [
            [  # Frame 0
                [(100, 100), (200, 200), (300, 300)],
            ],
        ]

        result = compute_acceleration_metric(pred_data)
        expected_shape = (1, 1, 3)  # (frames, persons, keypoints)
        self.assertEqual(result.values.shape, expected_shape)
        np.testing.assert_array_equal(result.values, np.nan * np.ones(expected_shape))

    def test_missing_keypoints(self):
        """
        Test acceleration computation with missing keypoints. 
        Every time a keypoint is missing in one of three consecutive frames, the acceleration is NaN.
        """
        pred_data = [
            [  # Frame 0
                [(0, 0),     (0, 0),     (300, 300), (400, 400), (500, 500),   (600, 600)],
            ],
            [  # Frame 1
                [(110, 110), (0, 0),     (0, 0),     (0, 0),     (660, 660),   (610, 610)],
            ],
            [  # Frame 2
                [(130, 130), (260, 260), (420, 420), (0, 0),     (980, 980),   (620, 620)],
            ],
            [  # Frame 3
                [(170, 170), (340, 340), (580, 580), (960, 960), (1620, 1620), (0, 0)],
            ],
        ]

        result = compute_acceleration_metric(pred_data, fps=1)
        expected_shape = (2, 1, 6)  # (frames, persons, keypoints)
        self.assertEqual(result.values.shape, expected_shape)
        expected_result = np.array([
            [[np.nan, np.nan, np.nan, np.nan, 226, 0]],
            [[28, np.nan, np.nan, np.nan, 452, np.nan]]
        ])
        np.testing.assert_array_almost_equal(result.values, expected_result, decimal=0)

    def test_acceleration_with_missing_persons(self):
        """Test acceleration computation with missing persons."""
        pred_data = [
            [  # Frame 0
                [ # Person 0
                    (100, 100), (200, 200), (300, 300)
                ],
                [ # Person 1
                    (100, 100), (200, 200), (300, 300)
                ],
            ],
            [  # Frame 1
                [ # Person 0
                    (110, 110), (220, 220), (340, 340)
                ],
            ],
            [  # Frame 2
                [ # Person 0
                    (130, 130), (260, 260), (380, 380)
                ],
                [ # Person 1
                    (130, 130), (260, 260), (380, 380)
                ],
            ],
            [  # Frame 3
                [ # Person 0
                    (170, 170), (340, 340), (580, 580)
                ],
            ],
        ]

        result = compute_acceleration_metric(pred_data, fps=1)
    