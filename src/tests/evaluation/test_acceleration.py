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

    def test_mismatched_person_indices_over_frames(self):
        """
        Test acceleration computation with multiple persons that are out of order over different frames.
        """
        pred_data = [
            [  # Frame 0
                [(100, 100), (100, 200), (150, 150)], # Person 0
                [(300, 300), (300, 400), (350, 350)], # Person 1
            ],
            [  # Frame 1
                [(290, 290), (290, 390), (340, 340)], # Person 1
                [(110, 110), (110, 210), (160, 160)], # Person 0
            ],
            [  # Frame 2
                [(270, 270), (270, 370), (320, 320)], # Person 1
                [(120, 120), (120, 220), (170, 170)], # Person 0
            ],
            [  # Frame 3
                [(130, 130), (130, 230), (180, 180)], # Person 0
                [(230, 230), (230, 330), (280, 280)], # Person 1
            ]
        ]

        result = compute_acceleration_metric(pred_data, fps=1)
        self.assertEqual(result.values.shape, (2, 2, 3))
        expected_accelerations = np.array([
            [ # Pseudo-Frame 0
                [0, 0, 0],      # Person 0
                [14, 14, 14]    # Person 1
            ],
            [ # Pseudo-Frame 1
                [0, 0, 0],      # Person 0
                [28, 28, 28]    # Person 1
            ]
        ])
        np.testing.assert_array_almost_equal(result.values, expected_accelerations, decimal=0)
    