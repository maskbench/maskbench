import unittest
import numpy as np
import numpy.ma as ma
import pytest

from evaluation.metrics.pck import PCKMetric
from inference.pose_result import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult

def create_example_video_pose_result(keypoints_data, video_name="example_video"):
    """Helper function to create a VideoPoseResult from keypoints data.
    
    Args:
        keypoints_data: List of frames, where each frame is a list of persons,
                       and each person is a list of (x,y) coordinates
    """
    frames = []
    for frame_idx, frame_data in enumerate(keypoints_data):
        persons = []
        for person_data in frame_data:
            keypoints = [
                PoseKeypoint(x=float(x), y=float(y))
                for x, y in person_data
            ]
            persons.append(PersonPoseResult(keypoints=keypoints))
        frames.append(FramePoseResult(persons=persons, frame_idx=frame_idx))
    
    return VideoPoseResult(
        fps=30,
        frame_width=1920,
        frame_height=1080,
        frames=frames,
        video_name=video_name
    )

def compute_pck_metric(gt_data, pred_data, threshold=0.1, normalize_by="bbox"):
    """Compute the PCK metric."""
    gt_video_result = create_example_video_pose_result(gt_data, "ground_truth")
    pred_video_result = create_example_video_pose_result(pred_data, "prediction")

    pck_config = {
        "threshold": threshold,
        "normalize_by": normalize_by
    }
    pck_metric = PCKMetric(config=pck_config)
    result = pck_metric.compute(
        video_result=pred_video_result,
        gt_video_result=gt_video_result,
        model_name="example_model"
    )
    return result

class TestPCKMetric(unittest.TestCase):
    """Test suite for the PCKMetric class."""

    def test_basic_computation(self):
        """Test basic PCK computation, where one person is detected correctly and one is not."""
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

        result = compute_pck_metric(gt_data, pred_data)
        np.testing.assert_array_equal(result.values, np.array([0.5]))

    def test_basic_computation_with_multiple_frames(self):
        """Test basic PCK computation, where a single person is detected correctly in each frame."""
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

        result = compute_pck_metric(gt_data, pred_data)
        np.testing.assert_array_equal(result.values, np.array([1.0, 1.0]))

    def test_correct_detection_wrong_order(self):
        """
        Test that the PCK metric handles two correct person detections which are in the wrong order compared to the ground truth.
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

        result = compute_pck_metric(gt_data, pred_data)
        np.testing.assert_array_equal(result.values, np.array([1.0]))

    def test_missing_person_in_prediction(self):
        """
        Test that the PCK metric handles missing persons in the prediction.
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

        result = compute_pck_metric(gt_data, pred_data)
        np.testing.assert_array_equal(result.values, np.array([0.5]))

    def test_additional_person_in_prediction(self):
        """
        Test that the PCK metric handles additional persons in the prediction.
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

        result = compute_pck_metric(gt_data, pred_data)
        np.testing.assert_array_equal(result.values, np.array([1.0]))


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

        result = compute_pck_metric(gt_data, pred_data)
        np.testing.assert_array_equal(result.values, np.array([0.5]))