
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


class TestPCKMetric:
    """Test suite for the PCKMetric class."""
    
    def test_basic_computation(self):
        """Test basic MPJPE computation with valid inputs."""
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

        gt_video_result = create_example_video_pose_result(gt_data, "ground_truth")
        pred_video_result = create_example_video_pose_result(pred_data, "prediction")
        
        pck_config = {
            "threshold": 0.1,
            "normalize_by": "bbox"
        }
        pck_metric = PCKMetric(config=pck_config)
        
        result = pck_metric.compute(
            video_result=pred_video_result,
            gt_video_result=gt_video_result,
            model_name="example_model"
        )

