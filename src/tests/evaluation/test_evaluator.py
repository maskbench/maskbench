import unittest
import numpy as np
import numpy.ma as ma

from evaluation.evaluator import Evaluator
from evaluation.metrics import Metric, MetricResult, FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS
from inference.pose_result import VideoPoseResult, FramePoseResult, PersonPoseResult, PoseKeypoint
from tests.utils import create_test_pose_result


class XCoordinateMappingMetric(Metric):
    """A simple test metric that just returns the x-coordinate of each keypoint.
    
    This metric is used for testing because:
    1. It's easy to understand and verify
    2. The x-coordinates follow a simple pattern in our test data
    3. It demonstrates how to implement a basic metric
    """
    
    def __init__(self):
        super().__init__(name="XCoordinateMappingMetric")
    
    def compute(self, video_result, gt_video_result=None, model_name=None):
        # Convert pose data to masked array and take only x-coordinates
        # Shape: (frames, persons, keypoints, 2) -> (frames, persons, keypoints)
        values = video_result.to_numpy_ma()[:, :, :, 0]  # Take only x coordinates
        
        return MetricResult(
            values=values,
            axis_names=[FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS],
            metric_name=self.name,
            video_name=video_result.video_name,
            model_name=model_name
        )


class TestEvaluator(unittest.TestCase):
    """Test cases for Evaluator class.
    
    The test suite verifies:
    1. Basic evaluation of a single model across multiple videos
    2. Different aggregation methods (frames, persons, keypoints)
    3. Handling of masked values (missing persons)
    """
    
    def setUp(self):
        """Create sample pose data for testing.
        
        Creates two videos with different numbers of persons per frame to test masking:
        - Video 1: Tests transition from 2 to 3 persons
        - Video 2: Tests transition from 1 to 2 persons
        """
        # Video 1: Tests handling of increasing number of persons (2->3)
        video1_poses = [
            # Frame 0: 2 persons
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],  # Person 1: x=1,2,3
                [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]   # Person 2: x=4,5,6
            ],
            # Frame 1: 3 persons
            [
                [[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]],    # Person 1: x=7,8,9
                [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],  # Person 2: x=10,11,12
                [[13.0, 13.0], [14.0, 14.0], [15.0, 15.0]]   # Person 3: x=13,14,15
            ]
        ]
        
        # Video 2: Tests handling of increasing number of persons (1->2)
        video2_poses = [
            # Frame 0: 1 person
            [
                [[16.0, 16.0], [17.0, 17.0], [18.0, 18.0]]  # Person 1: x=16,17,18
            ],
            # Frame 1: 2 persons
            [
                [[19.0, 19.0], [20.0, 20.0], [21.0, 21.0]],  # Person 1: x=19,20,21
                [[22.0, 22.0], [23.0, 23.0], [24.0, 24.0]]   # Person 2: x=22,23,24
            ]
        ]
        
        # Create VideoPoseResult objects using the utility function
        self.video1 = create_test_pose_result(video1_poses, video_name="video1")
        self.video2 = create_test_pose_result(video2_poses, video_name="video2")
        
        # Create model results
        self.model_results = {
            "model1": [self.video1, self.video2]
        }
        
        # Create evaluator with our test metric
        self.evaluator = Evaluator(metrics=[XCoordinateMappingMetric()])
    
    def test_evaluate_model(self):
        """Test evaluation of a single model.
        
        Verifies:
        1. Correct structure of results (metric -> model -> video)
        2. Correct shapes of arrays (accounting for masking)
        3. Specific values in the results
        4. Handling of varying number of persons
        """
        results = self.evaluator.evaluate(self.model_results)
        
        # Check structure of results
        self.assertIn("XCoordinateMappingMetric", results)
        self.assertIn("model1", results["XCoordinateMappingMetric"])
        self.assertIn("video1", results["XCoordinateMappingMetric"]["model1"])
        self.assertIn("video2", results["XCoordinateMappingMetric"]["model1"])
        
        # Check video1 results (2->3 persons)
        video1_result = results["XCoordinateMappingMetric"]["model1"]["video1"]
        self.assertEqual(video1_result.values.shape, (2, 3, 3))  # 2 frames, max 3 persons, 3 metric values per keypoint
        
        # Check specific values
        np.testing.assert_array_almost_equal(
            video1_result.values[0, 0],  # Frame 0, Person 0
            [1.0, 2.0, 3.0]  # First person's x-coordinates
        )
        np.testing.assert_array_almost_equal(
            video1_result.values[1, 2],  # Frame 1, Person 2
            [13.0, 14.0, 15.0]  # Third person's x-coordinates
        )
        
        # Check video2 results (1->2 persons)
        video2_result = results["XCoordinateMappingMetric"]["model1"]["video2"]
        self.assertEqual(video2_result.values.shape, (2, 2, 3))  # 2 frames, max 2 persons, 3 metric values per keypoint
        
        # Check specific values
        np.testing.assert_array_almost_equal(
            video2_result.values[0, 0],  # Frame 0, Person 0
            [16.0, 17.0, 18.0]  # First person's x-coordinates
        )
    
    def test_aggregate_results(self):
        """Test aggregation of metric results.
        
        Verifies:
        1. Aggregation over frames (mean of each person's keypoints across frames)
        2. Aggregation over persons (mean of each frame's keypoints across persons)
        3. Aggregation over keypoints (mean of each person's frames across keypoints)
        4. Overall aggregation (mean of all values)
        
        The test data is designed so that:
        - Frame aggregation: Tests handling of varying number of persons
        - Person aggregation: Tests handling of masked values
        - Keypoint aggregation: Tests basic mean computation
        - Overall aggregation: Tests handling of all masked values
        """
        results = self.evaluator.evaluate(self.model_results)
        video1_result = results["XCoordinateMappingMetric"]["model1"]["video1"]
        
        # Aggregate over persons
        # Result: mean of each frame's keypoints across persons
        person_agg = video1_result.aggregate(PERSON_AXIS)
        self.assertEqual(person_agg.values.shape, (2, 3))  # frames, keypoints
        
        # Aggregate over keypoints
        # Result: mean of each person's frames across keypoints
        keypoint_agg = video1_result.aggregate(KEYPOINT_AXIS)
        self.assertEqual(keypoint_agg.values.shape, (2, 3))  # frames, persons

        person_keypoint_agg = video1_result.aggregate([PERSON_AXIS, KEYPOINT_AXIS])
        self.assertEqual(person_keypoint_agg.values.shape, (2,))  # frames, persons
        
        # Aggregate all
        # Result: mean of all unmasked values
        # For video1: (1+2+3+4+5+6+7+8+9+10+11+12+13+14+15) / 15 = 8.0
        overall_mean_video1 = video1_result.aggregate_all()
        self.assertAlmostEqual(overall_mean_video1, 8.0)


if __name__ == '__main__':
    unittest.main()
