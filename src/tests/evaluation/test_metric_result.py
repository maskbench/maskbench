import unittest
import numpy as np
import numpy.ma as ma

from evaluation.metrics.metric_result import MetricResult, FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS


class TestMetricResult(unittest.TestCase):
    """Test cases for MetricResult class."""
    
    def setUp(self):
        """Create a sample metric result for testing."""
        # Some values will be masked to simulate missing data
        values = np.array([
            # Frame 0
            [
                [1.0, 2.0, 3.0, 4.0],  # Person 0
                [5.0, 6.0, 7.0, 8.0],  # Person 1
                [9.0, 10.0, 11.0, 12.0]  # Person 2
            ],
            # Frame 1
            [
                [13.0, 14.0, 15.0, 16.0],  # Person 0
                [17.0, 18.0, 19.0, 20.0],  # Person 1
                [21.0, 22.0, 23.0, 24.0]  # Person 2
            ]
        ])
        
        mask = np.zeros_like(values, dtype=bool)
        mask[0, 2] = True  # Mask person 2 in frame 0
        mask[1, 1] = True  # Mask person 1 in frame 1
        
        self.metric_result = MetricResult(
            values=ma.array(values, mask=mask),
            axis_names=[FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS],
            metric_name='test_metric',
            video_name='test_video'
        )
    
    def test_aggregate_single_axis(self):
        """Test aggregating over a single axis."""
        keypoint_agg = self.metric_result.aggregate(KEYPOINT_AXIS)
        self.assertEqual(keypoint_agg.values.shape, (2, 3))  # (frames, persons)
        self.assertEqual(keypoint_agg.axis_names, [FRAME_AXIS, PERSON_AXIS])
        
        # For frame 0, person 0: mean of all keypoints
        np.testing.assert_array_almost_equal(
            keypoint_agg.values[0, 0],
            (1.0 + 2.0 + 3.0 + 4.0) / 4
        )
        
        # For frame 0, person 1: mean of all keypoints
        np.testing.assert_array_almost_equal(
            keypoint_agg.values[0, 1],
            (5.0 + 6.0 + 7.0 + 8.0) / 4
        )
        
        # For frame 1, person 0: mean of all keypoints
        np.testing.assert_array_almost_equal(
            keypoint_agg.values[1, 0],
            (13.0 + 14.0 + 15.0 + 16.0) / 4
        )
        
        # For frame 1, person 2: mean of all keypoints
        np.testing.assert_array_almost_equal(
            keypoint_agg.values[1, 2],
            (21.0 + 22.0 + 23.0 + 24.0) / 4
        )
    
    def test_aggregate_multiple_axes(self):
        """Test aggregating over multiple axes."""
        # Aggregate over frames and persons
        result = self.metric_result.aggregate([FRAME_AXIS, PERSON_AXIS])
        self.assertEqual(result.values.shape, (4,))  # (keypoints,)
        self.assertEqual(result.axis_names, [KEYPOINT_AXIS])
        
        # Values should be mean of all unmasked entries
        expected = np.array([
            (1.0 + 5.0 + 13.0 + 21.0) / 4,  # First keypoint
            (2.0 + 6.0 + 14.0 + 22.0) / 4,  # Second keypoint
            (3.0 + 7.0 + 15.0 + 23.0) / 4,  # Third keypoint
            (4.0 + 8.0 + 16.0 + 24.0) / 4   # Fourth keypoint
        ])
        np.testing.assert_array_almost_equal(result.values, expected)
    
    def test_different_aggregation_methods(self):
        """Test different aggregation methods (mean, median, min, max)."""
        # Create simple data with known statistics
        values = ma.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        metric = MetricResult(
            values=values,
            axis_names=['dim1', 'dim2'],
            metric_name='test_metric',
            video_name='test_video'
        )
        
        # Test different methods along first dimension
        mean_result = metric.aggregate('dim1', method='mean')
        np.testing.assert_array_almost_equal(
            mean_result.values,
            [4.0, 5.0, 6.0]  # Mean of each column
        )
        
        median_result = metric.aggregate('dim1', method='median')
        np.testing.assert_array_almost_equal(
            median_result.values,
            [4.0, 5.0, 6.0]  # Median of each column
        )
        
        min_result = metric.aggregate('dim1', method='min')
        np.testing.assert_array_almost_equal(
            min_result.values,
            [1.0, 2.0, 3.0]  # Min of each column
        )
        
        max_result = metric.aggregate('dim1', method='max')
        np.testing.assert_array_almost_equal(
            max_result.values,
            [7.0, 8.0, 9.0]  # Max of each column
        )
    
    def test_aggregate_all(self):
        """Test aggregating over all dimensions."""
        # Should return mean of all unmasked values
        result = self.metric_result.aggregate_all()
        
        expected_mean = ma.array([ # should be 11.5
            # Frame 0
            [1.0, 2.0, 3.0, 4.0],    # Person 0
            [5.0, 6.0, 7.0, 8.0],    # Person 1
            # Person 2 is masked
            
            # Frame 1
            [13.0, 14.0, 15.0, 16.0],  # Person 0
            # Person 1 is masked
            [21.0, 22.0, 23.0, 24.0]   # Person 2
        ]).mean()
        self.assertAlmostEqual(result, expected_mean)
    
    def test_invalid_aggregation(self):
        """Test error handling for invalid aggregation parameters."""
        # Test invalid axis name
        with self.assertRaises(KeyError):
            self.metric_result.aggregate('invalid_axis')
        
        # Test invalid aggregation method
        with self.assertRaises(ValueError):
            self.metric_result.aggregate(FRAME_AXIS, method='invalid_method')


if __name__ == '__main__':
    unittest.main() 