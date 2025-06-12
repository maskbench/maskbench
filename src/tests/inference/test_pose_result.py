"""Tests for pose result functionality."""
import unittest
import numpy as np
import numpy.ma as ma

from inference.pose_result import PoseKeypoint, PersonPoseResult, FramePoseResult, VideoPoseResult


class TestPoseResult(unittest.TestCase):
    """Test cases for pose result classes."""
    
    def test_empty_video(self):
        """Test converting an empty video (no frames) to numpy array."""
        video = VideoPoseResult(
            fps=30,
            frame_width=1920,
            frame_height=1080,
            frames=[],
            video_name="empty_video"
        )
        result = video.to_numpy_ma()
        
        self.assertIsInstance(result, ma.MaskedArray)
        self.assertEqual(result.shape, (0, 0, 0, 2))
    
    def test_single_person_constant(self):
        """Test video with single person with same position in all frames."""
        # Create 3 frames with 1 person, 2 keypoints each
        frames = []
        for i in range(3):
            keypoints = [
                PoseKeypoint(x=100.0, y=200.0),  # First keypoint
                PoseKeypoint(x=150.0, y=250.0)   # Second keypoint
            ]
            person = PersonPoseResult(keypoints=keypoints)
            frame = FramePoseResult(persons=[person], frame_idx=i)
            frames.append(frame)
            
        video = VideoPoseResult(
            fps=30,
            frame_width=1920,
            frame_height=1080,
            frames=frames,
            video_name="single_person"
        )
        result = video.to_numpy_ma()
        
        # Check shape: 3 frames, 1 person, 2 keypoints, 2 coordinates
        self.assertEqual(result.shape, (3, 1, 2, 2))
        
        # Check values for first keypoint
        np.testing.assert_array_equal(
            result[:, 0, 0],  # All frames, first person, first keypoint
            [[100.0, 200.0]] * 3  # Should be same in all frames
        )
        
        # Check values for second keypoint
        np.testing.assert_array_equal(
            result[:, 0, 1],  # All frames, first person, second keypoint
            [[150.0, 250.0]] * 3  # Should be same in all frames
        )
        
        # Check that no values are masked
        self.assertFalse(result.mask.any())
    
    def test_varying_persons_per_frame(self):
        """Test video with different number of persons in different frames."""
        
        # Helper to create a person with 2 keypoints at specified positions
        def create_person(x1, y1, x2, y2):
            return PersonPoseResult(keypoints=[
                PoseKeypoint(x=x1, y=y1),
                PoseKeypoint(x=x2, y=y2)
            ])
        
        frames = []
        # Frame 0: 2 persons
        frames.append(FramePoseResult(
            persons=[
                create_person(100, 200, 150, 250),
                create_person(300, 400, 350, 450)
            ],
            frame_idx=0
        ))
        
        # Frame 1: 1 person
        frames.append(FramePoseResult(
            persons=[create_person(100, 200, 150, 250)],
            frame_idx=1
        ))
        
        # Frame 2: 3 persons
        frames.append(FramePoseResult(
            persons=[
                create_person(100, 200, 150, 250),
                create_person(300, 400, 350, 450),
                create_person(500, 600, 550, 650)
            ],
            frame_idx=2
        ))
        
        video = VideoPoseResult(
            fps=30,
            frame_width=1920,
            frame_height=1080,
            frames=frames,
            video_name="varying_persons"
        )
        result = video.to_numpy_ma()
        
        # Check shape: 3 frames, 3 persons max, 2 keypoints, 2 coordinates
        self.assertEqual(result.shape, (3, 3, 2, 2))
        
        # Check masking for frame 0 (2 persons)
        self.assertFalse(result.mask[0, :2].any())  # First 2 persons should not be masked
        self.assertTrue(result.mask[0, 2].all())    # Third person should be masked
        
        # Check masking for frame 1 (1 person)
        self.assertFalse(result.mask[1, 0].any())   # First person should not be masked
        self.assertTrue(result.mask[1, 1:].all())   # Second and third persons should be masked
        
        # Check masking for frame 2 (3 persons)
        self.assertFalse(result.mask[2].any())      # No persons should be masked
        
        # Check some actual values
        np.testing.assert_array_equal(
            result[0, 0, 0],  # Frame 0, first person, first keypoint
            [100.0, 200.0]
        )
        np.testing.assert_array_equal(
            result[1, 0, 0],  # Frame 1, first person, first keypoint
            [100.0, 200.0]
        )
        np.testing.assert_array_equal(
            result[2, 2, 1],  # Frame 2, third person, second keypoint
            [550.0, 650.0]
        )


if __name__ == '__main__':
    unittest.main() 