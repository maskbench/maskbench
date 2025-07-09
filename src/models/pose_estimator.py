from abc import ABC, abstractmethod
import cv2

from inference.pose_result import VideoPoseResult


class PoseEstimator(ABC):
    def __init__(self, name: str, config: dict = None):
        """
        Initialize the PoseEstimator with a name and configuration.
        Args:
            name (str): The name of the estimator (e.g. "YoloPose", "MediaPipe", "OpenPose", ...).
            config (dict): Configuration dictionary for the pose estimator. This can include arbitrary parameters for the model that are necessary for inference (e.g. "confidence_threshold", "weights_file_name", ...).
        """

        self.name = name
        self.config = config if config else {}

    @abstractmethod
    def estimate_pose(self, video_path: str) -> VideoPoseResult:
        """
        Abstract method to estimate the pose of a video using the specific pose estimation model.
        This method should be implemented by subclasses.
        Args:
            video_path (str): The full path to the input video file.
        Returns:
            VideoPoseResult: An object containing the pose estimation results for the video.
                  The VideoPoseResult object must contain as many FramePoseResult objects as there are frames in the video (asserted by assert_frame_count_is_correct).
                  The FramePoseResult contains a list of PersonPoseResult objects, one for each person in the frame. If there are no persons in the frame, the list is empty (persons=[]).
                  Every PersonPoseResult contains a list of PoseKeypoints, one for each keypoint in the model output format. If a keypoint is not detected, the PoseKeypoint object should have x=0 and y=0 and confidence=None.
        """
        pass

    @abstractmethod
    def get_keypoint_pairs(self) -> list:
        pass

    def assert_frame_count_is_correct(self, frame_results: list, video_metadata: dict):
        """
        Assert that the number of frames in the frame results matches the number of frames in the video. Should be called at the end of the estimate_pose method.
        Args:
            frame_results (list): A list of FramePoseResult objects.
            video_metadata (dict): A dictionary containing the video metadata with the key "frame_count".
        Raises:
            Exception: If the number of frames in the frame results does not match the number of frames in the video.
        """
        if len(frame_results) != video_metadata.get("frame_count"):
            raise Exception(f"Number of frames in the video ({video_metadata.get('frame_count')}) does not match the number of frames in the frame results ({len(frame_results)})")