from abc import ABC, abstractmethod
import cv2

from inference.pose_result import VideoPoseResult


class PoseEstimator(ABC):
    def __init__(self, name: str, config: dict):
        """
        Initialize the PoseEstimator with a name and configuration.
        
        Args:
            name (str): The name of the estimator (e.g. "YoloPose", "MediaPipe", "OpenPose", ...).
            config (dict): Configuration dictionary for the pose estimator. This can include arbitrary parameters for the model that are necessary for inference (e.g. "confidence_threshold", "weights_file_name", ...). The config parameter "confidence_threshold" is required. This has no effect for MaskAnyonePoseEstimators, because they do not provide confidence scores. If you do not want to filter, set confidence_threshold to 0.
        """
        if not config or "confidence_threshold" not in config:
            raise ValueError(f"Config for {name} must include a 'confidence_threshold' key.")

        self.name = name
        self.config = config
        self.confidence_threshold = config["confidence_threshold"]

    @abstractmethod
    def estimate_pose(self, video_path: str) -> VideoPoseResult:
        """
        Abstract method to estimate the pose of a video using the specific pose estimation model.
        This method should be implemented by subclasses.
        It receives the full path to the input video file and returns a VideoPoseResult object.
        The user is responsible for the following three steps after creating the initial VideoPoseResult object:
        1. Assert that the number of frames in the frame results matches the number of frames in the video (call assert_frame_count_is_correct)
        2. Filter out low confidence keypoints (call filter_low_confidence_keypoints)
        3. If the config contains a "save_keypoints_in_coco_format" key, convert the keypoints to the COCO format (call utils.convert_keypoints_to_coco_format, providing a mapping from the model output format to the COCO format)

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
        """
        Abstract method to get the pairs of keypoints that should be connected when visualizing the pose.
        This method should be implemented by subclasses.
        There are pre-defined list of keypoint pairs for various models available in the project.
        
        Returns:
            list: A list of tuples, where each tuple contains two integers representing the indices 
                 of keypoints that should be connected with a line when visualizing the pose.
                 For example, [(0,1), (1,2)] means that keypoint 0 should be connected to keypoint 1,
                 and keypoint 1 should be connected to keypoint 2.
        """
        pass

    def assert_frame_count_is_correct(self, video_pose_result: VideoPoseResult, video_metadata: dict):
        """
        Assert that the number of frames in the frame results matches the number of frames in the video. Should be called at the end of the estimate_pose method.

        Args:
            frame_results (list): A list of FramePoseResult objects.
            video_metadata (dict): A dictionary containing the video metadata with the key "frame_count".
        Raises:
            Exception: If the number of frames in the frame results does not match the number of frames in the video.
        """
        if len(video_pose_result.frames) != video_metadata.get("frame_count"):
            raise Exception(f"Number of frames in the video ({video_metadata.get('frame_count')}) does not match the number of frames in the frame results ({len(video_pose_result.frames)})")

    def filter_low_confidence_keypoints(self, video_pose_result: VideoPoseResult):
        for frame_result in video_pose_result.frames:
            for person_result in frame_result.persons:
                for keypoint in person_result.keypoints:
                    if keypoint.confidence is not None and keypoint.confidence < self.confidence_threshold:
                        keypoint.x = 0
                        keypoint.y = 0
                        keypoint.confidence = None
        return video_pose_result