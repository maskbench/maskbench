import os
import utils
from inference import VideoPoseResult
from models import PoseEstimator
from keypoint_pairs import *

class MaskAnyoneUiPoseEstimator(PoseEstimator):
    def __init__(self, name: str, config: dict):
        """
        Initialize the MaskAnyoneUiPoseEstimator with a name and configuration.
        """
        super().__init__(name, config)
        self.dataset_folder_path = self.config.get("dataset_folder_path")
        self.options = utils.maskanyone_get_config(self.config)
        self.model_keypoint_pairs = {"mp_pose": MEDIAPIPE_KEYPOINT_PAIRS, "openpose_body25b": OPENPOSE_BODY25B_KEYPOINT_PAIRS, "openpose": OPENPOSE_BODY25_KEYPOINT_PAIRS}
        self.model_to_coco_mapping = {"mp_pose": COCO_TO_MEDIAPIPE, "openpose_body25b": COCO_TO_OPENPOSE_BODY25B, "openpose": COCO_TO_OPENPOSE_BODY25}

    def get_keypoint_pairs(self):
        if self.config.get("save_keypoints_in_coco_format", False):
            return COCO_KEYPOINT_PAIRS
        else:
            return self.model_keypoint_pairs[self.config.get("overlay_strategy")]

    def estimate_pose(self, video_path: str) -> VideoPoseResult:
        """
        Estimate the pose of a video using Mask Anyone Ui  estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            VideoPoseResult: A standardized result object containing the pose estimation results for the video.
        """
        _, video_metadata = utils.get_video_metadata(video_path)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        dir_path = os.path.join(self.dataset_folder_path, video_name)
        json_path = os.path.join(self.dataset_folder_path, f"{video_name}.json")
        
        if not os.path.exists(dir_path) and not os.path.exists(json_path):
            raise FileNotFoundError(f"Neither {dir_path} nor {json_path} was found in Mask Anyone Ui Dataset Folder Path")

        overlay_strategy = self.config.get("overlay_strategy")
        if os.path.exists(json_path): # If it's a single JSON file, process it directly
            frame_results = utils.maskanyone_convert_json_to_nested_arrays(json_path, overlay_strategy)
        elif os.path.exists(dir_path) and os.path.isdir(dir_path): # If it's a directory containing JSON chunks, combine them
            frame_results = utils.maskanyone_combine_json_files(dir_path, overlay_strategy)
        else:
            raise ValueError(f"Neither {dir_path} is a directory nor {json_path} is a file")
    
        video_pose_result = VideoPoseResult(
            fps=video_metadata.get("fps"),
            frame_width=video_metadata.get("width"),
            frame_height=video_metadata.get("height"),
            video_name=video_name,
            frames=frame_results
        )

        self.assert_frame_count_is_correct(video_pose_result, video_metadata)
        video_pose_result = self.filter_low_confidence_keypoints(video_pose_result) # this call will have no effect, because MaskAnyone does not provide confidence scores
        if self.config.get("save_keypoints_in_coco_format", False):
            video_pose_result.frames = utils.convert_keypoints_to_coco_format(video_pose_result.frames, self.model_to_coco_mapping[self.config.get("overlay_strategy")])
        return video_pose_result
    