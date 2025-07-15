import os
import utils
from inference import VideoPoseResult
from models import PoseEstimator
from keypoint_pairs import COCO_KEYPOINT_PAIRS, OPENPOSE_KEYPOINT_PAIRS, MEDIAPIPE_KEYPOINT_PAIRS

class MaskAnyoneUiPoseEstimator(PoseEstimator):
    def __init__(self, name: str, config: dict):
        """
        Initialize the MaskAnyoneUiPoseEstimator with a name and configuration.
        """
        super().__init__(name, config)
        self.maskanyone_ui_dataset = self.config.get("dataset_folder_path")
        self.options = utils.maskanyone_get_config(self.config)

    def get_keypoint_pairs(self):
        if self.config.get("save_keypoints_in_coco_format", False):
            return COCO_KEYPOINT_PAIRS
        elif self.config.get("overlay_strategy") == "mp_pose":
            return MEDIAPIPE_KEYPOINT_PAIRS 
        elif self.config.get("overlay_strategy") == "openpose_body25b":
            return OPENPOSE_KEYPOINT_PAIRS
        else:
            raise ValueError(f"Invalid overlay strategy. Valid options are: mp_pose and openpose_body25b ")
        

    def estimate_pose(self, video_path: str) -> VideoPoseResult:
        """
        Estimate the pose of a video using Mask Anyone Ui  estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            VideoPoseResult: A standardized result object containing the pose estimation results for the video.
        """
        _, video_metadata = utils.get_video_metadata(video_path) # get video specs

        video_name = os.path.splitext(os.path.basename(video_path))[0] # get video name
        results_path = os.path.join(self.maskanyone_ui_dataset, video_name) # path of jsons
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"{results_path} was not found in Mask Anyone Ui Dataset Folder Path")

        frame_results = utils.maskanyone_combine_json_files(results_path)  # Combine the JSON files from processed chunks
    
        if self.config.get("save_keypoints_in_coco_format", False):
            frame_results = utils.convert_keypoints_to_coco_format(frame_results, self.config.get("overlay_strategy"))

        video_pose_result = VideoPoseResult(
            fps=video_metadata.get("fps"),
            frame_width=video_metadata.get("width"),
            frame_height=video_metadata.get("height"),
            video_name=video_name,
            frames=frame_results
        )

        self.assert_frame_count_is_correct(video_pose_result, video_metadata)
        video_pose_result = self.filter_low_confidence_keypoints(video_pose_result) # this call will have no effect, because MaskAnyone does not provide confidence scores
        return video_pose_result
    