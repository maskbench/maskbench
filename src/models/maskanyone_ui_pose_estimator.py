import os
import utils

from inference import VideoPoseResult
from models import PoseEstimator

class MaskAnyoneUiPoseEstimator(PoseEstimator):
    def __init__(self, name: str, config: dict):
        """
        Initialize the MaskAnyoneUiPoseEstimator with a name and configuration.
        """
        super().__init__(name, config)
        self.maskanyone_ui_dataset = self.config.get("dataset_folder_path")

    def get_keypoint_pairs(self):
        return [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 7),
            (0, 4),
            (4, 5),
            (5, 6),
            (6, 8),
            (9, 10),
            (11, 12),
            (11, 13),
            (13, 15),
            (15, 19),
            (15, 17),
            (17, 19),
            (15, 21),
            (12, 14),
            (14, 16),
            (16, 20),
            (16, 18),
            (18, 20),
            (11, 23),
            (12, 24),
            (16, 22),
            (23, 25),
            (24, 26),
            (25, 27),
            (26, 28),
            (23, 24),
            (28, 30),
            (28, 32),
            (30, 32),
            (27, 29),
            (27, 31),
            (29, 31),
        ]

    def estimate_pose(self, video_path: str) -> VideoPoseResult:
        """
        Estimate the pose of a video using Mask Anyone Ui  estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            VideoPoseResult: A standardized result object containing the pose estimation results for the video.
        """
        _, video_metadata = utils.get_video_metadata(video_path) # get video 

        video_name = os.path.splitext(os.path.basename(video_path))[0] # get video name
        results_path = os.path.join(self.maskanyone_ui_dataset, video_name) # path of jsons
        if not os.path.exists(results_path):
            raise f"{video_name} was not found in Mask Anyone Ui Dataset Folder Path"
            
        frame_results = utils.maskanyone_combine_json_files(results_path)  # Combine the JSON files from processed chunks
        return VideoPoseResult(
            fps=video_metadata.get("fps"),
            frame_width=video_metadata.get("width"),
            frame_height=video_metadata.get("height"),
            video_name=video_name,
            frames=frame_results
        )

    