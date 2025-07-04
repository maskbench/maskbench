import os
import torch
import utils
from ultralytics import YOLO

from models import PoseEstimator
from inference import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult
from keypoint_pairs import YOLO_KEYPOINT_PAIRS

class YoloPoseEstimator(PoseEstimator):
    def __init__(self, name: str, config: dict):
        """
        Initialize the YoloPoseEstimator with a model name and configuration.
        Args:
            name (str): The name of the model (e.g. "yolo-pose-v8", "yolo-pose-v11").
            config (dict): Configuration dictionary for the model. It must contain the key "weights" with the path to the weights file relative to the weights folder. Note that MaskBench does not download the weights for you. Please visit https://docs.ultralytics.com/tasks/pose/ to download the weights.
        """

        super().__init__(name, config)

        weights_file = self.config.get("weights")
        pre_built_weights_file_path = os.path.join("/weights/pre_built", weights_file)
        user_weights_file_path = os.path.join("/weights/user_weights", weights_file)

        if os.path.exists(user_weights_file_path):
            weights_file_path = user_weights_file_path
        elif os.path.exists(pre_built_weights_file_path):
            weights_file_path = pre_built_weights_file_path
        else:
            raise ValueError(
                f"Could not find weights file {weights_file}. Please download the weights from https://docs.ultralytics.com/tasks/pose/ and place them in the weights folder."
            )

        self.model = YOLO(weights_file_path)
        # only for dev
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

    def get_keypoint_pairs(self):
        return YOLO_KEYPOINT_PAIRS

    def estimate_pose(self, video_path: str) -> VideoPoseResult:
        """
        Estimate the pose of a video using YOLO pose estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            VideoPoseResult: A standardized result object containing the pose estimation results for the video.
        """

        cap, video_metadata = utils.get_video_metadata(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap.release()

        confidence = self.config.get("confidence_threshold", 0.85)
        results = self.model.track(
            video_path, conf=confidence, stream=True, verbose=False
        )

        frame_results = []
        for frame_idx, result in enumerate(results):
            if not result.keypoints:  # if no keypoints detected
                frame_results.append(FramePoseResult(persons=[], frame_idx=frame_idx))

            persons = []
            num_persons = result.keypoints.shape[0]
            num_keypoints = result.keypoints.shape[1]

            for i in range(num_persons):
                keypoints = []
                for j in range(num_keypoints):
                    xy = result.keypoints.xy
                    conf = result.keypoints.conf
                    kp = PoseKeypoint(
                        x=xy[i, j, 0],
                        y=xy[i, j, 1],
                        confidence=conf[i, j] if conf is not None else None,
                    )

                    keypoints.append(kp)
                persons.append(PersonPoseResult(keypoints=keypoints))
            frame_results.append(FramePoseResult(persons=persons, frame_idx=frame_idx))

        self.assert_frame_count_is_correct(frame_results, video_metadata)

        video_result = VideoPoseResult(
            fps=video_metadata.get("fps"),
            frame_width=video_metadata.get("width"),
            frame_height=video_metadata.get("height"),
            frames=frame_results,
            video_name=video_name,
        )
        return video_result
