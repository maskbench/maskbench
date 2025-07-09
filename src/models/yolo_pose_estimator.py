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
            name (str): The name of the model (e.g. "YoloPose").
            config (dict): Configuration dictionary for the model. It must contain the key "weights" with the path to the weights file relative to the weights folder, otherwise it uses 'yolo11n-pose.pt'.
        """

        super().__init__(name, config)

        weights_file = self.config.get("weights", "yolo11n-pose.pt")
        print("Using weights file: ", weights_file)
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

        results = self.model.track(
            video_path, conf=self.confidence_threshold, stream=True, verbose=False
        )

        frame_results = []
        for frame_idx, frame_result in enumerate(results):
            if not frame_result.keypoints:  # if no keypoints detected
                frame_results.append(FramePoseResult(persons=[], frame_idx=frame_idx))
                continue

            xys = frame_result.keypoints.xy.cpu().numpy()
            confidences = frame_result.keypoints.conf

            if xys.size == 0: # if no persons detected
                frame_results.append(FramePoseResult(persons=[], frame_idx=frame_idx))
                continue

            persons = []
            num_persons = frame_result.keypoints.shape[0]
            num_keypoints = frame_result.keypoints.shape[1]

            for i in range(num_persons):
                keypoints = []
                for j in range(num_keypoints):
                    conf = float(confidences[i, j]) if (confidences is not None) and (xys[i, j, 0] != 0 and xys[i, j, 1] != 0) else None
                    kp = PoseKeypoint(
                        x=float(xys[i, j, 0]),
                        y=float(xys[i, j, 1]),
                        confidence=conf,
                    )

                    keypoints.append(kp)
                persons.append(PersonPoseResult(keypoints=keypoints))
            frame_results.append(FramePoseResult(persons=persons, frame_idx=frame_idx))


        video_pose_result = VideoPoseResult(
            fps=video_metadata.get("fps"),
            frame_width=video_metadata.get("width"),
            frame_height=video_metadata.get("height"),
            frames=frame_results,
            video_name=video_name,
        )

        self.assert_frame_count_is_correct(video_pose_result, video_metadata)
        video_pose_result = self.filter_low_confidence_keypoints(video_pose_result)
        return video_pose_result
