import os

import cv2
import mediapipe as mp

import utils
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from inference import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult
from models import PoseEstimator
from keypoint_pairs import MEDIAPIPE_KEYPOINT_PAIRS

class MediaPipePoseEstimator(PoseEstimator):
    def __init__(self, name: str, config: dict):
        """
        Initialize the MediaPipePoseEstimator with a name and configuration.
        Args:
            estimator_name (str): The name of the estimator (e.g. "mediapipe_pose").
            config (dict): Configuration dictionary for the estimator. It must contain the key "weights" with the path to the weights file relative to the weights folder.
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
                f"Could not find weights file {weights_file}. Please download the weights from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models and place them in the weights folder."
            )

        device = 0
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=weights_file_path, delegate=device
            ),
            running_mode=RunningMode.VIDEO,  # informs model we will provide videos/ sequence of frames | adds temporal sequencing
            output_segmentation_masks=False,
        )

    def get_keypoint_pairs(self):
        return MEDIAPIPE_KEYPOINT_PAIRS

    def estimate_pose(self, video_path: str) -> VideoPoseResult:
        """
        Estimate the pose of a video using MediaPipe pose estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            VideoPoseResult: A standardized result object containing the pose estimation results for the video.
        """
        self.detector = PoseLandmarker.create_from_options(self.options)

        cap, video_metadata = utils.get_video_metadata(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        width = video_metadata.get("width")
        height = video_metadata.get("height")
        fps = video_metadata.get("fps")

        frame_number = 0
        frame_results = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = self._execute_on_frame(frame, frame_number, fps)

            if not result.pose_landmarks:
                continue

            persons = []
            for person_landmarks in result.pose_landmarks:
                keypoints = []

                # we only extract x, y and ignore z, visibility and presence
                # we also convert normalized landmarks to image coordinates
                for lm in person_landmarks:  # for every keypoint
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    keypoints.append(PoseKeypoint(x=x, y=y))

                persons.append(PersonPoseResult(keypoints=keypoints))

            frame_results.append(
                FramePoseResult(persons=persons, frame_idx=frame_number)
            )
            frame_number += 1

        cap.release()
        self.detector.close()

        return VideoPoseResult(
            fps=fps,
            frame_width=width,
            frame_height=height,
            frames=frame_results,
            video_name=video_name,
        )

    def _execute_on_frame(self, frame, frame_number: int, fps: int):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp = int(((frame_number + 1) * 1_000_000 / fps))
        return self.detector.detect_for_video(mp_image, timestamp)
