import os
import utils
import cv2
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from inference import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult
from models import PoseEstimator
from keypoint_pairs import COCO_KEYPOINT_PAIRS, MEDIAPIPE_KEYPOINT_PAIRS, COCO_TO_MEDIAPIPE

class MediaPipePoseEstimator(PoseEstimator):
    def __init__(self, name: str, config: dict):
        """
        Initialize the MediaPipePoseEstimator with a name and configuration.
        Args:
            estimator_name (str): The name of the estimator (e.g. "mediapipe_pose").
            config (dict): Configuration dictionary for the estimator. It must contain the key "weights" with the path to the weights file relative to the weights folder, otherwise it uses 'pose_landmarker_lite.task'.
            It can also contain the key "max_num_poses" with the maximum number of poses to detect, otherwise it uses 3.
        """

        super().__init__(name, config)

        weights_file = self.config.get("weights", "pose_landmarker_lite.task")
        print("Using weights file: ", weights_file)
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
            num_poses=self.config.get("max_num_poses", 3)
        )

    def get_keypoint_pairs(self):
        if self.config.get("save_keypoints_in_coco_format", False):
            return COCO_KEYPOINT_PAIRS
        else:
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
                frame_results.append(FramePoseResult(persons=[], frame_idx=frame_number))
            else:
                persons = []
                for person_landmarks in result.pose_landmarks:
                    keypoints = []

                    for lm in person_landmarks:
                        if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1): # for undetected keypoints, x and y can be outside the range [0, 1]
                            keypoints.append(PoseKeypoint(x=0, y=0, confidence=None)) # standardized handling of missing keypoints by setting x and y to 0
                            continue

                        x = lm.x * width # convert normalized landmarks to image coordinates
                        y = lm.y * height
                        keypoints.append(PoseKeypoint(x=x, y=y, confidence=lm.visibility))

                    persons.append(PersonPoseResult(keypoints=keypoints))

                frame_results.append(FramePoseResult(persons=persons, frame_idx=frame_number))
            frame_number += 1

        cap.release()
        self.detector.close()

        video_pose_result = VideoPoseResult(
            fps=fps,
            frame_width=width,
            frame_height=height,
            frames=frame_results,
            video_name=video_name,
        )

        self.assert_frame_count_is_correct(video_pose_result, video_metadata)
        video_pose_result = self.filter_low_confidence_keypoints(video_pose_result)
        if self.config.get("save_keypoints_in_coco_format", False):
            video_pose_result = utils.convert_keypoints_to_coco_format(video_pose_result, COCO_TO_MEDIAPIPE)
        return video_pose_result

    def _execute_on_frame(self, frame, frame_number: int, fps: int):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp = int(((frame_number + 1) * 1_000_000 / fps))
        return self.detector.detect_for_video(mp_image, timestamp)
