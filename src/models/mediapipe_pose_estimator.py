import os

import cv2
import mediapipe as mp

import utils
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from inference import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult
from models import PoseEstimator


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
          
        if os.path.exists(pre_built_weights_file_path): # if weights are pre-built
            weights_file_path = pre_built_weights_file_path
        elif os.path.exists(user_weights_file_path): # if weights are custom installed 
            weights_file_path = user_weights_file_path
        else: # if weight file not found
            raise ValueError(f"Could not find weights file under {weights_file_path}. Please download the weights from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models and place them in the weights folder.")
        
        device = 0
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=weights_file_path, delegate=device),
            running_mode=RunningMode.VIDEO, # informs model we will provide videos/ sequence of frames | adds temporal sequencing
            output_segmentation_masks=False,
        )

        
    def get_keypoint_pairs(self):
        return [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 19), (15, 17), (17, 19), (15, 21),
        (12, 14), (14, 16), (16, 20), (16, 18), (18, 20), (11, 23), (12, 24), (16, 22),
        (23, 25), (24, 26), (25, 27), (26, 28), (23, 24),
        (28, 30), (28, 32), (30, 32), (27, 29), (27, 31), (29, 31)
        ]


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
                for lm in person_landmarks: # for every keypoint
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    keypoints.append(PoseKeypoint(x=x, y=y))
            
                persons.append(PersonPoseResult(keypoints=keypoints)) 

            frame_results.append(FramePoseResult(persons=persons, frame_idx=frame_number))
            frame_number += 1

        cap.release()
        self.detector.close()

        return VideoPoseResult(
            fps=fps,
            frame_width=width,
            frame_height=height,
            frames=frame_results
        )


    def _execute_on_frame(self, frame, frame_number: int, fps: int):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp = int(((frame_number + 1) * 1_000_000 / fps))
        return self.detector.detect_for_video(mp_image, timestamp)
       