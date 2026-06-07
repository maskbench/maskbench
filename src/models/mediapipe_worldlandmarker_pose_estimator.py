import os
import utils
import cv2
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from pose_result_class import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult
from models import PoseEstimator
from keypoint_pairs import MEDIAPIPE_KEYPOINT_PAIRS, MEDIAPIPE_HAND_KEYPOINT_PAIRS

class MediaPipeWorldLandmarkerPoseEstimator(PoseEstimator):
    def __init__(self, name: str, config: dict):
        """
        Initialize the MediaPipeWorldLandmarkerPoseEstimator with a name and configuration.
        Args:
            estimator_name (str): The name of the estimator (e.g. "mediapipe_worldlandmarker").
            config (dict): Configuration dictionary for the estimator. It must contain the key "body_weights" with the path to the body weights file relative to the weights folder, and "hand_weights" with the path to the hand weights file, otherwise it uses 'pose_landmarker_lite.task'.
            It can also contain the key "max_num_poses" with the maximum number of poses to detect, otherwise it uses 3.
        """

        super().__init__(name, config)

        body_weights_file = self.config.get("body_weights", "pose_landmarker_lite.task")
        print("Mediapipe: Using body weights file: ", body_weights_file)
        pre_built_body_weights_file_path = os.path.join("/weights/pre_built", body_weights_file)

        hand_weights_file = self.config.get("hand_weights", "mediapipe_hand_landmarker.task")
        print("Mediapipe: Using hand weights file: ", hand_weights_file)
        pre_built_hand_weights_file_path = os.path.join("/weights/pre_built", hand_weights_file)

        if os.path.exists(pre_built_body_weights_file_path) and os.path.exists(pre_built_hand_weights_file_path):
            body_weights_file_path = pre_built_body_weights_file_path
            hand_weights_file_path = pre_built_hand_weights_file_path
        else:
            raise ValueError(
                f"Could not find weights file {body_weights_file} or {hand_weights_file}. Please download the weights from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models and place them in the weights folder."
            )

        device = 0
        self.body_options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=body_weights_file_path, delegate=device
            ),
            running_mode=RunningMode.VIDEO,  # informs model we will provide videos/ sequence of frames | adds temporal sequencing
            output_segmentation_masks=False,
            num_poses=self.config.get("max_num_poses", 3)
        )
        self.hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=hand_weights_file_path, delegate=device
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )

    def get_keypoint_pairs(self):
        return (MEDIAPIPE_KEYPOINT_PAIRS, MEDIAPIPE_HAND_KEYPOINT_PAIRS)

    def estimate_pose(self, video_path: str) -> VideoPoseResult:
        """
        Estimate the pose of a video using MediaPipe pose estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            VideoPoseResult: A standardized result object containing the pose estimation results for the video.
        """
        self.body_detector = PoseLandmarker.create_from_options(self.body_options)
        self.hand_detector = HandLandmarker.create_from_options(self.hand_options)

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
            mp_image, timestamp = self.format_frame(frame, frame_number, fps)
            body_persons, body_persons_world = self.detect_body(mp_image, timestamp, width, height)
            hand_persons, hand_persons_world = self.detect_hands(mp_image, timestamp, width, height)

            frame_results.append(FramePoseResult(persons=body_persons, persons_world_landmark=body_persons_world, hands=hand_persons, hands_world_landmark=hand_persons_world, frame_idx=frame_number))
            frame_number += 1

        cap.release()
        self.body_detector.close()
        self.hand_detector.close()
        
        video_pose_result = VideoPoseResult(
            fps=fps,
            frame_width=width,
            frame_height=height,
            frames=frame_results,
            video_name=video_name,
        )

        self.assert_frame_count_is_correct(video_pose_result, video_metadata)
        video_pose_result = self.filter_low_confidence_keypoints(video_pose_result)
        return video_pose_result

    def detect_body(self, mp_image, timestamp, width, height):
        # Required for 2D rendering
        result = self.body_detector.detect_for_video(mp_image, timestamp)
        persons = []
        if result.pose_landmarks:
            for person_landmarks in result.pose_landmarks:
                keypoints = []

                for lm in person_landmarks:
                    if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1): # for undetected keypoints, x and y can be outside the range [0, 1]
                        keypoints.append(PoseKeypoint(x=None, y=None, z=None, confidence=None)) # standardized handling of missing keypoints by setting x and y to None
                        continue

                    x = lm.x * width # convert normalized landmarks to image coordinates
                    y = lm.y * height
                    keypoints.append(PoseKeypoint(x=x, y=y, confidence=lm.visibility))

                persons.append(PersonPoseResult(keypoints=keypoints))
        
        # 3D World Landmarks
        persons_world_landmark = []
        if result.pose_world_landmarks:
            for person_landmarks in result.pose_world_landmarks:
                keypoints = []
                for lm in person_landmarks:
                    keypoints.append(PoseKeypoint(x=lm.x, y=lm.y, z=lm.z, confidence=lm.visibility))

                persons_world_landmark.append(PersonPoseResult(keypoints=keypoints))
        
        return persons, persons_world_landmark

    def detect_hands(self, mp_image, timestamp, width, height):
        result = self.hand_detector.detect_for_video(mp_image, timestamp)
        persons = []
        # Note: person[0] is hand[0] and person[1] is hand[1] -- handlandmarker is single person
        # we keep the same name for easier adaption
        # Required for 2D rendering
        if result.hand_landmarks:
            seen_hands = set() # to track which hands have been seen in the current frame -- mediapipe sometimes returns duplicate hand detections
            for i, person_landmarks in enumerate(result.hand_landmarks):
                hand_label = None
                hand_score = None
                if result.handedness and i < len(result.handedness):
                    hand_label = 0 if result.handedness[i][0].category_name == "Left" else 1
                    hand_score = result.handedness[i][0].score  # confidence score for the hand detection
                if hand_label in seen_hands:
                    continue
                seen_hands.add(hand_label)
                
                keypoints = []
                for lm in person_landmarks:
                    if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1): # for undetected keypoints, x and y can be outside the range [0, 1]
                        keypoints.append(PoseKeypoint(x=None, y=None, z=None, hand=hand_label, confidence=None)) # standardized handling of missing keypoints by setting x and y to None
                        continue

                    x = lm.x * width # convert normalized landmarks to image coordinates
                    y = lm.y * height
                    # Note: hand landmark does not provide visibility. score is same for all keypoints
                    keypoints.append(PoseKeypoint(x=x, y=y, hand=hand_label, confidence=hand_score))

                persons.append(PersonPoseResult(keypoints=keypoints))
        
        # 3D World Landmarks
        persons_world_landmark = []
        if result.hand_world_landmarks:
            seen_hands = set() # to track which hands have been seen in the current frame -- mediapipe sometimes returns duplicate hand detections
            for i, person_landmarks in enumerate(result.hand_world_landmarks):
                hand_label = None
                hand_score = None
                if result.handedness and i < len(result.handedness):
                    hand_label = 0 if result.handedness[i][0].category_name == "Left" else 1
                    hand_score = result.handedness[i][0].score  # confidence score for the hand detection
                if hand_label in seen_hands:
                    continue
                seen_hands.add(hand_label)

                keypoints = []
                for lm in person_landmarks:
                    keypoints.append(PoseKeypoint(x=lm.x, y=lm.y, z=lm.z, hand=hand_label, confidence=hand_score))

                persons_world_landmark.append(PersonPoseResult(keypoints=keypoints))
        return persons, persons_world_landmark

    def format_frame(self, frame, frame_number: int, fps: int):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp = int(((frame_number + 1) * 1_000_000 / fps))
        return mp_image, timestamp
