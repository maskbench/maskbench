import logging

import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, List

from utils import parse_filename
from pose_result_class import VideoPoseResult
from save_formats import SpecialFormat
from checkpointer import Checkpointer


class NpzFormat(SpecialFormat):
    def __init__(self, name: str, checkpointer: Checkpointer):
        super().__init__(name, checkpointer)
        self.dir = Path(self.checkpointer.checkpoint_dir) / "npz"

    def create(self, pose_results: Dict[str, Dict[str, List[VideoPoseResult]]]) -> None:
        for estimator_name, videos in pose_results.items():
            estimator_dir = self.dir / estimator_name
            estimator_dir.mkdir(parents=True, exist_ok=True)

            progress_bar = tqdm.tqdm(total=len(videos), desc=f"Saving NPZ for {estimator_name}", unit="videos")
            print()

            for video_name, video_pose_results in videos.items():
                output_path = estimator_dir / f"{video_name}.npz"
                self.save_npz(video_pose_results, output_path)
                progress_bar.update(1)
                logging.info(progress_bar.__str__())
            progress_bar.close()
            print()

    def extract_hand_landmarks(self, hand_landmarks, dimensions: int):
        num_keypoints = next(
            (len(hand.keypoints) for frame in hand_landmarks 
                if frame 
                for hand in frame if hand.keypoints),
            21
        ) # 21 is default -- mediapipe hand keypoints
        nan_kps = [[np.nan] * dimensions for _ in range(num_keypoints)]
        coords = ['x', 'y', 'z'][:dimensions]

        left_hand_landmarks = []
        right_hand_landmarks = []
        for frame in hand_landmarks:
            for side, hand_id in [(left_hand_landmarks, 0), (right_hand_landmarks, 1)]:
                kps = [
                    [getattr(kp, coord, np.nan) for coord in coords]
                     for hand in frame
                     for kp in hand.keypoints
                     if kp.hand == hand_id
                ] if frame else []
                side.append(kps if kps else nan_kps)

        return np.array(left_hand_landmarks, dtype=float), np.array(right_hand_landmarks, dtype=float)

    def extract_body_landmarks(self, body_landmarks, dimensions: int):
        num_keypoints = next(
            (len(body.keypoints) for frame in body_landmarks 
                if frame 
                for body in frame if body.keypoints),
            33
        ) # 33 is default -- mediapipe body keypoints
        nan_kps = [[np.nan] * dimensions for _ in range(num_keypoints)]
        coords = ['x', 'y', 'z'][:dimensions]

        body_landmarks = [
            [
                [getattr(kp, coord, np.nan) for coord in coords]
                for kp in frame[0].keypoints
            ]
            if (frame and frame[0] and frame[0].keypoints) else nan_kps
            for frame in body_landmarks
        ]
            
        return np.array(body_landmarks, dtype=float)

    # This is specific to envision gesture challenge, we can modify this to be more general if needed
    def save_npz(self, video_pose_result: VideoPoseResult, output_path: Path) -> None:
        if output_path.exists():
            print(f"Output file {output_path} already exists. Skipping save in format {self.name}.")
            logging.info(f"Output file {output_path} already exists. Skipping save in format {self.name}.")
            return
        
        video_name = video_pose_result.video_name
        fps = video_pose_result.fps
        frame_width = video_pose_result.frame_width
        frame_height = video_pose_result.frame_height
        frames = video_pose_result.frames

        corpus, speaker, clip_id, category, subtype, is_mirror = parse_filename(video_name).values()

        persons_world_landmark = [frame.persons_world_landmark for frame in frames] # 3d
        hand_world_landmark = [frame.hands_world_landmark for frame in frames] # 3d
        hands = [frame.hands for frame in frames] # 2d
        persons = [frame.persons for frame in frames] # 2d

        # This assumes single person video
        # we convert None/ NULL to nan for convinience
        # frame[0] represents person[0] for hand landmarks
        world_left_hand_landmarks, world_right_hand_landmarks = self.extract_hand_landmarks(hand_world_landmark, 3)
        image_left_hand_landmarks, image_right_hand_landmarks = self.extract_hand_landmarks(hands, 2)
        world_landmarks_array = self.extract_body_landmarks(persons_world_landmark, 3)
        image_landmarks_array = self.extract_body_landmarks(persons, 2)

        np.savez(
            output_path,
            video_name=video_name,
            is_mirror=is_mirror,
            corpus=corpus,
            speaker=speaker,
            clip_id=clip_id,
            category=category,
            subtype=subtype,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            world_body_landmarks=world_landmarks_array,
            image_body_landmarks=image_landmarks_array,
            world_left_hand_landmarks=world_left_hand_landmarks,
            world_right_hand_landmarks=world_right_hand_landmarks,
            image_left_hand_landmarks=image_left_hand_landmarks,
            image_right_hand_landmarks=image_right_hand_landmarks
        )
        
        return