import logging
import numpy as np
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import parse_filename
from save_formats import SpecialFormat
from checkpointer import Checkpointer
from datasets import Dataset

class NpzFormat(SpecialFormat):
    def __init__(self, name: str, checkpointer: Checkpointer):
        super().__init__(name, checkpointer)
        self.dir = Path(self.checkpointer.checkpoint_dir) / "npz"

    def save_pose_results(self, dataset: Dataset, estimators: List[str], max_workers: int = None) -> None:
        if max_workers is None:
            max_workers = 20

        logging.info(f"Saving pose results in NPZ format using {max_workers} workers.")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # add tasks - saves NPZ files in parallel
            future_to_estimator = {}
            for video in dataset:
                for estimator in estimators:
                   future = executor.submit(self.save_npz, video.get_filename(), estimator.name)
                   future_to_estimator[future] = video
            
            # process result
            for future in as_completed(future_to_estimator):
                video = future_to_estimator[future]
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Saving NPZ for video {video.get_filename()} generated an exception: {e}")

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
    def save_npz(self, video_name: str, estimator_name: str) -> None:
        estimator_dir = self.dir / estimator_name
        estimator_dir.mkdir(parents=True, exist_ok=True)
        output_path = estimator_dir / f"{video_name}.npz"
        if output_path.exists():
            print(f"Output file {output_path} already exists. Skipping save in format {self.name}.")
            return
        
        if not self.checkpointer.exists(estimator_name, video_name):
            logging.error(f"No pose results found for video {video_name} using estimator {estimator_name}. Skipping save in format {self.name}.")
            return
        video_pose_result = self.checkpointer.load_pose_result(estimator_name, video_name)
        
        video_name = video_pose_result.video_name
        fps = video_pose_result.fps
        frame_width = video_pose_result.frame_width
        frame_height = video_pose_result.frame_height
        frames = video_pose_result.frames

        corpus, speaker, clip_id, category, subtype = parse_filename(video_name).values()

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