from typing import Dict, List
import cv2
import os
import logging
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

from pose_result_class import PersonPoseResult, VideoPoseResult
from datasets import Dataset, VideoSample
from checkpointer import Checkpointer
from utils import get_color_palette, get_video_metadata, parse_filename


class PoseRenderer:
    def __init__(self, dataset: Dataset, estimators_point_pairs: dict, checkpointer: Checkpointer, render_poses_only: bool = False, line_thickness: int = 6):
        self.dataset = dataset
        self.estimators_point_pairs = estimators_point_pairs
        self.checkpointer = checkpointer
        self.render_poses_only = render_poses_only
        self.line_thickness = line_thickness

    def render_all_videos(self, pose_results: Dict[str, Dict[str, List[VideoPoseResult]]], max_workers: int = None):
        """
        Render all videos in the dataset with the provided pose results.
        Args:
            pose_results (Dict[str, Dict[str, List[VideoPoseResult]]]): Dictionary where keys are estimator names and values are dictionaries mapping video names to lists of VideoPoseResult objects.
        """
        if max_workers is None:
            max_workers = mp.cpu_count()
        print(f"Rendering videos using {max_workers} workers.")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # add tasks - renders videos in parallel
            future_to_estimator = {}
            for video in self.dataset:
                video_name = video.get_filename()
                video_pose_results = {}
                for estimator in pose_results.keys():
                    if video_name not in pose_results[estimator]:
                        print(f"No pose results found for video {video_name} using estimator {estimator}. Skipping.")
                        logging.error(f"No pose results found for video {video_name} using estimator {estimator}. Skipping Rendering")
                        continue
                    video_pose_results[estimator] = pose_results[estimator][video_name]
                future = executor.submit(self.render_video, video, video_pose_results)
                future_to_estimator[future] = video
            
            # process result
            for future in as_completed(future_to_estimator):
                video = future_to_estimator[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Rendering video {video.get_filename()} generated an exception: {e}")
                    logging.exception(e)

    def render_video(
        self,
        video: VideoSample,
        video_pose_results: Dict[str, VideoPoseResult],
    ):
        """
        Render video with keypoints and save it to output path.
        Args:
            video (VideoSample): The video sample to render.
            video_pose_results (Dict[str, VideoPoseResult]): Dictionary of pose results for each estimator.
        """
        print(f"Rendering video {video.get_filename()}")
        video_name = video.get_filename()
        cap, video_metadata = get_video_metadata(video.path)
        fps = video_metadata["fps"]
        width = video_metadata["width"]
        height = video_metadata["height"]
        frame_count = video_metadata["frame_count"]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_writers = []  # initialize video writers
        video_name = video.get_filename()

        for estimator_name in self.estimators_point_pairs.keys():  # video writer for every model
            output_path = os.path.join(self.checkpointer.renderings_dir, video_name, f"{video_name}_{estimator_name}.mp4")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            video_writers.append((estimator_name, out))

        color_palette = get_color_palette()

        frame_number = 0
        while frame_number < frame_count:  # for every frame
            if self.render_poses_only:
                frame = np.zeros((height, width, 3), dtype=np.uint8)  # black frame
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            
            frame_copies = [
                frame.copy() for _ in range(len(video_writers))
            ]  # deep copy of frames to avoid overwriting

            for idx, (estimator_name, writer) in enumerate(video_writers):  # for every model
                try:
                    frame_keypoints = video_pose_results[estimator_name].frames[frame_number]
                    point_pairs = self.estimators_point_pairs[estimator_name]
                    # if we have both hand and body keypoints. we additionally draw hand.
                    if type(point_pairs) == tuple:
                        body_point_pairs, hand_point_pairs = point_pairs
                        frame_copies[idx] = self.draw_keypoints(
                            video_name,
                            frame_copies[idx],
                            frame_keypoints.hands,
                            hand_point_pairs,
                            self.hex_to_bgr(color_palette[idx]),
                        )
                        point_pairs = body_point_pairs

                    # default drawing with body keypoints
                    frame_copies[idx] = self.draw_keypoints(
                        video_name,
                        frame_copies[idx],
                        frame_keypoints.persons,
                        point_pairs,
                        self.hex_to_bgr(color_palette[idx]),
                    )  # draw keypoints on frame
                    writer.write(frame_copies[idx])  # write rendered frame
                except KeyError as e:
                    print(f"No pose results for estimator {estimator_name} in video {video_name}")
                    logging.error(f"No pose results for estimator {estimator_name} in video {video_name}")
                    writer.write(np.zeros_like(frame))  # write blank frame if exception occurs
                except IndexError as e:
                    print(f"{frame_number} is not in list, length of list is {len(video_pose_results[estimator_name].frames)}")
                    logging.error(f"Video: {video_name}, Estimator Name: {estimator_name}, frame {frame_number} is not in list, length of list is {len(video_pose_results[estimator_name].frames)}")                  
                    writer.write(np.zeros_like(frame))  # write blank frame if exception occurs
            
            frame_number += 1

        cap.release()
        for estimator_name, writer in video_writers:
            self.checkpointer.save_rendered_video(video_name, estimator_name, writer)

    def draw_keypoints(
        self, video_name: str, frame, frame_pose_result: List[PersonPoseResult], point_pairs, color
    ):
        """Draw keypoints and join keypoint pairs on 1 frame"""
        if not frame_pose_result:  # if this frame has no keypoints
            return frame

        # for hands - person 0 is and person 1 is hand 0 and hand 1
        for person in frame_pose_result:
            if not person or not person.keypoints: # if there are no keypoints for this person
                continue
            # Note: This is for 2D keypoints
            for keypoint in person.keypoints: # draw a circle for each keypoint if it exists
                if keypoint and keypoint.x is not None and keypoint.y is not None:
                    center = (int(keypoint.x), int(keypoint.y))
                    cv2.circle(frame, center, self.line_thickness, color, -1)
            
            for pair in point_pairs:  # iterate over point pairs to add lines between keypoints
                try: # some keypoints might be missing, which would lead to an IndexError
                    point1 = person.keypoints[pair[0]]
                    point2 = person.keypoints[pair[1]]
                except IndexError as e:
                    continue
                
                if (point1 is None) or (point2 is None) \
                    or (point1.x is None) or (point1.y is None) \
                    or (point2.x is None) or (point2.y is None):
                    continue

                point1 = (int(point1.x), int(point1.y))
                point2 = (int(point2.x), int(point2.y))
                cv2.line(frame, point1, point2, color=color, thickness=self.line_thickness)

        self.add_text_to_frame(frame, video_name)

        return frame
    
    def add_text_to_frame(self, frame, text, position=(5, 15)):
        """Add text to a frame at the specified position.
            Text can be split into multiple lines using the specified delimeter. Each line will be rendered below the previous one with a fixed spacing.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # white color
        thickness = 1
        corpus, speaker, clip_id, category, subtype, is_mirror = parse_filename(text).values()
        parts = {
            "Corpus": corpus,
            "Speaker": speaker,
            "Clip ID": clip_id,
            "Category": category,
            "Subtype": subtype,
        }

        # This is specific to envision gesture challenge
        for idx, (key, value) in enumerate(parts.items()):
            text_part = f"{key}: {value}"
            text_part_position = (position[0], position[1] + idx * 15)  # adjust y position for each part
            cv2.putText(frame, text_part, text_part_position, font, font_scale, color, thickness)
        return frame

    def hex_to_bgr(self, hex_color: str) -> tuple[int, int, int]:
        """Convert hex color to BGR tuple."""
        hex_color = hex_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bgr_color = rgb_color[::-1] # OpenCV uses BGR format, not RGB, therefore we need to reverse the tuple
        return bgr_color
