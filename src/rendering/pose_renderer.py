import time
from typing import Dict, List
import cv2
import os

from inference import FramePoseResult, VideoPoseResult
from datasets import Dataset, VideoSample
from checkpointer import Checkpointer
from utils import get_color_palette


class PoseRenderer:
    def __init__(self, dataset: Dataset, estimators_point_pairs: dict, checkpointer: Checkpointer, line_thickness: int = 6):
        self.dataset = dataset
        self.estimators_point_pairs = estimators_point_pairs
        self.checkpointer = checkpointer
        self.line_thickness = line_thickness

    def render_all_videos(self, pose_results: Dict[str, Dict[str, List[VideoPoseResult]]]):
        """
        Render all videos in the dataset with the provided pose results.
        Args:
            pose_results (Dict[str, Dict[str, List[VideoPoseResult]]]): Dictionary where keys are estimator names and values are dictionaries mapping video names to lists of VideoPoseResult objects.
        """
        for video in self.dataset:
            start_time = time.time()
            video_name = video.get_filename()

            video_pose_results = {
                estimator: pose_results[estimator][video_name] for estimator in pose_results.keys()
            }
            self.render_video(video, video_pose_results)

            time_taken = time.time() - start_time
            print(f"Rendered {video.path} - {time_taken:.2f} seconds")

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
        cap = cv2.VideoCapture(video.path)  # load the video
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video.path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))  # get video specs
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        while cap.isOpened():  # for every frame
            ret, frame = cap.read()
            if not ret:
                break
            frame_copies = [
                frame.copy() for _ in range(len(video_writers))
            ]  # deep copy of frames to avoid overwriting

            for idx, (estimator_name, writer) in enumerate(video_writers):  # for every model
                try:
                    frame_keypoints = video_pose_results[estimator_name].frames[frame_number]
                    frame_copies[idx] = self.draw_keypoints(
                        frame_copies[idx],
                        frame_keypoints,
                        self.estimators_point_pairs[estimator_name],
                        self.hex_to_bgr(color_palette[idx]),
                    )  # draw keypoints on frame
                except IndexError as e:
                    print(f"{frame_number} is not in list, length of list is {len(video_pose_results[estimator_name].frames)}")
                writer.write(frame_copies[idx])  # write rendered frame

            frame_number += 1

        cap.release()
        for estimator_name, writer in video_writers:
            self.checkpointer.save_rendered_video(video_name, estimator_name, writer)

    def draw_keypoints(
        self, frame, frame_pose_result: FramePoseResult, point_pairs, color
    ):
        """Draw keypoints and join keypoint pairs on 1 frame"""
        if not frame_pose_result.persons:  # if this frame has no keypoints
            return frame

        for person in frame_pose_result.persons:
            if not person or not person.keypoints:
                continue

            for keypoint in person.keypoints: # draw a circle for each keypoint if it exists
                if keypoint: 
                    center = (int(keypoint.x), int(keypoint.y))
                    cv2.circle(frame, center, 6, color, -1)
                
            for pair in point_pairs:  # iterate over point pairs to add lines between keypoints
                try: # some keypoints might be missing, which would lead to an IndexError
                    point1 = person.keypoints[pair[0]]
                    point2 = person.keypoints[pair[1]]
                except IndexError as e:
                    continue
                
                if (point1 is None) or (point2 is None) or \
                    ((point1.x <= 0) and (point1.y <= 0)) or ((point2.x <= 0) and (point2.y <= 0)):
                    continue

                point1 = (int(point1.x), int(point1.y))
                point2 = (int(point2.x), int(point2.y))
                cv2.line(frame, point1, point2, color=color, thickness=self.line_thickness)

        return frame

    def hex_to_bgr(self, hex_color: str) -> tuple[int, int, int]:
        """Convert hex color to BGR tuple."""
        hex_color = hex_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bgr_color = rgb_color[::-1] # OpenCV uses BGR format, not RGB, therefore we need to reverse the tuple
        return bgr_color
