import time
from typing import Dict, List
import cv2
import os
import json

from inference import FramePoseResult, VideoPoseResult
from datasets import Dataset, VideoSample

COLORS = [ # these are colors that even color-blind people can see
    (0, 115, 178), # blue
    (204, 102, 0), # orange
    (0, 153, 127), # green-blue
    (242, 229, 64), # yellow
    (89, 178, 229), # cyan
    (204, 153, 178), # pink
    (229, 153, 0), # gold
]

class PoseRenderer():
    def __init__(self, dataset: Dataset, estimators_point_pairs: dict):
        self.dataset = dataset
        self.base_output_path = "/output"
        self.estimators_point_pairs = estimators_point_pairs

    
    def get_keypoints(self, output_path: str, estimator_name: str): 
        """ Fetch keypoints from json for specific video and model """
        file_path = os.path.join(output_path, estimator_name + ".json")
        with open(file_path) as f:
            keypoints = json.load(f)
        return keypoints

    
    def render_all_videos(self, pose_results: Dict[str, List[VideoPoseResult]]):
        """
        Render all videos in the dataset with the provided pose results.
        Args:
            pose_results (Dict[str, List[VideoPoseResult]]): Dictionary where keys are estimator names and values are lists of VideoPoseResult objects.
        """
        for idx, video in enumerate(self.dataset):
            start_time = time.time()
            output_path = os.path.join(self.base_output_path, video.get_filename())
            os.makedirs(output_path, exist_ok=True) # create folder if doesnt exist 

            video_pose_results = {estimator: pose_results[estimator][idx] for estimator in pose_results}
            self.render_video(video, video_pose_results, output_path)

            print(f"Rendering {video.path} - {time.time() - start_time}")
    

    def render_video(self, video: VideoSample, video_pose_results: Dict[str, VideoPoseResult], output_path: str):
        """
        Render video with keypoints and save it to output path.
        Args:
            video (VideoSample): The video sample to render.
            video_pose_results (Dict[str, VideoPoseResult]): Dictionary of pose results for each estimator.
            output_path (str): The path where the rendered video will be saved.
        """
        cap = cv2.VideoCapture(video.path) # load the video
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video.path}")
    
        fps = int(cap.get(cv2.CAP_PROP_FPS)) # get video specs 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writers = [] # initialize video writers 

        for estimator_name in self.estimators_point_pairs.keys(): # video writer for every model
            out = cv2.VideoWriter(f"{output_path}/{estimator_name}.mp4", fourcc, fps, (width, height))
            video_writers.append(out)


        frame_number = 0
        while cap.isOpened(): # for every frame
            ret, frame = cap.read()
            if not ret:
                break
            frame_copies = [frame.copy() for _ in range(len(video_writers))] # deep copy of frames to avoid overwriting
            model_idx = 0

            for idx, (estimator_name, model_points_pair) in enumerate(self.estimators_point_pairs.items()): # for every model
                frame_keypoints = video_pose_results[estimator_name].frames[frame_number]
                frame_copies[idx] = self.draw_keypoints(frame_copies[idx], frame_keypoints, model_points_pair, COLORS[model_idx]) # draw keypoints on frame
                video_writers[idx].write(frame_copies[idx]) # write rendered frame
                model_idx += 1

            frame_number += 1
    
        cap.release()
        for i in video_writers:
            i.release()
    
    def draw_keypoints(self, frame, frame_pose_result: FramePoseResult, point_pairs, color):
        """ Draw keypoints and join keypoint pairs on 1 frame """
        if not frame_pose_result.persons: # if this frame has no keypoints
            return frame  

        for person in frame_pose_result.persons: # every person
            if not person or not person.keypoints:
                continue

            for keypoint in person.keypoints: # every keypoint
                center = (int(keypoint.x), int(keypoint.y))
                cv2.circle(frame, center, 4, color, -1) # draw the keypoint
            
            for pair in point_pairs: # add lines between keypoints
                try:
                    point1 = (int(person.keypoints[pair[0]].x), int(person.keypoints[pair[0]].y))
                    point2 = (int(person.keypoints[pair[1]].x), int(person.keypoints[pair[1]].y))
                except IndexError:
                    print(person.keypoints)

                if point1 is None or point2 is None or \
                            point1[0] < 1 and point1[1] < 1 or \
                            point2[0] < 1 and point2[1] < 1:
                    continue
                cv2.line(frame, point1, point2, color=color, thickness=2) 
        
        return frame  



