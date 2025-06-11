import io
import json
import os
import pickle

import requests
import utils
from evaluation.pose_result import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult
from models.pose_estimator import PoseEstimator

class OpenPoseEstimator(PoseEstimator):
    def __init__(self, model_name: str, config: dict):
        """
        Initialize the OpenPoseEstimator with a model name and configuration.
        """
        super().__init__(model_name, config)
        

    def get_point_pairs(self):
        return [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16), (15, 19), (19, 20), (15, 21), (16, 22), (22, 23), (16, 24), (5, 17), (6, 17), (11, 12), (17, 18), (5, 6)]


    def estimate_pose(self, video_path: str) -> VideoPoseResult:
        """
        Estimate the pose of a video using Open pose estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            VideoPoseResult: A standardized result object containing the pose estimation results for the video.
        """
        pose_data, video_metadata = self._query_openpose_container(video_path) 
        video_pose_result = self._convert_to_video_pose_result(pose_data, video_metadata)
        return video_pose_result


    def _query_openpose_container(self, video_path: str):
        url = "http://openpose:8000/openpose/estimate-pose-on-video" # docker image link
        options = {"model_pose": "BODY_25B"} # config

        extension = os.path.splitext(video_path)[1].lower()
        if extension == ".mp4":
            mime_type = "video/mp4"
        elif extension == ".avi":
            mime_type = "video/x-msvideo"

        
        _ , video_metadata = utils.get_video_metadata(video_path) # get video metadata
        with open(video_path, "rb") as f: # only returns 1 person
            files = {'video': (f"video{extension}", f, mime_type)}
            frame ={"options": json.dumps(options)}

            response = requests.post(url, files=files, data=frame)
            if response.status_code == 200:
                buffer = io.BytesIO(response.content)
                pose_data = pickle.load(buffer)
            else:
                raise ValueError(f"Error in OpenPose API: {response.status_code} - {response.text}")
        return pose_data, video_metadata


    def _convert_to_video_pose_result(self, pose_data, video_metadata: dict) -> VideoPoseResult:
        frame_results = []
        for idx, frame in enumerate(pose_data): # every frame
            if frame and (frame.get("pose_keypoints").size > 0): # if data from frame or no pose detected
                keypoints = []
                for kp in frame.get("pose_keypoints"):
                    keypoints.append(PoseKeypoint(
                        x=kp[0],
                        y=kp[1],
                        confidence=kp[2]
                    ))
                person = PersonPoseResult(keypoints=keypoints)
                frame_results.append(FramePoseResult(persons=[person], frame_idx=idx)) # the MaskAnyone Openpose container only returns one person per frame
            else:
                frame_results.append(FramePoseResult(persons=[], frame_idx=idx))

        return VideoPoseResult(
            frames=frame_results,
            frame_width=video_metadata.get("width"),
            frame_height=video_metadata.get("height"),
            fps=video_metadata.get("fps"),
        )
