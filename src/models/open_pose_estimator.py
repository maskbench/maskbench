from models.pose_estimator import PoseEstimator
import requests
import pickle
import json
import io
import os

class OpenPoseEstimator(PoseEstimator):
    def __init__(self, model_name: str, config: dict):
        """
        Initialize the OpenPoseEstimator with a model name and configuration.
        """
        
    def get_pair_points(self):
        return [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16), (15, 19), (19, 20), (15, 21), (16, 22), (22, 23), (16, 24), (5, 17), (6, 17), (11, 12), (17, 18), (5, 6)]

    def estimate_pose(self, video_path: str) -> list:
        """
        Estimate the pose of a video using Open pose estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            list: A list of lists containing the keypoints for each frame.
        """
        url = "http://openpose:8000/openpose/estimate-pose-on-video" # docker image link
        options = {"model_pose": "BODY_25B"} # config
        key_points_list = []

        extension = os.path.splitext(video_path)[1].lower()
        if extension == ".mp4":
            mime_type = "video/mp4"
        elif extension == ".avi":
            mime_type = "video/x-msvideo"
        
        with open(video_path, "rb") as f: # only returns 1 person
            files = {'video': (f"video{extension}", f, mime_type)}
            data ={ "options": json.dumps(options) }

            response = requests.post(url, files=files, data=data)
            if response.status_code == 200:
                buffer = io.BytesIO(response.content)
                pose_data = pickle.load(buffer)                
            else:
                raise ValueError(f"Error in OpenPose API: {response.status_code} - {response.text}")

        for data in pose_data: # every frame
            if data and (data.get("pose_keypoints").size > 0): # if no data from frame or no pose detected
                keypoints = data.get("pose_keypoints") # removes face, left and right hand which we are not interested in
                keypoints = keypoints[:,:2].astype(int).tolist() # converted to int and removed confidence score
                key_points_list.append([keypoints]) # kept as list since only 1 person detected while render expects multiple people
            else:
                key_points_list.append([])

        return key_points_list