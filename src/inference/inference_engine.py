import time
import json
import os
import glob
import numpy as np
from .pose_result import VideoPoseResult, PoseKeypoint, PersonPoseResult, FramePoseResult
from typing import Dict, List
from dataclasses import asdict

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class InferenceEngine:
    def __init__(self, dataset: dict, pose_estimators: list):
        self.dataset = dataset
        self.pose_estimators = pose_estimators
        self.base_output_path = "/output"
        self.estimator_point_pairs = dict()


    def estimate_pose_keypoints(self) -> Dict[str, Dict[str, List[VideoPoseResult]]]:
        results = {}
        for estimator in self.pose_estimators:
            results[estimator.name] = {}

            for video in self.dataset:
                print(f"Running estimator '{estimator.name}' on video {video.path}")

                start_time = time.time()
                try: # {model_name: {video_name: VideoPoseResult}}
                    video_pose_result = estimator.estimate_pose(video.path)
                    results[estimator.name][video.get_filename()] = video_pose_result
                    self.save_as_json(video_pose_result, video.get_filename(), estimator.name)
                except Exception as e:
                    raise Exception(f"Faced Exception: {e} on Video: {video.get_filename()} with Estimator: {estimator.name}")

                end_time = time.time()

                print(f"Inference time: '{estimator.name}': {end_time - start_time}")

        return results

    def save_as_json(self, video_pose_result: VideoPoseResult, video_name: str, estimator_name: str):
        json_folderpath = os.path.join(self.base_output_path, video_name)
        json_filepath = os.path.join(json_folderpath, f"{estimator_name}.json")
        os.makedirs(json_folderpath, exist_ok=True)
        
        serialized_video_pose_result = {
            "fps": video_pose_result.fps,
            "frame_width": video_pose_result.frame_width,
            "frame_height": video_pose_result.frame_height,
            "frames": [asdict(frame) for frame in video_pose_result.frames]
        }
        
        with open(json_filepath, "w+") as f:
            json.dump(serialized_video_pose_result, f, indent=2, cls=NumpyEncoder)
                    
    
    def load_pose_results_from_json(self) -> Dict[str, Dict[str, List[VideoPoseResult]]]:
        all_json_files = glob.glob(os.path.join(self.base_output_path, "*", "*.json")) # get all json files from output folder
        results = {}
        for json_file in all_json_files:
            _, video_name, estimator_name = json_file.split("/")
            estimator_name = estimator_name.split(".")[0]  # remove .json extension

            with open(json_file, "r") as f:
                data = json.load(f)
                frames = data.get("frames", [])
                frame_pose_results = []

                for frame in frames:
                    persons = frame.get("persons", [])
                    person_pose_results = []
                    for person in persons:
                        keypoints = person.get("keypoints", [])
                        pose_keypoints = [PoseKeypoint(x=k["x"], y=k["y"], confidence=k.get("confidence", None)) for k in keypoints]
                        person_pose_results.append(PersonPoseResult(keypoints=pose_keypoints))
                    frame_pose_results.append(FramePoseResult(persons=person_pose_results, frame_idx=frame.get("frame_idx", 0)))

                video_pose_result = VideoPoseResult(
                    fps=data.get("fps", None),
                    frame_width=data.get("frame_width", None),
                    frame_height=data.get("frame_height", None),
                    video_name=video_name,
                    frames=frame_pose_results
                )
                results[estimator_name][video_name] = video_pose_result

        return results