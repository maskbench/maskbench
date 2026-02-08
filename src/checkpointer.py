import os
import json
import datetime
import shutil
import numpy as np
import logging
import cv2 as cv
from typing import Dict, Optional
from filelock import FileLock

from inference.pose_result import VideoPoseResult

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class Checkpointer:
    def __init__(self, dataset_name: str, checkpoint_name: Optional[str] = None):
        """
        Initialize the Checkpointer.
        
        Args:
            dataset_name (str): Name of the dataset being processed
            load_checkpoint (Optional[str]): Name of checkpoint to load (format: datasetname-date-time)
        """
        self.dataset_name = dataset_name
        self.base_output_path = "/output"
        
        if checkpoint_name != None: # load existing checkpoint
            self.load_checkpoint = True
            self.checkpoint_dir = os.path.join(self.base_output_path, checkpoint_name)
            if not os.path.exists(self.checkpoint_dir):
                raise ValueError(f"Checkpoint directory {self.checkpoint_dir} does not exist")
        else: # create new checkpoint
            self.load_checkpoint = False
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.checkpoint_dir = os.path.join(self.base_output_path, f"{dataset_name}-{current_time}")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        # Create subdirectories
        self.poses_dir = os.path.join(self.checkpoint_dir, "poses")
        self.plots_dir = os.path.join(self.checkpoint_dir, "plots")
        self.renderings_dir = os.path.join(self.checkpoint_dir, "renderings")
        
    def save_rendered_video(self, video_name: str, estimator_name: str, video_writer: cv.VideoWriter) -> str:
        """
        Save a rendered video for a specific estimator.
        
        Args:
            video_name (str): Name of the video being rendered
            estimator_name (str): Name of the pose estimator (e.g., 'Yolo', 'Mediapipe')
            video_writer: OpenCV VideoWriter object with the rendered video
            
        Returns:
            str: Path where the video was saved
        """
        video_dir = os.path.join(self.renderings_dir, video_name)
        os.makedirs(video_dir, exist_ok=True)
        
        output_path = os.path.join(video_dir, f"{video_name}_{estimator_name}.mp4")

        video_writer.release()
        
        return output_path
        
    def save_video_pose_result(self, video_pose_result: VideoPoseResult, estimator_name: str) -> str:
        """
        Save pose estimation results for a video.
        
        Args:
            video_pose_result (VideoPoseResult): The pose estimation results to save
            estimator_name (str): Name of the pose estimator (e.g., 'Yolo', 'Mediapipe')
            
        Returns:
            str: Path where the results were saved
        """
        os.makedirs(self.poses_dir, exist_ok=True)

        estimator_dir = os.path.join(self.poses_dir, estimator_name)
        os.makedirs(estimator_dir, exist_ok=True)
        
        output_path = os.path.join(estimator_dir, f"{video_pose_result.video_name}_poses.json")
        
        with open(output_path, "w+") as f:
            json.dump(video_pose_result.to_json(), f, indent=2, cls=NumpyEncoder)
            
        return output_path

    def save_inference_time(self, estimator_name: str, video_name: str, inference_time: float) -> str:
        """
        Save the inference time for a specific estimator and video.
        """
        inference_file_path = os.path.join(self.checkpoint_dir, "inference_times.json")
        lock = FileLock(inference_file_path + ".lock")  # to prevent concurrent access
        
        # Load existing inference times or create new dict if file doesn't exist
        with lock:
            if os.path.exists(inference_file_path):
                with open(inference_file_path, 'r') as f:
                    inference_times = json.load(f)
            else:
                inference_times = {}

            if estimator_name not in inference_times:
                inference_times[estimator_name] = {}
                
            inference_times[estimator_name][video_name] = inference_time # add new inference time
            
            with open(inference_file_path, 'w') as f:
                json.dump(inference_times, f, indent=4)
                
            print(f"Inference time for {estimator_name} on {video_name}: {inference_time:.3f}s")

    def save_config(self, config_file_path: str):
        """
        Copies the config file to the checkpoint directory.
        """
        config_file_name = os.path.basename(config_file_path)
        shutil.copy(config_file_path, os.path.join(self.checkpoint_dir, config_file_name))

    def load_pose_results(self, pose_estimator_names: list[str]) -> Dict[str, Dict[str, VideoPoseResult]]:
        """
        Load all pose results from the checkpoint.
        
        Returns:
            Dict[str, Dict[str, VideoPoseResult]]: Dictionary mapping estimator names to dictionaries
            mapping video names to their pose results.
        """
        if not os.path.exists(self.poses_dir):
            print(f"No pose results found in checkpoint {self.checkpoint_dir}. Will run all models again.")
            logging.error("No pose results found in checkpoint %s. Will run all models again.", self.checkpoint_dir)
            return {}
            
        results = {}
        
        for estimator_name in pose_estimator_names:
            if estimator_name not in os.listdir(self.poses_dir):
                print(f"No pose results found for estimator {estimator_name} in checkpoint {self.checkpoint_dir}. Will run model again.")
                logging.error(f"No pose results found for estimator {estimator_name} in checkpoint {self.checkpoint_dir}. Will run model again.")
                continue

            estimator_dir = os.path.join(self.poses_dir, estimator_name)
            results[estimator_name] = {}
            
            for pose_file in os.listdir(estimator_dir):
                if not pose_file.endswith("_poses.json"):
                    continue
                    
                video_name = pose_file.replace("_poses.json", "")
                json_path = os.path.join(estimator_dir, pose_file)
                video_pose_result = VideoPoseResult.from_json(json_path, video_name)
                results[estimator_name][video_name] = video_pose_result
                    
        return results 

    def load_inference_times(self) -> Dict[str, Dict[str, float]]:
        """
        Load all inference times from the checkpoint.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping estimator names to
            video names to their inference times in seconds.
        """
        inference_file_path = os.path.join(self.checkpoint_dir, "inference_times.json")
        
        if not os.path.exists(inference_file_path):
            print(f"No inference times found in checkpoint {self.checkpoint_dir}. Skipping inference time plot.")
            return {}
            
        with open(inference_file_path, 'r') as f:
            inference_times = json.load(f)
            
        return inference_times
