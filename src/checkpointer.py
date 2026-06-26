import os
import json
import datetime
import shutil
import subprocess
import numpy as np
import logging
import cv2 as cv
from typing import Dict, Optional
from filelock import FileLock
from tqdm import tqdm

from pose_result_class import VideoPoseResult


# ── inference_times.json schema ───────────────────────────────────────────────
# Current on-disk schema (version 1):
#   {
#     "schema_version": 1,
#     "metadata": {"total_videos": int, "total_time_taken": float},
#     "estimators": {<estimator>: {<video>: seconds}},
#     "videos_processed_per_estimator": {<estimator>: int},
#     "total_time_per_estimator": {<estimator>: float},
#   }
# Two earlier shapes exist on disk and must be read without crashing or corrupting:
#   legacy-flat: {<estimator>: {<video>: seconds}}            (no bookkeeping keys)
#   mixed:       bookkeeping keys + <estimator> keys at the top level, no "estimators"
INFERENCE_TIMES_SCHEMA_VERSION = 1
_RESERVED_INFERENCE_KEYS = {
    "schema_version", "metadata", "estimators",
    "videos_processed_per_estimator", "total_time_per_estimator",
}


def estimator_timings(data: dict) -> dict:
    """Return ``{estimator: {video: seconds}}`` from any inference_times shape.

    Normalises the three on-disk shapes (nested / mixed / legacy-flat) so callers
    that iterate estimators never see the reserved bookkeeping keys.
    """
    if "estimators" in data:                                  # nested (current)
        return data["estimators"]
    if any(k in data for k in _RESERVED_INFERENCE_KEYS):      # mixed (strip reserved)
        return {k: v for k, v in data.items() if k not in _RESERVED_INFERENCE_KEYS}
    return data                                               # legacy-flat


def migrate_inference_times(data: dict, total_videos: int) -> dict:
    """Normalise any inference_times shape (incl. ``{}``) to the current schema.

    Idempotent and non-destructive: roll-ups are recomputed from the estimator
    timings, never assumed, so re-migrating a current file is a no-op.
    """
    if "estimators" in data and "metadata" in data:           # already current
        data.setdefault("schema_version", INFERENCE_TIMES_SCHEMA_VERSION)
        data.setdefault("videos_processed_per_estimator", {})
        data.setdefault("total_time_per_estimator", {})
        return data
    est = estimator_timings(data)                             # excludes reserved keys
    return {
        "schema_version": INFERENCE_TIMES_SCHEMA_VERSION,
        "metadata": {
            "total_videos": total_videos,
            "total_time_taken": sum(sum(v.values()) for v in est.values()),
        },
        "estimators": est,
        "videos_processed_per_estimator": {e: len(v) for e, v in est.items()},
        "total_time_per_estimator": {e: sum(v.values()) for e, v in est.items()},
    }


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
    def __init__(self, dataset_name: str, total_videos: int, checkpoint_name: Optional[str] = None):
        """
        Initialize the Checkpointer.
        
        Args:
            dataset_name (str): Name of the dataset being processed
            total_videos (int): Total number of videos in the dataset
            checkpoint_name (Optional[str]): Name of checkpoint to load (format: datasetname-date-time)
        """
        self.dataset_name = dataset_name
        self.base_output_path = "/output"
        self.total_videos = total_videos

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

        # add ffmpeg command to ensure correct encoding and metadata
        temp_output_path = os.path.join(video_dir, f"{video_name}_{estimator_name}_temp.mp4")
        command = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-hide_banner",
            "-loglevel", "error",
            "-i", output_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "128k",
            temp_output_path
        ]
        subprocess.run(command, check=True)
        os.replace(temp_output_path, output_path)  # replace original file with re-encoded file
        
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
        
        # Load existing inference times (any on-disk shape) or start fresh, then
        # normalise to the current schema before updating. migrate_* handles the
        # legacy-flat / mixed / empty cases so this never KeyErrors on resume.
        with lock:
            data = {}
            if os.path.exists(inference_file_path):
                with open(inference_file_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse {inference_file_path}; rebuilding it.")
                        data = {}
            inference_times = migrate_inference_times(data, self.total_videos)

            estimators = inference_times["estimators"]
            estimators.setdefault(estimator_name, {})
            inference_times["videos_processed_per_estimator"].setdefault(estimator_name, 0)
            inference_times["total_time_per_estimator"].setdefault(estimator_name, 0.0)

            if video_name in estimators[estimator_name]:
                print(f"Warning: Overwriting existing inference time for {estimator_name} on {video_name}")
                logging.warning(f"Overwriting existing inference time for {estimator_name} on {video_name}")
                old_time = estimators[estimator_name][video_name]
                inference_times["total_time_per_estimator"][estimator_name] -= old_time
                inference_times["metadata"]["total_time_taken"] -= old_time
                inference_times["videos_processed_per_estimator"][estimator_name] -= 1

            estimators[estimator_name][video_name] = inference_time # add new inference time
            inference_times["total_time_per_estimator"][estimator_name] += inference_time
            inference_times["metadata"]["total_time_taken"] += inference_time
            inference_times["videos_processed_per_estimator"][estimator_name] += 1

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

            progress_bar = tqdm(os.listdir(estimator_dir), desc=f"Loading pose results for {estimator_name}", unit="file")
            
            for pose_file in os.listdir(estimator_dir):
                if not pose_file.endswith("_poses.json"):
                    continue
                    
                video_name = pose_file.replace("_poses.json", "")
                json_path = os.path.join(estimator_dir, pose_file)
                video_pose_result = VideoPoseResult.from_json(json_path, video_name)
                results[estimator_name][video_name] = video_pose_result
                progress_bar.update(1)
                    
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
            try:
                inference_times = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Could not parse {inference_file_path}; returning empty inference times.")
                return {}

        # Return only the estimator -> {video: seconds} map, normalised across all
        # on-disk shapes, so consumers (visualizer, plots) never see bookkeeping keys.
        return estimator_timings(inference_times)
