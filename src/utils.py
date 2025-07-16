import subprocess
import cv2
import glob
import os
import json
from typing import List
from inference import FramePoseResult, PersonPoseResult, PoseKeypoint

def convert_keypoints_to_coco_format(frame_results: List[FramePoseResult], model_to_coco_mapping: list) -> List[FramePoseResult]:
    for frame in frame_results: 
        if frame and frame.persons: # ensuring they are not None
            for person in frame.persons:
                if len(person.keypoints): # person is either [] or contains all keypoints
                    coco_keypoints = [person.keypoints[idx] for idx in model_to_coco_mapping]
                    person.keypoints = coco_keypoints 
    
    return frame_results

def maskanyone_get_config(options: dict):
        """"Ensures Options are valid"""
        valid_hiding_strategies = ['solid_fill', 'transparent_fill', 'blurring', 'pixelation', 'contours', 'none']
        valid_overlay_strategies = ['mp_pose', 'openpose_body25b']
        
        if options.get("hiding_strategy") not in valid_hiding_strategies:
            raise ValueError(f"Invalid hiding strategy. Valid options are: {valid_hiding_strategies}")
        if options.get("overlay_strategy") not in valid_overlay_strategies:
            raise ValueError(f"Invalid overlay strategy. Valid options are: {valid_overlay_strategies}")
        
        return options

def maskanyone_combine_json_files(processed_chunks_dir: str, overlay_strategy: str) -> list:
        """
            Combine JSON files from Mask Anyone Ui dataset into a standardized format.
            processed_chunks_dir: Directory containing Json file for video chunks
        """
        json_file_paths = glob.glob(os.path.join(processed_chunks_dir, "*.json"))
        # Expected chunk file path dataset/video_name/chunk_x.json where x is chunk number
        json_file_paths = sorted(json_file_paths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])) # sort them by chunk number
        all_chunks_keypoints = []  # combined keypoints for all frames all persons

        for chunk_file in json_file_paths:
            frame_results = maskanyone_convert_json_to_nested_arrays(chunk_file, overlay_strategy)
            all_chunks_keypoints.extend(frame_results)

        return all_chunks_keypoints

def maskanyone_convert_json_to_nested_arrays(chunk_poses_file: str, overlay_strategy: str) -> list:
        with open(chunk_poses_file, 'r') as f: 
            data = json.load(f)
            first_person_data = next(iter(data.values()))
            number_of_frames = len(first_person_data)
            
            frame_results = [FramePoseResult(persons=[], frame_idx=i) for i in range(number_of_frames)]
            
            for person_idx, data_person_keypoints in data.items():
                frames = []
                for frame_idx, data_frame_keypoints in enumerate(data_person_keypoints):
                    keypoints = []
                    if data_frame_keypoints is None:
                         frames.append(keypoints)
                         continue

                    # The output of MaskAnyone API for a frame is different for MediaPipe and OpenPose:
                    # For Openpose, the frame output is a dictionary with a key "pose_keypoints" (and other keys like "face_keypoints", "hand_keypoints")
                    # For MediaPipe, the frame output is a list of keypoints
                    if overlay_strategy == "openpose_body25b":
                        data_pose_keypoints = data_frame_keypoints.get("pose_keypoints", None)
                    elif overlay_strategy == "mp_pose": 
                        data_pose_keypoints = data_frame_keypoints
                    else:
                         raise ValueError(f"Invalid overlay strategy provided to maskanyone_combine_json_files in utils.py") 

                    if data_pose_keypoints is None:
                        continue

                    for keypoint in data_pose_keypoints:
                        if not keypoint:
                            keypoints.append(PoseKeypoint(x=0, y=0))
                            continue

                        keypoints.append(PoseKeypoint(x=keypoint[0], y=keypoint[1]))
                    
                    frame_results[frame_idx].persons.append(PersonPoseResult(keypoints=keypoints))
            
            return frame_results


def get_video_metadata(video: str | cv2.VideoCapture) -> dict:
    """
    Get metadata of a video capture object.

    Args:
        cap (str | cv2.VideoCapture): The path to the video or the video capture object.

    Returns:
        dict: A dictionary containing the video's metadata such as width, height, fps, and duration.
    """
    if isinstance(video, str):
        cap = cv2.VideoCapture(video)
    elif isinstance(video, cv2.VideoCapture):
        cap = video

    if not cap.isOpened():
        raise ValueError("Video capture is not opened.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = get_frame_count_ffprobe(video)
    duration = frame_count / fps if fps > 0 else 0

    metadata = {"width": width, "height": height, "fps": fps, "duration": duration, "frame_count": frame_count}

    return cap, metadata

def get_frame_count_ffprobe(video_path: str) -> int:
    """
    Returns the accurate number of video frames using ffprobe.
    We cannot use cv2.CAP_PROP_FRAME_COUNT because it is not accurate for some videos.

    Args:
        video_path (str): Path to the video file.

    Returns:
        int: Number of frames in the video.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        frame_count = int(result.stdout.strip())
        return frame_count
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr.strip()}")
    except ValueError:
        raise RuntimeError("Could not parse frame count from ffprobe output.")
