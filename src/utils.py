import subprocess
import cv2
import glob
import os
import json
from inference import FramePoseResult, PersonPoseResult, PoseKeypoint
from keypoint_pairs import COCO_TO_MEDIAPIPE, COCO_TO_OPENPOSE

def convert_keypoints_to_coco_format(frame_results: list, model_name: str) -> list:
    if model_name == "YoloPose":
         return frame_results # Yolo keypoints are already stored in Coco format
    
    model_to_coco_mapping = {"MediaPipePose": COCO_TO_MEDIAPIPE, "OpenPose": COCO_TO_OPENPOSE, "mp_pose": COCO_TO_MEDIAPIPE, "openpose_body25b": COCO_TO_OPENPOSE}
    if model_name not in model_to_coco_mapping:
         raise ValueError(f"{model_name} is an invalid pose estimators. Acceptable pose estimators are {list(model_to_coco_mapping.keys())} and YoloPose")

    for frame in frame_results: 
        if frame and frame.persons: # ensuring they are not None
            for person in frame.persons:
                if len(person.keypoints): # person is either [] or contains all keypoints
                    coco_keypoints = [person.keypoints[idx] for idx in model_to_coco_mapping[model_name]]
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

def maskanyone_combine_json_files(processed_chunks_dir: str, valid_overlay_strategy: str) -> list:
        """
            Combine JSON files from Mask Anyone Ui dataset into a standardized format.
            processed_chunks_dir: Directory containing Json file for video chunks
        """
        json_file_paths = glob.glob(os.path.join(processed_chunks_dir, "*.json"))
        # Expected chunk file path dataset/video_name/chunk_x.json where x is chunk number
        json_file_paths = sorted(json_file_paths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])) # sort them by chunk number
        all_chunks_keypoints = []  # combined keypoints for all frames all persons

        for chunk_file in json_file_paths: # every chunk_file is a chunk
            person_frame_keypoint_array = maskanyone_convert_json_to_nested_arrays(chunk_file, valid_overlay_strategy)
            transposed_keypoints = maskanyone_transpose_keypoints(person_frame_keypoint_array)
            all_chunks_keypoints.extend(transposed_keypoints)  # combine frames keypoints in chunks

        video_results = maskanyone_standardize_keypoints(all_chunks_keypoints)
        return video_results

def maskanyone_convert_json_to_nested_arrays(chunk_poses_file: str, valid_overlay_strategy: str) -> list:
        with open(chunk_poses_file, 'r') as f: 
            data = json.load(f)
            persons = []
            for person_idx, data_person_keypoints in data.items():
                frames = []
                for frame_idx, data_frame_keypoints in enumerate(data_person_keypoints):                    
                    # The output of MaskAnyone API for a frame is different for MediaPipe and OpenPose:
                    # For Openpose, the frame output is a dictionary with a key "pose_keypoints" (and other keys like "face_keypoints", "hand_keypoints")
                    # For MediaPipe, the frame output is a list of keypoints
                    keypoints = []
                    if data_frame_keypoints is None:
                         frames.append(keypoints)
                         continue

                    if valid_overlay_strategy == "openpose_body25b":
                        data_pose_keypoints = data_frame_keypoints.get("pose_keypoints", None)
                    elif valid_overlay_strategy == "mp_pose": 
                        data_pose_keypoints = data_frame_keypoints
                    else:
                         raise ValueError(f"Invalid overlay strategy provided to maskanyone_combine_json_files in utils.py") 

                    if data_pose_keypoints is None:
                        frames.append(keypoints)
                        continue
                    
                    for kp in data_pose_keypoints:
                        if kp is None:
                            keypoints.append(PoseKeypoint(x=0, y=0)) 
                        else:
                            keypoints.append(PoseKeypoint(x=kp[0], y=kp[1])) # confidence is not provided by maskanyone
                    
                    frames.append(keypoints)
                persons.append(frames)
            return persons


def maskanyone_transpose_keypoints(person_frame_keypoints) -> list:
        # file has person -> frame -> keypoints while we need frame -> person -> keypoints
        number_of_frames = len(person_frame_keypoints[0])
        transposed_keypoints = [
                [person_frame_keypoints[person][frame] 
                for person in range(len(person_frame_keypoints))] 
                for frame in range(number_of_frames)]
        return transposed_keypoints


def maskanyone_standardize_keypoints(keypoints: list) -> list:
        frame_results = []
        for frame_idx, frame in enumerate(keypoints):
            frame_persons = []
            for person in frame:
                frame_persons.append(PersonPoseResult(keypoints = person))
            frame_results.append(FramePoseResult(persons=frame_persons, frame_idx=frame_idx))
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
