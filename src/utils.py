import subprocess
import cv2
import glob
import os
import json
from inference import FramePoseResult, PersonPoseResult, PoseKeypoint

def maskanyone_get_config(options: dict):
        """"Ensures Options are valid"""
        valid_hiding_strategies = ['solid_fill', 'transparent_fill', 'blurring', 'pixelation', 'contours', 'none']
        valid_overlay_strategies = ['mp_pose', 'openpose_body25b']
        
        if options.get("hiding_strategy") not in valid_hiding_strategies:
            raise ValueError(f"Invalid hiding strategy. Valid options are: {valid_hiding_strategies}")
        if options.get("overlay_strategy") not in valid_overlay_strategies:
            raise ValueError(f"Invalid overlay strategy. Valid options are: {valid_overlay_strategies}")
        
        return options

def maskanyone_combine_json_files(processed_chunks_dir: str) -> list:
        """
            Combine JSON files from Mask Anyone Ui dataset into a standardized format.
            processed_chunks_dir: Directory containing Json file for video chunks
        """
        json_file_paths = glob.glob(os.path.join(processed_chunks_dir, "*.json"))
        # Expected chunk file path dataset/video_name/chunk_x.json where x is chunk number
        json_file_paths = sorted(json_file_paths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])) # sort them by chunk number
        all_chunks_keypoints = []  # combined keypoints for all frames all persons

        for chunk_file in json_file_paths: # every chunk_file is a chunk
            person_frame_keypoint_array = maskanyone_convert_json_to_nested_arrays(chunk_file)
            transposed_keypoints = maskanyone_transpose_keypoints(person_frame_keypoint_array)
            all_chunks_keypoints.extend(transposed_keypoints)  # combine frames keypoints in chunks

        video_results = maskanyone_standardize_keypoints(all_chunks_keypoints)
        return video_results

def maskanyone_convert_json_to_nested_arrays(chunk_poses_file: str) -> list:
        with open(chunk_poses_file, 'r') as f: 
            data = json.load(f)
            persons = []
            for person_idx, data_person_keypoints in data.items():
                frames = []

                for frame_idx, data_frame_keypoints in enumerate(data_person_keypoints):
                    keypoints = []
                    
                    try: # The output of MaskAnyone API for a frame is different for MediaPipe and OpenPose:
                        # For Openpose, the frame output is a dictionary with a key "pose_keypoints" (and other keys like "face_keypoints", "hand_keypoints")
                        data_pose_keypoints = data_frame_keypoints.get("pose_keypoints")
                    except AttributeError:
                        # For MediaPipe, the frame output is a list of keypoints
                        data_pose_keypoints = data_frame_keypoints

                    for keypoint in data_pose_keypoints:
                        if not keypoint:
                            keypoints.append(PoseKeypoint(x=0, y=0))
                            continue

                        keypoints.append(PoseKeypoint(x=keypoint[0], y=keypoint[1]))
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
