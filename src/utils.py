import cv2
import glob
import os
import json
from inference import FramePoseResult, PersonPoseResult, PoseKeypoint


def maskanyone_combine_json_files(json_dir: str) -> list:
        """
            Combine JSON files from Mask Anyone Ui dataset into a standardized format.
            Json_dir: Directory containing Json file for video chunks
        """
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        json_files = sorted(json_files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])) # sort them by chunk number
        pose_result = []  # combined keypoints for all frames all persons

        for chunk_file in json_files: # every chunk_file is a chunk
            person_keypoints = maskanyone_get_person_keypoints(chunk_file)
            transposed_keypoints = maskanyone_transpose_keypoints(person_keypoints)
            pose_result.extend(transposed_keypoints)  # combine frames keypoints in chunks

        video_results = maskanyone_standardize_keypoints(pose_result)
        return video_results

def maskanyone_get_person_keypoints(chunk_file: str) -> list:
        with open(chunk_file, 'r') as f: 
            data = json.load(f)
            person_keypoints_combined = [] # combined keypoints for all persons
            for _, person_keypoints in data.items(): # for every person
                frame_keypoints_combined = [] # combined keypoints for all frames

                if person_keypoints: # if detection
                    for frame_keypoints in person_keypoints: # for every frame
                        frame_keypoints_structured = None
                        if frame_keypoints: # convert coordinates to int
                            frame_keypoints_structured = [PoseKeypoint(x=keypoint[0], y=keypoint[1]) if keypoint else None for keypoint in frame_keypoints]
                        frame_keypoints_combined.append(frame_keypoints_structured) 
                person_keypoints_combined.append(frame_keypoints_combined) # all frames for that person
            return person_keypoints_combined

def maskanyone_transpose_keypoints(keypoints) -> list:
        # file has person -> frame -> keypoints while we need frame -> person -> keypoints
        number_of_frames = len(keypoints[0])
        transposed_keypoints = [
                [keypoints[person][frame] 
                for person in range(len(keypoints))] 
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    metadata = {"width": width, "height": height, "fps": fps, "duration": duration}

    return cap, metadata
