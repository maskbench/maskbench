import os
import json
import glob
from evaluation.pose_result import FramePoseResult, PersonPoseResult, PoseKeypoint

def get_person_keypoints(frame: str) -> list:
    with open(frame, 'r') as f: 
        data = json.load(f)
        person_keypoints_combined = [] # combined keypoints for all persons
        for _, person_keypoints in data.items(): # for every person
            frame_keypoints_combined = [] # combined keypoints for all frames
            
            if person_keypoints: # if detection
                for frame_keypoints in person_keypoints: # for every frame
                    if frame_keypoints: # convert coordinates to int
                        frame_keypoints = [PoseKeypoint(x=keypoint[0], y =keypoint[2]) if keypoint else None for keypoint in frame_keypoints]

                    frame_keypoints_combined.append(frame_keypoints) 
            person_keypoints_combined.append(frame_keypoints_combined) # all frames for that person
        return person_keypoints_combined

def transpose_keypoints(keypoints) -> list:
    # file has person -> frame -> keypoints while we need frame -> person -> keypoints
    number_of_frames = len(keypoints[0])
    transposed_keypoints = [
            [keypoints[person][frame] 
            for person in range(len(keypoints))] 
            for frame in range(number_of_frames)]
    return transposed_keypoints

        
def standardize_keypoints(keypoints: list) -> list:
    frame_results = []
    for frame_idx, frame in enumerate(keypoints):
        frame_persons = []
        for person in frame:
            frame_persons.append(PersonPoseResult(keypoints = person))
        frame_results.append(FramePoseResult(persons=frame_persons, frame_idx=frame_idx))
    return frame_results

def combine_json_files(json_dir: str) -> list:
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    json_files = sorted(json_files, key=lambda x: int(os.path.basename(x).split('_')[1])) # we need to sort them by frame number
    
    pose_result = []  # combined keypoints for all frames all persons
    for frame in json_files: # every file is a frame
        person_keypoints = get_person_keypoints(frame)
        transpose_keypoints = transpose_keypoints(person_keypoints)
    pose_result.extend(transpose_keypoints)  # concatenate frames across files
    standardized_keypoints = standardize_keypoints(pose_result)
    return standardized_keypoints


