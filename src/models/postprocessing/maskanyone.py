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
                        frame_keypoints = [PoseKeypoint(x=keypoint[0], y=keypoint[1]) if keypoint else None for keypoint in frame_keypoints]

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
    json_files = sorted(json_files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])) # we need to sort them by frame number
    print(json_files)
    pose_result = []  # combined keypoints for all frames all persons
    
    for file in json_files: # every file is a frame
        person_keypoints = get_person_keypoints(file)
        transposed_keypoints = transpose_keypoints(person_keypoints)
        print(len(transposed_keypoints), "frames in file", file)
        pose_result.extend(transposed_keypoints)  # concatenate frames
    with open(os.path.join("combined_json", "pose_results.json"), 'w') as f:
        json.dump(pose_result, f, default=lambda o: o.__dict__, indent=4)

    frame_results = standardize_keypoints(pose_result)

    os.makedirs("combined_json", exist_ok=True)  # Ensure the output directory exists
    with open(os.path.join("combined_json", "combined_keypoints.json"), 'w') as f:
        json.dump(frame_results, f, default=lambda o: o.__dict__, indent=4)

    return frame_results


