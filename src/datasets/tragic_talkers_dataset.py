import os
import json
import glob
from typing import List
from dataclasses import asdict
from .dataset import Dataset
from .video_sample import VideoSample
from inference import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult

class TragicTalkersDataset(Dataset):
    def __init__(self, name: str, dataset_folder: str, config: dict = None):
        super().__init__(name, dataset_folder, config)
    
    def _load_samples(self) -> List[VideoSample]:
        self.video_folder = os.path.join(self.dataset_folder, self.config.get("video_folder")) # adjust according to folder structure
        self.gt_folder = os.path.join(self.dataset_folder, self.config.get("ground_truth_folder")) # adjust according to folder structure
        self.combine_ground_truth_files = self.config.get("combine_ground_truth_files", False)

        if self.combine_ground_truth_files:
            list_of_json_folders = glob.glob(os.path.join(self.gt_folder, "*", "*")) # for every video & camera angle
            self.return_combined_json(list_of_json_folders)
        else:
            print("Json Combination is set to False.")
        
        samples = []
        video_extensions = (".avi", ".mp4")
        list_of_videos = glob.glob(os.path.join(self.video_folder, "*", "*"))

        for video in list_of_videos:
            if video.endswith(video_extensions):
                samples.append(VideoSample(video))
      
        print(f"Tragic Talkers Dataset is Loaded")
        return samples
    
    def get_gt_pose_results(self) -> List[VideoPoseResult]:
        ground_truth_pose_result = []
        ground_truth_json_files = glob.glob(os.path.join(self.video_folder, "*", "*.json"))
        
        for gt_file in ground_truth_json_files:
            with open(gt_file, 'r') as f:
                data = json.load(f)
                frames = data.get("frames", [])
                frame_pose_results = []

                for frame in frames:
                    persons = frame.get("persons", [])
                    person_pose_results = []
                    for person in persons:
                        keypoints = person.get("keypoints", [])
                        pose_keypoints = [PoseKeypoint(x=k["x"], y=k["y"], confidence=k["confidence"]) for k in keypoints]
                        person_pose_results.append(PersonPoseResult(keypoints=pose_keypoints))
                    frame_pose_results.append(FramePoseResult(persons=person_pose_results, frame_idx=frame.get("frame_idx", 0)))
                
                video_pose_result = VideoPoseResult(
                    fps=data.get("fps", None),
                    frame_width=data.get("frame_width", None),
                    frame_height=data.get("frame_height", None),
                    video_name=os.path.splitext(os.path.basename(gt_file))[0],
                    frames=frame_pose_results
                )
                ground_truth_pose_result.append(video_pose_result)
        
        return ground_truth_pose_result

    def return_combined_json(self, list_of_json_folders: list):
        for json_folder in list_of_json_folders:
            all_json_files = glob.glob(os.path.join(json_folder, "*"))
            all_json_files = sorted(all_json_files, key=lambda x: int(os.path.basename(x).split('-')[1].split('_')[0])) # we need to sort them by frame number
            
            all_frames_keypoints = self.combine_json_files(all_json_files)
            serialized_video_pose_result = {
                        "fps": None,
                        "frame_width": None,
                        "frame_height": None,
                        "frames": [asdict(frame) for frame in all_frames_keypoints]
                    }
            
            json_filepath = self.get_file_name(json_folder)
            with open(json_filepath, "w+") as f:
                json.dump(serialized_video_pose_result, f, indent=2)
    
    def combine_json_files(self, all_json_files: list) -> str:
        all_frames_keypoints = []
        for frame_idx, file in enumerate(all_json_files): # for every frame
            frame_keypoints = [] 
            with open(file, 'r') as f:
                data = json.load(f)
                people = data.get('people', [])
                if people:
                    for person in people: # for every person in the frame
                        person_keypoints = []
                        pose_keypoints = person.get('pose_keypoints_2d', [])
                        if pose_keypoints: # get keypoints for that person
                            person_keypoints = [
                                PoseKeypoint(x=pose_keypoints[i], y=pose_keypoints[i+1], confidence=pose_keypoints[i+2])
                                for i in range(0, len(pose_keypoints), 3)]
                        frame_keypoints.append(PersonPoseResult(keypoints=person_keypoints))
            all_frames_keypoints.append(FramePoseResult(persons=frame_keypoints, frame_idx=frame_idx))
            
        return all_frames_keypoints
        
    def get_file_name(self, file_path: str) -> str:
        relative_path = os.path.relpath(file_path, self.gt_folder)
        video_name, camera_angle = relative_path.split(os.sep)
        camera_angle = f"cam{int(camera_angle.split("-")[1]):02d}"
        json_filename = f"{video_name}-{camera_angle}.json"
        json_filepath = os.path.join(self.video_folder, video_name, json_filename)
        return json_filepath
