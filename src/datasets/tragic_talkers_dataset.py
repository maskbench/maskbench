import os
import json
import glob
from typing import Dict, List
from dataclasses import asdict

from inference import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult
from keypoint_pairs import COCO_KEYPOINT_PAIRS, COCO_TO_TRAGIC_TALKERS_OPENPOSE, OPENPOSE_KEYPOINT_PAIRS
from utils import convert_keypoints_to_coco_format
from .dataset import Dataset
from .video_sample import VideoSample

class TragicTalkersDataset(Dataset):
    def __init__(self, name: str, dataset_folder: str, config: dict = None):
        super().__init__(name, dataset_folder, config)
        self.convert_gt_keypoints_to_coco = config.get("convert_gt_keypoints_to_coco", False)
    
    def _load_samples(self) -> List[VideoSample]:
        self.video_folder = os.path.join(self.dataset_folder, self.config.get("video_folder")) # adjust according to folder structure
        self.gt_folder = os.path.join(self.dataset_folder, self.config.get("ground_truth_folder")) # adjust according to folder structure

        samples = []
        video_extensions = (".avi", ".mp4")
        list_of_videos = glob.glob(os.path.join(self.video_folder, "*", "*"))

        for video in list_of_videos:
            if video.endswith(video_extensions):
                samples.append(VideoSample(video))
      
        return samples

    def get_gt_keypoint_pairs(self) -> List[tuple]:
        if self.convert_gt_keypoints_to_coco:
            return COCO_KEYPOINT_PAIRS
        else:
            return OPENPOSE_KEYPOINT_PAIRS

    def get_gt_pose_results(self) -> Dict[str, VideoPoseResult]:
        gt_pose_results = {}
        video_json_folders = glob.glob(os.path.join(self.gt_folder, "*", "*")) # for every video & camera angle
        for video_json_folder in video_json_folders:
            video_name = self._extract_video_name_from_labels_folder(video_json_folder)
            gt_pose_result = self.combine_json_files_for_video(video_json_folder, video_name)
            if self.convert_gt_keypoints_to_coco:
                gt_pose_result.frames = convert_keypoints_to_coco_format(gt_pose_result.frames, COCO_TO_TRAGIC_TALKERS_OPENPOSE)
            gt_pose_results[video_name] = gt_pose_result
        return gt_pose_results

    def combine_json_files_for_video(self, video_json_folder: str, video_name: str) -> str:
        all_json_files = glob.glob(os.path.join(video_json_folder, "*"))
        all_json_files = sorted(all_json_files, key=lambda x: int(os.path.basename(x).split('-')[1].split('_')[0])) # we need to sort them by frame number

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

        return VideoPoseResult(
            fps=30,
            frame_width=2448,
            frame_height=2048,
            video_name=video_name,
            frames=all_frames_keypoints
        )

    def _extract_video_name_from_labels_folder(self, path: str) -> str:
        """
        Extract video name from a labels folder.
        For example, a path like /datasets/tragic_talkers/labels/conversation1_t3/cam-022 will be converted to conversation1_t3-cam22.
        """
        parts = path.split(os.sep)
        conversation = parts[-2]  # e.g. conversation1_t3
        camera = parts[-1]  # e.g. cam-022
        
        # Extract camera number and format it
        cam_number = camera.split('-')[1]  # e.g. 022
        cam_number = cam_number[1:] # cam number in labels has 3 digits, we need to remove the leading one
        
        return f"{conversation}-cam{cam_number}"
