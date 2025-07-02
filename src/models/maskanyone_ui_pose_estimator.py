import glob
import os
import json
import utils

from inference import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult
from models import PoseEstimator

class MaskAnyoneUiPoseEstimator(PoseEstimator):
    def __init__(self, name: str, config: dict):
        """
        Initialize the MaskAnyoneUiPoseEstimator with a name and configuration.
        """
        super().__init__(name, config)
        self.maskanyone_ui_dataset = self.config.get("dataset_folder_path")

    def get_keypoint_pairs(self):
        return [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 7),
            (0, 4),
            (4, 5),
            (5, 6),
            (6, 8),
            (9, 10),
            (11, 12),
            (11, 13),
            (13, 15),
            (15, 19),
            (15, 17),
            (17, 19),
            (15, 21),
            (12, 14),
            (14, 16),
            (16, 20),
            (16, 18),
            (18, 20),
            (11, 23),
            (12, 24),
            (16, 22),
            (23, 25),
            (24, 26),
            (25, 27),
            (26, 28),
            (23, 24),
            (28, 30),
            (28, 32),
            (30, 32),
            (27, 29),
            (27, 31),
            (29, 31),
        ]

    def estimate_pose(self, video_path: str) -> VideoPoseResult:
        """
        Estimate the pose of a video using Mask Anyone Ui  estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            VideoPoseResult: A standardized result object containing the pose estimation results for the video.
        """
        _, video_metadata = utils.get_video_metadata(video_path) # get video 

        video_name = os.path.splitext(os.path.basename(video_path))[0] # get video name
        results_path = os.path.join(self.maskanyone_ui_dataset, video_name) # path of jsons
        if not os.path.exists(results_path):
            raise f"{video_name} was not found in Mask Anyone Ui Dataset Folder Path"
            
        frame_results = self.combine_json_files(results_path)  # Combine the JSON files from processed chunks
        return VideoPoseResult(
            fps=video_metadata.get("fps"),
            frame_width=video_metadata.get("width"),
            frame_height=video_metadata.get("height"),
            frames=frame_results
        )

    def combine_json_files(self, json_dir: str) -> list:
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        json_files = sorted(json_files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])) # sort them by chunk number
        pose_result = []  # combined keypoints for all frames all persons

        for file in json_files: # every file is a chunk
            person_keypoints = self._get_person_keypoints(file)
            transposed_keypoints = self._transpose_keypoints(person_keypoints)
            pose_result.extend(transposed_keypoints)  # combine frames keypoints in chunks

        frame_results = self._standardize_keypoints(pose_result)
        return frame_results

    def _get_person_keypoints(self, chunk: str) -> list:
        with open(chunk, 'r') as f: 
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

    def _transpose_keypoints(self, keypoints) -> list:
        # file has person -> frame -> keypoints while we need frame -> person -> keypoints
        number_of_frames = len(keypoints[0])
        transposed_keypoints = [
                [keypoints[person][frame] 
                for person in range(len(keypoints))] 
                for frame in range(number_of_frames)]
        return transposed_keypoints


    def _standardize_keypoints(self, keypoints: list) -> list:
        frame_results = []
        for frame_idx, frame in enumerate(keypoints):
            frame_persons = []
            for person in frame:
                frame_persons.append(PersonPoseResult(keypoints = person))
            frame_results.append(FramePoseResult(persons=frame_persons, frame_idx=frame_idx))
        return frame_results