import requests
import zipfile
import json
import io
import os
import utils
import glob
import shutil

from models import PoseEstimator
from video_chunker import VideoChunker
from inference import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult
from keypoint_pairs import MEDIAPIPE_KEYPOINT_PAIRS, OPENPOSE_KEYPOINT_PAIRS

class MaskAnyoneApiPoseEstimator(PoseEstimator):
    def __init__(self, name: str, config: dict):
        """
        Initialize the MaskAnyoneApiPoseEstimator with a name and configuration.
        """
        super().__init__(name, config)
        self.docker_url = "http://maskanyone_api:8000/mask-video"
        self.chunk_output_dir = "/tmp/chunks" # Temporary directory for video chunks
        self.processed_output_dir = "/tmp/processed_chunks" # Temporary directory for processed chunks
        self.options = self._get_config()

    def get_keypoint_pairs(self):
        overlay_strategy = self.options.get("overlay_strategy")
        if overlay_strategy == "openpose_body25b":
            return OPENPOSE_KEYPOINT_PAIRS
        elif overlay_strategy == "mp_pose":
            return MEDIAPIPE_KEYPOINT_PAIRS
        else:
            raise ValueError(f"Overlay strategy {overlay_strategy} is not supported by MaskBench.")

    def estimate_pose(self, video_path: str) -> list:
        """
        Estimate the pose of a video using Mask Anyone Api estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            VideoPoseResult: A standardized result object containing the pose estimation results for the video.

        """
        
        _, video_metadata = utils.get_video_metadata(video_path) # get video 

        print("Creating Chunks")
        video_chunks = VideoChunker(chunk_length=60).chunk_video(video_path, self.chunk_output_dir) # chunk videos
        print("Processing Chunks")
        self._process_chunks(video_chunks, self.processed_output_dir)  # Process each chunk with MaskAnyone API
        print("Combining Chunks Json Files")
        frame_results = self.combine_json_files(self.processed_output_dir)  # Combine the JSON files from processed chunks

        shutil.rmtree(self.chunk_output_dir)  # Clean up temporary output directory
        shutil.rmtree(self.processed_output_dir)  # Clean up temporary output directory
        
        return VideoPoseResult(
            fps=video_metadata.get("fps"),
            frame_width=video_metadata.get("width"),
            frame_height=video_metadata.get("height"),
            video_name=os.path.splitext(os.path.basename(video_path))[0],
            frames=frame_results
        )
    
    def _get_config(self):
        options = self.config
        valid_hiding_strategies = ['solid_fill', 'transparent_fill', 'blurring', 'pixelation', 'contours', 'none']
        valid_overlay_strategies = ['mp_pose', 'openpose_body25b']
        
        if options.get("hiding_strategy") not in valid_hiding_strategies:
            raise ValueError(f"Invalid hiding strategy. Valid options are: {valid_hiding_strategies}")
        if options.get("overlay_strategy") not in valid_overlay_strategies:
            raise ValueError(f"Invalid overlay strategy. Valid options are: {valid_overlay_strategies}")
        
        return options

    def _process_chunks(self, video_chunks: list, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)  # we will store the chunk json files here

        video_extension = os.path.splitext(video_chunks[0])[1]          
        if video_extension == ".mp4":
            mime_type = "video/mp4"
        elif video_extension == ".avi":
            mime_type = "video/x-msvideo"
        else:
            raise ValueError(f"Unsupported video format: {video_extension}. Supported formats are .mp4 and .avi.")

        for _, chunk_path in enumerate(video_chunks):
            with open(chunk_path, "rb") as f:
                files = {'video': (os.path.basename(chunk_path), f, mime_type)}
                data ={ "options": json.dumps(self.options) }

                try:
                    response = requests.post(self.docker_url, files=files, data=data)
                    if response.status_code == 200:
                        zip_content = io.BytesIO(response.content)
                        with zipfile.ZipFile(zip_content, 'r') as zip_file:
                            zip_file.extractall(output_dir)  # Extract to the 'output' directory
                    else:
                        print(f"Received Response Status Code: {response.status_code}")
                except Exception as e:
                    print(f"Error in MaskAnyone API for {chunk_path}: {e}")
    
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



