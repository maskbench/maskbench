from models.pose_estimator import PoseEstimator
from models.postprocessing.maskanyone import combine_json_files
from evaluation.pose_result import VideoPoseResult
from video_chunker import VideoChunker
import requests
import zipfile
import json
import io
import os
import shutil
import utils


class MaskAnyonePoseEstimator(PoseEstimator):
    def __init__(self, model_name: str, config: dict):
        """
        Initialize the OpenPoseEstimator with a model name and configuration.
        """
        super().__init__(model_name, config)
        self.docker_url = "http://maskanyone:8000/mask-video"
        
    def get_pair_points(self): # is it using yolo or mediapipe?
        return [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16), (15, 19), (19, 20), (15, 21), (16, 22), (22, 23), (16, 24), (5, 17), (6, 17), (11, 12), (17, 18), (5, 6)]

    def get_config(self):
        options = self.config
        valid_hiding_strategies = ['solid_fill', 'transparent_fill', 'blurring', 'pixelation', 'contours', 'none']
        valid_overlay_strategies = ['mp_hand', 'mp_face', 'mp_pose', 'none', 'openpose', 'openpose_body25b', 'openpose_face','openpose_body_135']
        
        if options.get("hiding_strategy") not in valid_hiding_strategies:
            raise ValueError(f"Invalid hiding strategy. Valid options are: {valid_hiding_strategies}")
        if options.get("overlay_strategy") not in valid_overlay_strategies:
            raise ValueError(f"Invalid overlay strategy. Valid options are: {valid_overlay_strategies}")
        
        return options

    def process_chunks(self, video_extension:str, video_chunks: list, output_dir: str):

        os.makedirs(output_dir, exist_ok=True)  # we will store the chunk json files here
        options = self.get_config()  # Get the configuration options

        if video_extension == ".mp4":
            mime_type = "video/mp4"
        elif video_extension == ".avi":
            mime_type = "video/x-msvideo"

        for chunk_index, chunk_path in enumerate(video_chunks):
            with open(chunk_path, "rb") as f:
                video_name = os.path.basename(chunk_path).split('.')[0]
                options["video_name"] = f"chunk_{chunk_index + 1}"
                files = {'video': (f"{video_name}{video_extension}", f, mime_type)}
                data ={ "options": json.dumps(options) }

                response = requests.post(self.docker_url, files=files, data=data)
                
                if response.status_code == 200:
                    zip_content = io.BytesIO(response.content)
                    with zipfile.ZipFile(zip_content, 'r') as zip_file:
                        # save pose file only
                        zip_file.extractall(output_dir)  # Extract to the 'output' directory
                else:
                    raise ValueError(f"Error in MaskAnyone API: {response.status_code} - {response.text}")

    def add_keypoints(self):
        pass
    
    def estimate_pose(self, video_path: str) -> list:
        """
        Estimate the pose of a video using Mask Anyone estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            list: A list of lists containing the keypoints for each frame.

        """
        
        video_extension = os.path.splitext(video_path)[1].lower()

        chunk_output_dir = "/temp_chunk_dir"  # Temporary directory for video chunks
        processed_output_dir = "/temp_processed_dir"  # Temporary directory for processed chunks
        
        video_chunks = VideoChunker(chunk_length=30).chunk_video(video_path, chunk_output_dir) # chunk video into 30 seconds videos
        self.process_chunks(video_extension, video_chunks, processed_output_dir)  # Process each chunk with MaskAnyone API
        frame_results = combine_json_files(processed_output_dir)  # Combine the JSON files from processed chunks

        _, video_metadata = utils.get_video_metadata(video_path) # get video metadata
        width = video_metadata.get("width")
        height = video_metadata.get("height")
        fps = video_metadata.get("fps")

        shutil.rmtree(chunk_output_dir)  # Clean up temporary output directory
        shutil.rmtree(processed_output_dir)  # Clean up temporary output directory
        return VideoPoseResult(
            fps=fps,
            frame_width=width,
            frame_height=height,
            frames=frame_results
        )