import requests
import zipfile
import json
import io
import os
import utils
from models.pose_estimator import PoseEstimator
from models.postprocessing.maskanyone import combine_json_files
from evaluation.pose_result import VideoPoseResult
from video_chunker import VideoChunker
import shutil

class MaskAnyonePoseEstimator(PoseEstimator):
    def __init__(self, model_name: str, config: dict):
        """
        Initialize the OpenPoseEstimator with a model name and configuration.
        """
        super().__init__(model_name, config)
        self.docker_url = "http://maskanyone:8000/mask-video"
        
    def get_point_pairs(self):
        return [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 19), (15, 17), (17, 19), (15, 21),
        (12, 14), (14, 16), (16, 20), (16, 18), (18, 20), (11, 23), (12, 24), (16, 22),
        (23, 25), (24, 26), (25, 27), (26, 28), (23, 24),
        (28, 30), (28, 32), (30, 32), (27, 29), (27, 31), (29, 31)
        ]
        
    def estimate_pose(self, video_path: str) -> list:
        """
        Estimate the pose of a video using Mask Anyone estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            list: A list of lists containing the keypoints for each frame.

        """
        
        chunk_output_dir = "temp_chunk_dir" # Temporary directory for video chunks
        processed_output_dir = "temp_processed_dir" # Temporary directory for processed chunks

        _, video_metadata = utils.get_video_metadata(video_path) # get video metadata
        width = video_metadata.get("width")
        height = video_metadata.get("height")
        fps = video_metadata.get("fps")
        
        print("Creating Chunks")
        video_chunks = VideoChunker(chunk_length=60).chunk_video(video_path, chunk_output_dir) # chunk video into 30 seconds videos
        print("Created Chunks")
        self._process_chunks(video_chunks, processed_output_dir)  # Process each chunk with MaskAnyone API
        print("Processed Chunks")
        frame_results = combine_json_files(processed_output_dir)  # Combine the JSON files from processed chunks
        print("Combined Chunks Json")

        shutil.rmtree(chunk_output_dir)  # Clean up temporary output directory
        shutil.rmtree(processed_output_dir)  # Clean up temporary output directory
        
        return VideoPoseResult(
            fps=fps,
            frame_width=width,
            frame_height=height,
            frames=frame_results
        )
    
    def _get_config(self):
        options = self.config
        valid_hiding_strategies = ['solid_fill', 'transparent_fill', 'blurring', 'pixelation', 'contours', 'none']
        valid_overlay_strategies = ['mp_hand', 'mp_face', 'mp_pose', 'none', 'openpose', 'openpose_body25b', 'openpose_face','openpose_body_135']
        
        if options.get("hiding_strategy") not in valid_hiding_strategies:
            raise ValueError(f"Invalid hiding strategy. Valid options are: {valid_hiding_strategies}")
        if options.get("overlay_strategy") not in valid_overlay_strategies:
            raise ValueError(f"Invalid overlay strategy. Valid options are: {valid_overlay_strategies}")
        
        return options

    def _process_chunks(self, video_chunks: list, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)  # we will store the chunk json files here
        options = self._get_config()  # Get the configuration options

        video_extension = os.path.splitext(video_chunks[0])[1]          
        if video_extension == ".mp4":
            mime_type = "video/mp4"
        elif video_extension == ".avi":
            mime_type = "video/x-msvideo"
        else:
            raise ValueError(f"Unsupported video format: {video_extension}. Supported formats are .mp4 and .avi.")

        for _, chunk_path in enumerate(video_chunks):
            print("Processing chunk:", chunk_path)
            with open(chunk_path, "rb") as f:
                files = {'video': (os.path.basename(chunk_path), f, mime_type)}
                data ={ "options": json.dumps(options) }

                try:
                    response = requests.post(self.docker_url, files=files, data=data)
                    if response.status_code == 200:
                        zip_content = io.BytesIO(response.content)
                        with zipfile.ZipFile(zip_content, 'r') as zip_file:
                            zip_file.extractall(output_dir)  # Extract to the 'output' directory
                except Exception as e:
                    print(f"Error in MaskAnyone API for {chunk_path}: {e}")