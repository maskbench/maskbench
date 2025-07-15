import requests
import zipfile
import json
import io
import os
import utils
import shutil
from models import PoseEstimator
from video_chunker import VideoChunker
from inference import VideoPoseResult
from keypoint_pairs import COCO_KEYPOINT_PAIRS, MEDIAPIPE_KEYPOINT_PAIRS, OPENPOSE_KEYPOINT_PAIRS

class MaskAnyoneApiPoseEstimator(PoseEstimator):
    def __init__(self, name: str, config: dict):
        """
        Initialize the MaskAnyoneApiPoseEstimator with a name and configuration.
        """
        super().__init__(name, config)
        self.docker_url = "http://maskanyone_api:8000/mask-video"
        self.chunk_output_dir = "/tmp/chunks" # Temporary directory for video chunks
        self.processed_output_dir = "/tmp/processed_chunks" # Temporary directory for processed chunks
        self.options = utils.maskanyone_get_config(self.config)

    def get_keypoint_pairs(self):
        if self.config.get("save_keypoints_in_coco_format", False):
            return COCO_KEYPOINT_PAIRS
        elif self.config.get("overlay_strategy") == "mp_pose":
            return MEDIAPIPE_KEYPOINT_PAIRS 
        elif self.config.get("overlay_strategy") == "openpose_body25b":
            return OPENPOSE_KEYPOINT_PAIRS
        else:
            raise ValueError(f"Invalid overlay strategy. Valid options are: mp_pose and openpose_body25b ")
    
    def estimate_pose(self, video_path: str) -> list:
        """
        Estimate the pose of a video using Mask Anyone Api estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            VideoPoseResult: A standardized result object containing the pose estimation results for the video.

        """
        _, video_metadata = utils.get_video_metadata(video_path)

        print("MaskAnyoneAPI: Splitting video into chunks.")
        video_chunk_paths = VideoChunker(chunk_length=60).chunk_video_using_opencv(video_path, self.chunk_output_dir)
        print("MaskAnyoneAPI: Processing chunks.")
        self._process_chunks(video_chunk_paths, self.processed_output_dir)
        print("MaskAnyoneAPI: Combining chunk outputs into a single video result.")
        frame_results = utils.maskanyone_combine_json_files(self.processed_output_dir, self.options.get("overlay_strategy"))

        # Clean up temporary output directory
        shutil.rmtree(self.chunk_output_dir)
        shutil.rmtree(self.processed_output_dir)

        if self.config.get("save_keypoints_in_coco_format", False):
            frame_results = utils.convert_keypoints_to_coco_format(frame_results, self.config.get("overlay_strategy"))

        video_pose_result = VideoPoseResult(
            fps=video_metadata.get("fps"),
            frame_width=video_metadata.get("width"),
            frame_height=video_metadata.get("height"),
            video_name=os.path.splitext(os.path.basename(video_path))[0],
            frames=frame_results
        )

        self.assert_frame_count_is_correct(video_pose_result, video_metadata)
        video_pose_result = self.filter_low_confidence_keypoints(video_pose_result) # this call will have no effect, because MaskAnyone does not provide confidence scores
        return video_pose_result

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
                        print(f"Error: Received Response Status Code: {response.status_code}")
                
                except requests.exceptions.RequestException as e:
                    print(f"Request Failed in MaskAnyone API for {chunk_path}: {e}")
                except zipfile.BadZipFile as e:
                    print(f"Bad Zip File in MaskAnyone API for {chunk_path}: {e}")
                except OSError as e:
                    print(f"OS Error in MaskAnyone API for {chunk_path}: {e}")
    
