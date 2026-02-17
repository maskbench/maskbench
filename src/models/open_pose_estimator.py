import io
import json
import os
import pickle

import requests
import utils
from inference import FramePoseResult, PersonPoseResult, PoseKeypoint, VideoPoseResult
from models import PoseEstimator
from keypoint_pairs import COCO_KEYPOINT_PAIRS, COCO_TO_OPENPOSE_BODY25, OPENPOSE_BODY25B_KEYPOINT_PAIRS, COCO_TO_OPENPOSE_BODY25B, OPENPOSE_BODY25_KEYPOINT_PAIRS
class OpenPoseEstimator(PoseEstimator):
    def __init__(self, name: str, config: dict):
        """
        Initialize the OpenPoseEstimator with a name and configuration.
        The config parameter 'overlay_strategy' must be one of 'BODY_25' or 'BODY_25B'.
        If no 'overlay_strategy' is provided in config, default is 'BODY_25B'.
        """
        super().__init__(name, config)
        self.overlay_strategy = config.get("overlay_strategy", "BODY_25B")
        if self.overlay_strategy not in ["BODY_25", "BODY_25B"]:
            raise ValueError(f"Invalid overlay strategy: {self.overlay_strategy}")

        self.model_keypoint_pairs = {"BODY_25": OPENPOSE_BODY25_KEYPOINT_PAIRS, "BODY_25B": OPENPOSE_BODY25B_KEYPOINT_PAIRS}
        self.model_to_coco_mapping = {"BODY_25": COCO_TO_OPENPOSE_BODY25, "BODY_25B": COCO_TO_OPENPOSE_BODY25B}

    def get_keypoint_pairs(self):
        if self.config.get("save_keypoints_in_coco_format", False):
            return COCO_KEYPOINT_PAIRS
        else:
            return self.model_keypoint_pairs[self.config.get("overlay_strategy")]

    def estimate_pose(self, video_path: str) -> VideoPoseResult:
        """
        Estimate the pose of a video using Open pose estimation.

        Args:
            video_path (str): The path to the input video file.
        Returns:
            VideoPoseResult: A standardized result object containing the pose estimation results for the video.
        """
        pose_data, video_metadata = self._query_openpose_container(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_pose_result = self._convert_to_video_pose_result(
            pose_data, video_metadata, video_name
        )
        return video_pose_result

    def _query_openpose_container(self, video_path: str):
        port = os.getenv("OPENPOSE_PORT", 8000)
        host = os.getenv("OPENPOSE_HOST", "openpose")
        url = (
            f"http://{host}:{port}/openpose/estimate-pose-on-video"  # docker image link
        )
        options = {"model_pose": self.overlay_strategy, "multi_person_detection": True}

        extension = os.path.splitext(video_path)[1].lower()
        if extension == ".mp4":
            mime_type = "video/mp4"
        elif extension == ".avi":
            mime_type = "video/x-msvideo"

        _, video_metadata = utils.get_video_metadata(video_path)  # get video metadata
        with open(video_path, "rb") as f:  # only returns 1 person
            files = {"video": (f"video{extension}", f, mime_type)}
            frame = {"options": json.dumps(options)}

            response = requests.post(url, files=files, data=frame)
            if response.status_code == 200:
                buffer = io.BytesIO(response.content)
                pose_data = pickle.load(buffer)
            else:
                raise ValueError(
                    f"Error in OpenPose API: {response.status_code} - {response.text}"
                )
        return pose_data, video_metadata

    def _convert_to_video_pose_result(
        self, pose_data, video_metadata: dict, video_name: str
    ) -> VideoPoseResult:
        frame_results = []
        for idx, frame in enumerate(pose_data):  # every frame
            if frame and frame.get("pose_keypoints").size > 0:  # if data from frame or a pose detected
                person_keypoints = []
                for person in frame.get("pose_keypoints"):
                    keypoints = []
                    for kp in person:
                        if kp[0] == 0 and kp[1] == 0:
                            keypoints.append(PoseKeypoint(x=0, y=0, confidence=None))
                        else:
                            keypoints.append(PoseKeypoint(x=kp[0], y=kp[1], confidence=kp[2]))
                    person_keypoints.append(PersonPoseResult(keypoints=keypoints))
                frame_results.append(FramePoseResult(persons=person_keypoints, frame_idx=idx))
            else:
                frame_results.append(FramePoseResult(persons=[], frame_idx=idx))

        video_pose_result = VideoPoseResult(
            frames=frame_results,
            frame_width=video_metadata.get("width"),
            frame_height=video_metadata.get("height"),
            fps=video_metadata.get("fps"),
            video_name=video_name
        )

        self.assert_frame_count_is_correct(video_pose_result, video_metadata)
        video_pose_result = self.filter_low_confidence_keypoints(video_pose_result)
        if self.config.get("save_keypoints_in_coco_format", False):
            video_pose_result.frames = utils.convert_keypoints_to_coco_format(video_pose_result.frames, self.model_to_coco_mapping[self.overlay_strategy])
        return video_pose_result
