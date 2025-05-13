import cv2
import os
from models.mediapipe_pose_estimator import get_mediapipe_pair_points
from models.yolo_pose_estimator import get_yolo_pair_points

PAIR_MAPPING = {"MediaPipePose": get_mediapipe_pair_points(), "YoloPose": get_yolo_pair_points()}
COLOR_MAPPING = {"MediaPipePose": (255,255,0), "YoloPose": (0,255,255)}

class PoseRender():
    def __init__(self):
        pass
    def render(self, video_path: str, output_path: str, keypoints: dict):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_number = 0
        video_writers = []

        for video_name in keypoints.keys(): # video writer for every model
            out = cv2.VideoWriter(f"{output_path}/{video_name}.mp4", fourcc, fps, (width, height))
            video_writers.append(out)
        
        while cap.isOpened(): # for every frame
            ret, frame = cap.read()
            if not ret:
                break
            frame_copies = [frame.copy() for _ in range(len(video_writers))] # deep copy of frames to avoid overwriting

            for i, model_name in enumerate(keypoints.keys()): # for every model
                frame_keypoints = keypoints[model_name][frame_number] # get keypoint for specific frame
                self.draw_keypoints(frame_copies[i], frame_keypoints, PAIR_MAPPING[model_name], COLOR_MAPPING[model_name])
                video_writers[i].write(frame_copies[i]) # write rendered frame
            frame_number += 1
        
        cap.release()
        for i in video_writers:
            i.release()

    def draw_keypoints(self, frame, frame_keypoints, pairs, color):
        if not frame_keypoints: # if this frame has no keypoints
            return frame  

        for person_keypoints in frame_keypoints: # every person
            for keypoint in person_keypoints: # every keypoint
               cv2.circle(frame, (keypoint[0], keypoint[1]), 4, color, -1) # draw the keypoint
            
            for pair in pairs: # add lines between keypoints
                point1 = (person_keypoints[pair[0]])
                point2 = (person_keypoints[pair[1]])
                if point1 is None or point2 is None or \
                            point1[0] < 1 and point1[1] < 1 or \
                            point2[0] < 1 and point2[1] < 1:
                    continue
                cv2.line(frame, point1, point2, color=color, thickness=2) 
        
        return frame  



