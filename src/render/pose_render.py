import cv2
import os
import json

from dataloader.video_sample import VideoSample
from evaluation.pose_result import FramePoseResult

COLORS = [ # these are colors that even color-blind people can see
(0, 115, 178), # blue
(204, 102, 0), # orange
(0, 153, 127), # green-blue
(242, 229, 64), # yellow
(89, 178, 229), # cyan
(204, 153, 178), # pink
(229, 153, 0), # gold
]

class PoseRender():
    def __init__(self):
        pass
    
    def get_keypoints(self, output_path:str, model_name: str): 
        """ Fetch keypoints from json for specific video and model """
        file_path = os.path.join(output_path, model_name + ".json")
        with open(file_path) as f:
            keypoints = json.load(f)
        return keypoints
    

    def render_video(self, video: VideoSample, output_path:str, models_point_pairs: dict):
        """
        Args:
        Input:
            video_path: video path of 1 video
            model_list: list of models classes
            points_pair: dict of shape -> {model_name: list of pair points} showing how to pair keypoints
        """       
        cap = cv2.VideoCapture(video.path) # load the video
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video.path}")
    
        fps = int(cap.get(cv2.CAP_PROP_FPS)) # get video specs 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writers = [] # initialize video writers 

        for model_name in models_point_pairs.keys(): # video writer for every model
            out = cv2.VideoWriter(f"{output_path}/{model_name}.mp4", fourcc, fps, (width, height))
            video_writers.append(out)


        frame_number = 0
        while cap.isOpened(): # for every frame
            ret, frame = cap.read()
            if not ret:
                break
            frame_copies = [frame.copy() for _ in range(len(video_writers))] # deep copy of frames to avoid overwriting
            model_idx = 0

            for idx, (model_name, model_points_pair) in enumerate(models_point_pairs.items()): # for every model
                try: # in case no keypoints for this frame
                    frame_keypoints = video.pose_results[model_name].frames[frame_number]
                    frame_copies[idx] = self.draw_keypoints(frame_copies[idx], frame_keypoints, model_points_pair, COLORS[model_idx]) # draw keypoints on frame
                except:
                    pass
                video_writers[idx].write(frame_copies[idx]) # write rendered frame
                model_idx += 1

            frame_number += 1
    
        cap.release()
        for i in video_writers:
            i.release()
    
    def draw_keypoints(self, frame, frame_pose_result: FramePoseResult, point_pairs, color):
        """ Draw keypoints and join keypoint pairs on 1 frame """
        if not frame_pose_result.persons: # if this frame has no keypoints
            return frame  

        for person in frame_pose_result.persons: # every person
            if not person or not person.keypoints:
                continue

            for keypoint in person.keypoints: # every keypoint
                center = (int(keypoint.x), int(keypoint.y))
                cv2.circle(frame, center, 4, color, -1) # draw the keypoint
            
            for pair in point_pairs: # add lines between keypoints
                try:
                    point1 = (int(person.keypoints[pair[0]].x), int(person.keypoints[pair[0]].y))
                    point2 = (int(person.keypoints[pair[1]].x), int(person.keypoints[pair[1]].y))
                except IndexError:
                    print("error in pose renderer", person.keypoints)
                
                if point1 is None or point2 is None or \
                            point1[0] < 1 and point1[1] < 1 or \
                            point2[0] < 1 and point2[1] < 1:
                    continue
                cv2.line(frame, point1, point2, color=color, thickness=2) 
        
        return frame  



