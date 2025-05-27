import json
import os
import shutil
import time 

class InferenceEngine():
    def __init__(self, dataloader:dict, models_dict: dict):
        self.dataloader = dataloader
        self.model_list = models_dict
        self.base_output_path = "/output"
        self.models_point_pairs = dict()

        # get keypoint pairing points for each model
        for model_name, model in self.model_list.items():
            self.models_point_pairs[model_name] = model.get_point_pairs()
        
    
    def estimate_pose_keypoints(self):
        for video_batch in self.dataloader: # every batches of video within this dataloader
            for video in video_batch: # every video
                print("Running inference for", video.path)

                for model_name, model in self.model_list.items(): # for every model 
                    start_time = time.time()

                    video_pose_result = model.estimate_pose(video.path) 
                    video.add_result(video_pose_result, model_name)

                    print(f"Inference time: {model_name} - {time.time() - start_time}")

    # only render for provided videos and models
    def render_all_videos(self, renderer):
        for video_batch in self.dataloader:
            for video in video_batch:
                start_time = time.time()
                output_path = os.path.join(self.base_output_path, video.get_filename())
                os.makedirs(output_path, exist_ok=True) # create folder if doesnt exist 
                
                print(f"Rendering {video.path} - {time.time() - start_time}")
                renderer.render_video(video, output_path, self.models_point_pairs)

            
 

