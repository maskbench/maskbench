import json
import os
import shutil
import time 

class InferenceEngine():
    def __init__(self, model_list: dict ):
        self.pair_points = dict()
        self.model_list = model_list
        self.base_output_path = "/output" 

        # get keypoint pairing points for each model
        for model_name, model in self.model_list.items():
            self.pair_points[model_name] = model.get_pair_points()
        
    
    def get_keypoints_engine(self, dataloader_list: list):
        print(dataloader_list)
        for dataloader in dataloader_list: # every dataloader
            for video_batch in  dataloader: # every batches of video within this dataloader
                print(video_batch)
                for video in video_batch: # every video
                    print(video)
                    for video_path in video.get_video_path():
                        print("working on", video_path)
                        video_name = os.path.splitext(os.path.basename(video_path))[0]
                        output_path = os.path.join(self.base_output_path, video_name)

                        if os.path.exists(output_path): 
                            shutil.rmtree(output_path)

                        os.makedirs(output_path, exist_ok=True)

                        for model_name, model in self.model_list.items(): # for every model 
                            start_time = time.time()
                            keypoints = model.estimate_pose(video_path) 
                            with open(os.path.join(output_path, model_name + ".json"), "w+") as f:
                                json.dump(keypoints, f)
                            print(f"get keypoints: {model_name} - {time.time() - start_time}")

    # only render for provided videos and models
    def render_engine(self, video_list: str, model_list: str, renderer):
        for video_path in video_list: # for every model
            start_time = time.time()
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.base_output_path, video_name)
            os.makedirs(output_path, exist_ok=True) # create folder if doesnt exist 
            
            renderer.render_video(video_path, output_path, model_list, self.pair_points)
            print(f"render: {video_path} - {time.time() - start_time}")

            
 

