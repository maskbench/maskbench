import importlib
import os
import yaml

from video_chunker import VideoChunker
from inference.engine import InferenceEngine
from render.pose_render import PoseRender
from dataloader.TED_dataloader import TED_DATALOADER

def main():
    config = load_config()
    pose_estimator_specifications = config.get("models")
    pose_render = PoseRender()

    pose_estimators = load_pose_estimators(pose_estimator_specifications)
    print(f"Available Pose Estimators: {pose_estimators}")

    ted_dataset = TED_DATALOADER("/datasets/ted-talks")

    # for batch in ted_dataset:
    #     print(batch[0].get_info())
    
    dataloaders = [ted_dataset]

    # video_path = ["/datasets/ted-talks/ted_kid.mp4", "/datasets/ted-talks/Y3.mp4", "/datasets/ted-talks/Y_2.mp4"]
    inference_engine = InferenceEngine(pose_estimators)
    inference_engine.get_keypoints_engine(dataloaders) # processing
    
    # # # [optional] select videos and models you want to render
    # inference_engine.render_engine(video_path, pose_estimators, pose_render) #render

    # evaluate 

def load_config():
    config_file_name = os.getenv("MASKBENCH_CONFIG_FILE", "maskbench-config.yml")
    config_file_path = os.path.join("/config", config_file_name)
    
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuration file is empty or not found.")
    
    return config


def load_pose_estimators(pose_estimator_specifications: list):
    pose_estimators = {}
    for spec in pose_estimator_specifications:
        estimator_name = spec.get("name")
        estimator_config = spec.get("config", {})
        estimator_enabled = spec.get("enabled", True)

        if not estimator_enabled:
            continue

        try:
            estimator_module = importlib.import_module(spec.get("module"))
            estimator_class = getattr(estimator_module, spec.get("class"))
            pose_estimator = estimator_class(estimator_name, estimator_config)
            pose_estimators[estimator_name] =  pose_estimator
        except (ImportError, AttributeError, TypeError) as e:
            print(f"Error instantiating pose estimator {estimator_name}: {e}")

    return pose_estimators

if __name__ == "__main__":
    main()