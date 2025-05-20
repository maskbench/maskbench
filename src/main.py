import importlib
import os
import yaml

from video_chunker import VideoChunker
from inference.engine import InferenceEngine
from render.pose_render import PoseRender

def main():
    config = load_config()
    pose_estimator_specifications = config.get("models")
    dataloader_specifications = config.get("dataloaders")
    pose_render = PoseRender()

    pose_estimators = load_pose_estimators(pose_estimator_specifications)
    dataloaders = load_dataloaders(dataloader_specifications)
    print(f"Available Pose Estimators: {pose_estimators}")
    print(f"Available Dataloaders Estimators: {dataloaders}")

    inference_engine = InferenceEngine(pose_estimators)
    inference_engine.get_keypoints_engine(dataloaders) # processing
    
    # # # [optional] select videos and models you want to render
    inference_engine.render_engine(dataloaders, pose_estimators, pose_render) #render

    # evaluate 

def load_config():
    config_file_name = os.getenv("MASKBENCH_CONFIG_FILE", "maskbench-config.yml")
    config_file_path = os.path.join("/config", config_file_name)
    
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuration file is empty or not found.")
    
    return config


def load_dataloaders(dataloader_specifications: list):
    dataloaders = {}
    for spec in dataloader_specifications:
        dataloader_name = spec.get("name")
        dataloader_config = spec.get("config", {})
        dataloader_enabled = spec.get("enabled", True)

        if not dataloader_enabled:
            continue

        try:
            dataloader_module = importlib.import_module(spec.get("module"))
            dataloader_class = getattr(dataloader_module, spec.get("class"))
            dataloader = dataloader_class(dataloader_config)
            dataloaders[dataloader_name] =  dataloader
        except (ImportError, AttributeError, TypeError) as e:
            print(f"Error instantiating dataloader {dataloader_name}: {e}")

    return dataloaders

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