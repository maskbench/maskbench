import importlib
import os
import yaml

from video_chunker import VideoChunker
from inference.engine import InferenceEngine
from render.pose_render import PoseRender

def main():
    config = load_config()
    pose_estimator_specifications = config.get("models")
    dataloader_specification = config.get("dataloader")
    pose_render = PoseRender()

    dataloader = load_dataloader(dataloader_specification)
    pose_estimators = load_pose_estimators(pose_estimator_specifications)
    print(f"Available Pose Estimators: {pose_estimators.keys()}")

    inference_engine = InferenceEngine(dataloader, pose_estimators)
    inference_engine.estimate_pose_keypoints() # processing
    inference_engine.render_all_videos(pose_render) #rendering
    print("Done")

def load_config():
    config_file_name = os.getenv("MASKBENCH_CONFIG_FILE", "maskbench-config.yml")
    config_file_path = os.path.join("/config", config_file_name)
    
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuration file is empty or not found.")
    
    return config


def load_dataloader(dataloader_specification: dict):
    shuffle = dataloader_specification.get("shuffle", False)
    batch_size = dataloader_specification.get("batch_size", 1)
    dataset_folder = dataloader_specification.get("dataset_folder", "/datasets")
    
    try:
        dataloader_module = importlib.import_module(dataloader_specification.get("module"))
        dataloader_class = getattr(dataloader_module, dataloader_specification.get("class"))
        dataloader = dataloader_class(dataset_folder, shuffle, batch_size) # initialize dataloader
    except (ImportError, AttributeError, TypeError) as e:
        print(f"Error instantiating dataloader {dataloader_specification.get("name")}: {e}")

    return dataloader

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
            pose_estimators[estimator_name] = pose_estimator
        except (ImportError, AttributeError, TypeError) as e:
            print(f"Error instantiating pose estimator {estimator_name}: {e}")

    return pose_estimators

if __name__ == "__main__":
    main()