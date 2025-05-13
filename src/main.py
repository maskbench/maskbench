import importlib
import os
import yaml
import shutil

from video_chunker import VideoChunker

def main():
    config = load_config()
    pose_estimator_specifications = config.get("models")
    renderer_specifications = config.get("renderers")

    pose_estimators = load_pose_estimators(pose_estimator_specifications)
    print(f"Available Pose Estimators: {pose_estimators}")

    renderers = load_renderers(renderer_specifications)
    print(f"Available Renderers: {renderers}")
    
    video_path = "/datasets/ted-talks/ted_kid.mp4"
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = f"/output/{video_name}/"

    if os.path.exists(video_output_path): # remove if already exists (only for development)
        shutil.rmtree(video_output_path)

    os.makedirs(video_output_path)
    
    all_keypoints = dict()
    for name, estimator in pose_estimators.items():
        all_keypoints[name] = estimator.estimate_pose(video_path, video_output_path) # right now, this is mediapipe estimator 
     
    renderers[0].render(video_path, video_output_path, all_keypoints)
    

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

def load_renderers(renderer_specifications: list):
    renderers = []
    for spec in renderer_specifications:
        estimator_name = spec.get("name")
        
        try:
            estimator_module = importlib.import_module(spec.get("module"))
            estimator_class = getattr(estimator_module, spec.get("class"))
            renderer = estimator_class()
            renderers.append(renderer)
        except (ImportError, AttributeError, TypeError) as e:
            print(f"Error instantiating renderer {estimator_name}: {e}")

    return renderers

if __name__ == "__main__":
    main()