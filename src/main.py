import importlib
import os
import yaml
from typing import List

from datasets import Dataset
from inference import InferenceEngine
from models import PoseEstimator
from rendering import PoseRenderer

def main():
    config = load_config()
    pose_estimator_specifications = config.get("pose_estimators")
    dataset_specification = config.get("dataset")

    dataset = load_dataset(dataset_specification)
    pose_estimators = load_pose_estimators(pose_estimator_specifications)
    print("Avaliable pose estimators:", [est.name for est in pose_estimators])

    run(dataset, pose_estimators)

    print("Done")


def run(dataset: Dataset, pose_estimators: List[PoseEstimator]):
    inference_engine = InferenceEngine(dataset, pose_estimators)
    pose_results = inference_engine.estimate_pose_keypoints()
    gt_pose_results = dataset.get_gt_pose_results()  

    estimators_point_pairs = {est.name: est.get_keypoint_pairs() for est in pose_estimators}
    pose_renderer = PoseRenderer(dataset, estimators_point_pairs)
    pose_renderer.render_all_videos(pose_results)


def load_config() -> dict:
    config_file_name = os.getenv("MASKBENCH_CONFIG_FILE", "maskbench-config.yml")
    config_file_path = os.path.join("/config", config_file_name)
    
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuration file is empty or not found.")
    
    return config


def load_dataset(dataset_specification: dict) -> Dataset:
    dataset_folder = dataset_specification.get("dataset_folder", "/datasets")
    
    try:
        dataset_module = importlib.import_module(dataset_specification.get("module"))
        dataset_class = getattr(dataset_module, dataset_specification.get("class"))
        dataset = dataset_class(dataset_folder) # initialize dataset
    except (ImportError, AttributeError, TypeError) as e:
        print(f"Error instantiating dataset {dataset_specification.get("name")}: {e}")
        raise e

    return dataset


def load_pose_estimators(pose_estimator_specifications: dict) -> List[PoseEstimator]:
    pose_estimators = []
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
            pose_estimators.append(pose_estimator)
        except (ImportError, AttributeError, TypeError) as e:
            print(f"Error instantiating pose estimator {estimator_name}: {e}")

    return pose_estimators

if __name__ == "__main__":
    main()