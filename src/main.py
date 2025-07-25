import importlib
import os
import yaml
from typing import List

from datasets import Dataset
from inference import InferenceEngine
from checkpointer import Checkpointer
from models import PoseEstimator
from rendering import PoseRenderer
from evaluation import Evaluator, MaskBenchVisualizer
from evaluation.metrics import Metric
from scripts.raw_masked_experiment import run_raw_masked_experiment

def main():
    config, config_file_path = load_config()

    dataset_specification = config.get("dataset", {})
    dataset = load_dataset(dataset_specification)
    print("Dataset:", dataset.name)

    pose_estimator_specifications = config.get("pose_estimators", [])
    pose_estimators = load_pose_estimators(pose_estimator_specifications)
    print("Available pose estimators:", [est.name for est in pose_estimators])

    metric_specifications = config.get("metrics", [])
    metrics = load_metrics(metric_specifications)
    print("Available metrics:", [metric.name for metric in metrics])

    checkpoint_name = config.get("checkpoint_name", None)
    checkpoint_name = checkpoint_name if checkpoint_name != "None" else None
    checkpointer = Checkpointer(dataset.name, checkpoint_name)
    checkpointer.save_config(config_file_path)
    
    run(dataset, pose_estimators, metrics, checkpointer)
    print("Done")


def run(dataset: Dataset, pose_estimators: List[PoseEstimator], metrics: List[Metric], checkpointer: Checkpointer):
    inference_engine = InferenceEngine(dataset, pose_estimators, checkpointer)
    gt_pose_results = dataset.get_gt_pose_results()
    pose_results = inference_engine.estimate_pose_keypoints()
    
    evaluator = Evaluator(metrics=metrics)
    metric_results = evaluator.evaluate(pose_results, gt_pose_results)

    visualizer = MaskBenchVisualizer(checkpointer)
    visualizer.generate_all_plots(metric_results)

    estimators_point_pairs = {est.name: est.get_keypoint_pairs() for est in pose_estimators}
    if gt_pose_results and dataset.get_gt_keypoint_pairs() is not None:
        pose_results["GroundTruth"] = gt_pose_results
        estimators_point_pairs["GroundTruth"] = dataset.get_gt_keypoint_pairs()

    pose_renderer = PoseRenderer(dataset, estimators_point_pairs, checkpointer)
    pose_renderer.render_all_videos(pose_results)


def load_config() -> dict:
    config_file_name = os.getenv("MASKBENCH_CONFIG_FILE")
    config_file_path = os.path.join("/config", config_file_name)

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuration file is empty or not found.")

    return config, config_file_path


def load_dataset(dataset_specification: dict) -> Dataset:
    dataset_folder = dataset_specification.get("dataset_folder", "/datasets")
    config = dataset_specification.get("config", {})

    try:
        dataset_name = dataset_specification.get("name")
        dataset_module = importlib.import_module(dataset_specification.get("module"))
        dataset_class = getattr(dataset_module, dataset_specification.get("class"))
        dataset = dataset_class(dataset_name, dataset_folder, config)  # initialize dataset
    except (ImportError, AttributeError, TypeError) as e:
        print(f"Error instantiating dataset {dataset_specification.get('name')}: {e}")
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


def load_metrics(metric_specifications: List[dict]) -> List[Metric]:
    metrics = []
    for spec in metric_specifications:
        metric_name = spec.get("name")
        metric_config = spec.get("config", {})

        try:
            metric_module = importlib.import_module(spec.get("module"))
            metric_class = getattr(metric_module, spec.get("class"))
            metric = metric_class(config=metric_config)
            metrics.append(metric)
        except (ImportError, AttributeError, TypeError) as e:
            print(f"Error instantiating metric {metric_name}: {e}")

    return metrics


if __name__ == "__main__":
    main()
    # run_raw_masked_experiment()
