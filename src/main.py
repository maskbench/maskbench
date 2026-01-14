import importlib
import os
import yaml
from typing import List
import logging 
import datetime

current_session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=f'{current_session}_maskbench.log')

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

    checkpoint_name = config.get("inference_checkpoint_name", None)
    checkpoint_name = checkpoint_name if checkpoint_name != "None" else None
    checkpointer = Checkpointer(dataset.name, checkpoint_name)
    checkpointer.save_config(config_file_path)

    execute_evaluation = config.get("execute_evaluation", True)
    execute_rendering = config.get("execute_rendering", True)
    
    run(dataset, pose_estimators, metrics, checkpointer, execute_evaluation, execute_rendering)
    print("Done")


def run(dataset: Dataset, pose_estimators: List[PoseEstimator], metrics: List[Metric], checkpointer: Checkpointer, execute_evaluation: bool, execute_rendering: bool):
    inference_engine = InferenceEngine(dataset, pose_estimators, checkpointer)
    gt_pose_results = dataset.get_gt_pose_results()
    pose_results = inference_engine.run_parallel_tasks()
    
    if execute_evaluation:
        print("Executing evaluation.")
        evaluator = Evaluator(metrics=metrics)
        metric_results = evaluator.evaluate(pose_results, gt_pose_results)

        visualizer = MaskBenchVisualizer(checkpointer)
        visualizer.generate_all_plots(metric_results)

    if execute_rendering:
        print("Executing rendering.")
        estimators_point_pairs = {est.name: est.get_keypoint_pairs() for est in pose_estimators}
        if gt_pose_results and dataset.get_gt_keypoint_pairs() is not None:
            pose_results["GroundTruth"] = gt_pose_results
            estimators_point_pairs["GroundTruth"] = dataset.get_gt_keypoint_pairs()

        pose_renderer = PoseRenderer(dataset, estimators_point_pairs, checkpointer)
        pose_renderer.render_all_videos(pose_results)


def parse_code_file(code_file: str) -> tuple[str, str]:
    if not code_file or '.' not in code_file:
        raise ValueError(f"Invalid code_file format: {code_file}. Expected format: 'module.path.ClassName'")
    
    parts = code_file.split('.')
    class_name = parts[-1]
    module_path = '.'.join(parts[:-1])
    
    return module_path, class_name


def load_config() -> dict:
    config_file_name = os.getenv("MASKBENCH_CONFIG_FILE")
    config_file_path = os.path.join("/config", config_file_name)

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuration file is empty or not found.")

    return config, config_file_path


def load_dataset(dataset_specification: dict) -> Dataset:
    video_folder = dataset_specification.get("video_folder")
    gt_folder = dataset_specification.get("gt_folder", None)  # Optional - can be None
    config = dataset_specification.get("config", {})

    if video_folder is None:
        raise ValueError("Dataset configuration must specify video_folder")

    try:
        dataset_name = dataset_specification.get("name")
        module_path, class_name = parse_code_file(dataset_specification.get("code_file"))
        dataset_module = importlib.import_module(module_path)
        dataset_class = getattr(dataset_module, class_name)
        dataset = dataset_class(dataset_name, video_folder=video_folder, gt_folder=gt_folder, config=config)
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
            module_path, class_name = parse_code_file(spec.get("code_file"))
            estimator_module = importlib.import_module(module_path)
            estimator_class = getattr(estimator_module, class_name)
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
            module_path, class_name = parse_code_file(spec.get("code_file"))
            metric_module = importlib.import_module(module_path)
            metric_class = getattr(metric_module, class_name)
            metric = metric_class(config=metric_config)
            metrics.append(metric)
        except (ImportError, AttributeError, TypeError) as e:
            print(f"Error instantiating metric {metric_name}: {e}")

    return metrics


if __name__ == "__main__":
    main()
    # run_raw_masked_experiment()
