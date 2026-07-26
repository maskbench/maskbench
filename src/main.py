import importlib
import os
import yaml
from typing import List
import logging 
import datetime

current_session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

from datasets import Dataset
from inference import InferenceEngine
from checkpointer import Checkpointer
from models import PoseEstimator
from rendering import PoseRenderer
from evaluation import Evaluator, MaskBenchVisualizer
from evaluation.metrics import Metric
from save_formats.special_format import SpecialFormat

def main():
    config, config_file_path = load_config()

    dataset_specification = config.get("dataset", {})
    dataset = load_dataset(dataset_specification)
    print("Dataset:", dataset.name)

    checkpoint_name = config.get("inference_checkpoint_name", None)
    checkpoint_name = checkpoint_name if checkpoint_name != "None" else None
    checkpointer = Checkpointer(dataset.name, len(dataset), checkpoint_name)
    log_folder =  checkpointer.checkpoint_dir or "/output"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s', filename=f'{log_folder}/{current_session}_maskbench.log')

    checkpointer.save_config(config_file_path)
    logging.info(f"Loaded dataset '{dataset.name}' with {len(dataset)} videos.")

    pose_estimator_specifications = config.get("pose_estimators", [])
    pose_estimators = load_pose_estimators(pose_estimator_specifications)
    print("Available pose estimators:", [est.name for est in pose_estimators])
    logging.info(f"Loaded {len(pose_estimators)} pose estimators: {[est.name for est in pose_estimators]}")

    metric_specifications = config.get("metrics", [])
    metrics = load_metrics(metric_specifications)
    print("Available metrics:", [metric.name for metric in metrics])
    logging.info(f"Loaded {len(metrics)} metrics: {[metric.name for metric in metrics]}")

    special_save_format_specifications = config.get("special_save_format", [])
    special_save_formats = load_special_save_formats(special_save_format_specifications, checkpointer)
    print("Available special save formats:", [format.name for format in special_save_formats])
    logging.info(f"Loaded {len(special_save_formats)} special save formats: {[format.name for format in special_save_formats]}")

    execute_evaluation = config.get("execute_evaluation", False)
    execute_rendering = config.get("execute_rendering", False)
    render_poses_only = config.get("render_poses_only", False)
    execute_processing = config.get("execute_processing", True)
    max_inference_workers = config.get("max_inference_workers", 10)
    max_rendering_workers = config.get("max_rendering_workers", 10)
    max_special_save_workers = config.get("max_special_save_workers", 10)

    run(dataset, pose_estimators, metrics, special_save_formats, checkpointer, execute_evaluation, execute_rendering, render_poses_only, execute_processing, max_inference_workers, max_rendering_workers, max_special_save_workers)
    print("Done")


def run(dataset: Dataset, pose_estimators: List[PoseEstimator], metrics: List[Metric], special_save_formats: List[SpecialFormat], checkpointer: Checkpointer, execute_evaluation: bool, execute_rendering: bool, render_poses_only: bool, execute_processing: bool, max_inference_workers: int, max_rendering_workers: int, max_special_save_workers: int):
    logging.info('Starting Inference')
    inference_engine = InferenceEngine(dataset, pose_estimators, checkpointer, execute_processing)
    inference_engine.run_parallel_tasks(max_workers=max_inference_workers)

    if execute_evaluation:
        print("Executing evaluation.")
        evaluator = Evaluator(metrics=metrics, checkpointer=checkpointer, dataset=dataset)
        model_list = [estimator.name for estimator in pose_estimators]
        metric_results = evaluator.evaluate(model_list)
        visualizer = MaskBenchVisualizer(checkpointer)
        visualizer.generate_all_plots(metric_results)

    if execute_rendering:
        logging.info("Executing rendering.")
        estimators_point_pairs = {est.name: est.get_keypoint_pairs() for est in pose_estimators}

        gt_keypoint_pairs = dataset.get_gt_keypoint_pairs()
        if gt_keypoint_pairs is None:
            logging.warning("Ground truth keypoint pairs not found. Rendering will proceed without ground truth poses.")
        else:
            estimators_point_pairs["GroundTruth"] = gt_keypoint_pairs

        pose_renderer = PoseRenderer(dataset, estimators_point_pairs, checkpointer, render_poses_only)
        pose_renderer.render_all_videos(max_workers=max_rendering_workers)

    for save_format in special_save_formats:
        logging.info(f"Saving results in special format: {save_format.name}")
        save_format.save_pose_results(dataset, pose_estimators, max_workers=max_special_save_workers)

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
        raise ValueError(f"Error instantiating dataset {dataset_specification.get('name')}: {e}")

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
            raise ValueError(f"Error instantiating pose estimator {estimator_name}: {e}")

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
            raise ValueError(f"Error instantiating metric {metric_name}: {e}")

    return metrics

def load_special_save_formats(special_save_format_specifications: List[dict], checkpointer: Checkpointer) -> List[SpecialFormat]:
    save_formats = []
    for spec in special_save_format_specifications:
        format_name = spec.get("name")
        format_enabled = spec.get("enabled", True)

        if not format_enabled:
            continue

        try:
            module_path, class_name = parse_code_file(spec.get("code_file"))
            format_module = importlib.import_module(module_path)
            format_class = getattr(format_module, class_name)
            save_format = format_class(name=format_name, checkpointer=checkpointer)
            save_formats.append(save_format)
        except (ImportError, AttributeError, TypeError) as e:
            raise ValueError(f"Error instantiating special save format {format_name}: {e}")

    return save_formats

if __name__ == "__main__":
    main()
    # run_raw_masked_experiment()
