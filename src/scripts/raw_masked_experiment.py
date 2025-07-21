# from main import load_config, load_dataset
from checkpointer import Checkpointer
from evaluation.utils import aggregate_results_over_all_videos
from evaluation.evaluator import Evaluator
from evaluation.metrics import PCKMetric, RMSEMetric
from tabulate import tabulate
from typing import Dict, List, Any

def _create_metric_table(
    results: Dict[str, Dict[str, float]],
) -> str:
    strategies = set()
    for pose_estimator_results in results.values():
        strategies.update(pose_estimator_results.keys())
    strategies = sorted(list(strategies))
    
    table_data = []
    for pose_estimator, pose_estimator_results in results.items():
        row = [pose_estimator] + [pose_estimator_results.get(strategy, "N/A") for strategy in strategies]
        table_data.append(row)
    
    headers = ["Pose Estimator"] + strategies
    return tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f")

def _evaluate_strategy(
    evaluator: Evaluator,
    pose_results: Dict[str, Any],
    gt_results: Dict[str, Any],
    pose_estimator_name: str,
    strategy: str
) -> Dict[str, Dict[str, float]]:
    print(f"Metric computation for '{strategy}'")
    pose_estimator_results = {pose_estimator_name: pose_results[pose_estimator_name]}
    metric_results = evaluator.evaluate(pose_estimator_results, gt_results)
    return aggregate_results_over_all_videos(metric_results)

def run_raw_masked_experiment():
    """
    This is a method to run the raw vs. masked experiment.
    It assumes that there are 5 folders in the output directory (each created by a MaskBench run), one for each hiding strategy.
    The folder names are:
        - RawMaskedExperiment-Raw
        - RawMaskedExperiment-Blurring
        - RawMaskedExperiment-Pixelation
        - RawMaskedExperiment-Contours
        - RawMaskedExperiment-Inpainting
    The experiment then runs the following:
        - For each pose estimator, it assumes that the pose results for the raw videos are the "ground truth" pose results.
        - For each pose estimator, it then evaluates the RMSE and PCK metrics for each of the 5 hiding strategies compared to the "ground truth" pose resultsfrom the raw videos.
        - It then prints the results in a table.
    """
    dataset_name = "RawMaskedExperiment"
    strategies = ["Raw", "Blurring", "Pixelation", "Contours", "Inpainting"]
    
    checkpointers = {strategy: Checkpointer(dataset_name, f"{dataset_name}-{strategy}") for strategy in strategies}
    pose_results = {strategy: checkpointer.load_pose_results() for strategy, checkpointer in checkpointers.items()}
    gt_pose_results = pose_results["Raw"]

    metrics = [
        PCKMetric(config={"threshold": 0.2, "normalize_by": "bbox"}),
        RMSEMetric(config={"normalize_by": "bbox"}),
    ]
    evaluator = Evaluator(metrics=metrics)

    pck_results = {}
    rmse_results = {}
    
    for pose_estimator_name in gt_pose_results.keys():
        print(f"Pose estimator: {pose_estimator_name}")
        gt_results = gt_pose_results[pose_estimator_name]
        
        pck_results[pose_estimator_name] = {}
        rmse_results[pose_estimator_name] = {}
        
        for strategy in [s for s in strategies if s != "Raw"]:
            aggregated_results = _evaluate_strategy(evaluator, pose_results[strategy], gt_results, pose_estimator_name, strategy)
            pck_results[pose_estimator_name][strategy] = aggregated_results["PCK"][pose_estimator_name]
            rmse_results[pose_estimator_name][strategy] = aggregated_results["RMSE"][pose_estimator_name]
        
        print()

    print("\nRMSE Results:")
    print(_create_metric_table(rmse_results))

    print("\nPCK Results:")
    print(_create_metric_table(pck_results))
