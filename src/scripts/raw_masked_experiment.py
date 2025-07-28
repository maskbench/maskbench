# from main import load_config, load_dataset
from checkpointer import Checkpointer
from evaluation.utils import aggregate_results_over_all_videos
from evaluation.evaluator import Evaluator
from evaluation.metrics import PCKMetric, RMSEMetric
from tabulate import tabulate
from typing import Dict, List, Any
import pandas as pd

STRATEGIES = ["Blurring", "Pixelation", "Contours", "Solid Fill"]
POSE_ESTIMATOR_ORDER = [
    "YoloPose",
    "MediaPipePose",
    "OpenPose",
    "MaskAnyoneAPI-MediaPipe",
    "MaskAnyoneAPI-OpenPose",
    "MaskAnyoneUI-MediaPipe",
    "MaskAnyoneUI-OpenPose"
]

def _create_metric_dataframe(
    results: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    df = pd.DataFrame(index=POSE_ESTIMATOR_ORDER, columns=STRATEGIES)
    df.index.name = "Pose Estimator"
    
    for pose_estimator in POSE_ESTIMATOR_ORDER:
        if pose_estimator in results:
            for strategy in STRATEGIES:
                df.loc[pose_estimator, strategy] = results[pose_estimator].get(strategy, "N/A")
    
    # Calculate average, handling "N/A" values
    df = df.replace("N/A", pd.NA)  # Convert string "N/A" to pandas NA
    df["Average"] = df[STRATEGIES].apply(lambda x: x.mean() if not x.isna().all() else "N/A", axis=1)
    df = df.fillna("N/A")  # Convert back NA to "N/A" string
    
    return df

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
        - RawMaskedExperiment-SolidFill
    The experiment then runs the following:
        - For each pose estimator, it assumes that the pose results for the raw videos are the "ground truth" pose results.
        - For each pose estimator, it then evaluates the RMSE and PCK metrics for each of the 5 hiding strategies compared to the "ground truth" pose resultsfrom the raw videos.
        - It then prints the results in a table.
    """
    dataset_name = "RawMaskedExperiment"
    strategies = ["Raw"] + STRATEGIES
    
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

    rmse_df = _create_metric_dataframe(rmse_results)
    rmse_df.to_csv("/output/raw_masked_rmse_results.csv", float_format="%.2f")
    print("\nRMSE Results:")
    print(tabulate(rmse_df, headers="keys", tablefmt="grid", floatfmt=".2f", showindex=True))

    pck_df = _create_metric_dataframe(pck_results)
    pck_df.to_csv("/output/raw_masked_pck_results.csv", float_format="%.2f")
    print("\nPCK Results:")
    print(tabulate(pck_df, headers="keys", tablefmt="grid", floatfmt=".2f", showindex=True))
