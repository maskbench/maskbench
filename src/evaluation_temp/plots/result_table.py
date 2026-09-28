from typing import Dict
import numpy as np
import pandas as pd
from tabulate import tabulate

from checkpointer import Checkpointer
from metric_result_class import MetricResult
from evaluation.utils import aggregate_results_over_all_videos

# def generate_result_table(metric_results: Dict[str, Dict[str, Dict[str, MetricResult]]]) -> pd.DataFrame:
#     '''
#     we can get keys from foler names
#     '''
#     aggregated_results = aggregate_results_over_all_videos(metric_results)

#     first_metric = list(metric_results.keys())[0]
#     pose_estimators = list(aggregated_results[first_metric].keys())

#     data = {
#         'Pose Estimator': pose_estimators
#     }

#     for metric_name in metric_results.keys(): # add column for each metric
#         data[metric_name] = [aggregated_results[metric_name][estimator] for estimator in pose_estimators]

#     df = pd.DataFrame(data)
    
#     for col in df.columns:
#         if col != 'Pose Estimator':
#             df[col] = df[col].apply(lambda x: f'{x:.2f}') # Format numeric columns to 2 decimal places

#     # Construct column alignment - first column left, metric columns right
#     column_alignments = tuple(["left"] + ["right"] * (len(df.columns) - 1))
            
#     table = tabulate(
#         df,
#         headers='keys',
#         tablefmt="fancy_grid", 
#         numalign="right",
#         stralign="right",
#         colalign=column_alignments,
#         floatfmt=".2f",
#         intfmt=","
#     )
#     print(table)
#     return df
    
def generate_result_table(metric_names:list[str], model_names: list[str], checkpointer:Checkpointer) -> pd.DataFrame:
    '''
    we can get keys from foler names
    '''
    data = {"Pose Estimator": model_names}
    for metric_name in metric_names:
        reduction_function = make_magnitude_median_reduction(
            coordinate_axis=COORDINATE_AXIS, method="median"
        )
        row = []
        for model_name in model_names:
            cache = sync_scalar_cache(
                checkpointer=checkpointer,
                task_name="plot2_median",  # shared with plot2 pass A
                metric_name=metric_name,
                model_name=model_name,
                cache_root=CACHE_ROOT,
                compute_video_scalar=reduction_function,
                method_version="magnitude_median",
            )
            row.append(np.median(list(cache.values())))
        data[metric_name] = row
    ## till here
    df = pd.DataFrame(data)
    
    for col in df.columns:
        if col != 'Pose Estimator':
            df[col] = df[col].apply(lambda x: f'{x:.2f}') # Format numeric columns to 2 decimal places

    # Construct column alignment - first column left, metric columns right
    column_alignments = tuple(["left"] + ["right"] * (len(df.columns) - 1))
            
    table = tabulate(
        df,
        headers='keys',
        tablefmt="fancy_grid", 
        numalign="right",
        stralign="right",
        colalign=column_alignments,
        floatfmt=".2f",
        intfmt=","
    )
    print(table)
    return df
    





