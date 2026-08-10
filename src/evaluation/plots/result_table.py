import numpy as np
import pandas as pd
from tabulate import tabulate

from evaluation.result_generator import ResultGenerator
from checkpointer import Checkpointer
from metric_result_class import COORDINATE_AXIS

def generate_result_table(checkpointer: Checkpointer) -> pd.DataFrame:
    convert_metric_names = ["Velocity", "Acceleration", "Jerk"]
    result_generator = ResultGenerator(checkpointer, test=True)
    metric_names = result_generator.return_metric_names()
    if not len(metric_names):
        return pd.DataFrame() # return empty dataframe

    aggregated_results = {}
    for metric_name in metric_names:
        aggregated_results[metric_name] = {}
        use_mean = any(substr in metric_name.lower() for substr in ['pck'])

        model_names = result_generator.return_model_names_for_metric(metric_name)
        for model_name in model_names:
            aggregated_video_results = []
            for _, metric_result in result_generator.iter_with_metric_model(metric_name, model_name):
                if metric_name in convert_metric_names: # kinematic magnitudes requires for these
                    metric_result = metric_result.aggregate([COORDINATE_AXIS], method='vector_magnitude')

                if use_mean:
                    aggregated_video_results.append(metric_result.aggregate_all(method='mean'))
                else:
                    aggregated_video_results.append(metric_result.aggregate_all(method='median'))

            if use_mean:
                aggregated_results[metric_name][model_name] = np.round(np.mean(aggregated_video_results), decimals=2)
            else:
                aggregated_results[metric_name][model_name] = np.round(np.median(aggregated_video_results), decimals=2)

    first_metric = metric_names[0]
    pose_estimators = list(aggregated_results[first_metric].keys())

    data = {
        'Pose Estimator': pose_estimators
    }

    for metric_name in metric_names: # add column for each metric
        data[metric_name] = [aggregated_results[metric_name][estimator] for estimator in pose_estimators]

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
    



