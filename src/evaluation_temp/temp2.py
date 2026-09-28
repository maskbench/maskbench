"""
Integration snippet: before/after for plot1, plot2, and summary.


Assumes:
    from scalar_cache_stage import (
        sync_scalar_cache,
        make_plot1_video_reduction,
        make_magnitude_median_reduction,
        make_raw_histogram_reduction,
        combine_masked_vectors,
        merge_histograms,
        rebin_to_display_buckets,
    )


CACHE_ROOT = Path("path/to/scalar_cache")  # small -- MBs, not GBs
"""


import numpy as np
import numpy.ma as ma
from pathlib import Path


from scalar_cache_stage import (
    sync_scalar_cache,
    make_plot1_video_reduction,
    make_magnitude_median_reduction,
    make_raw_histogram_reduction,
    combine_masked_vectors,
    merge_histograms,
    rebin_to_display_buckets,
)


CACHE_ROOT = Path("path/to/scalar_cache")




# ============================================================================
# PLOT 1
# ============================================================================
#
# BEFORE:
#
#   for model_name, video_results in metric_results.items():
#       model_values = []
#       unit = next(iter(video_results.values())).unit
#       for video_name, metric_result in video_results.items():
#           if self.convert_to_magnitude:
#               metric_result = metric_result.aggregate([COORDINATE_AXIS], method='vector_magnitude')
#           median_video_keypoint_values = metric_result.aggregate([FRAME_AXIS, PERSON_AXIS], method='median').values
#           model_values.append(median_video_keypoint_values)
#       median_model_keypoint_values = ma.median(np.stack(model_values, axis=0), axis=0)
#
# AFTER:


def plot1_per_model(self, metric_name, model_name, checkpointer, unit):
    reduction_fn = make_plot1_video_reduction(
        coordinate_axis=COORDINATE_AXIS,
        frame_axis=FRAME_AXIS,
        person_axis=PERSON_AXIS,
        convert_to_magnitude=self.convert_to_magnitude,
        method="median",
    )
    cache = sync_scalar_cache(
        checkpointer=checkpointer,
        task_name="plot1_keypoint_median",
        metric_name=metric_name,
        model_name=model_name,
        cache_root=CACHE_ROOT,
        compute_video_scalar=reduction_fn,
        method_version=f"magnitude={self.convert_to_magnitude}_median",
    )
    median_model_keypoint_values = combine_masked_vectors(cache, method="median")
    return median_model_keypoint_values




# ============================================================================
# PLOT 2 -- pass A (per-video / per-model medians, used for axis limits)
# ============================================================================
#
# BEFORE:
#
#   for pose_estimator_name, video_results in pose_estimator_results.items():
#       video_magnitudes = []
#       for video_name, metric_result in video_results.items():
#           magnitude_result = metric_result.aggregate([COORDINATE_AXIS], method='vector_magnitude')
#           video_magnitudes.append(magnitude_result.aggregate_all(method='median'))
#       pose_estimator_medians[pose_estimator_name] = np.median(video_magnitudes)
#
# AFTER:


def plot2_pass_a_per_model(self, metric_name, model_name, checkpointer):
    reduction_fn = make_magnitude_median_reduction(
        coordinate_axis=COORDINATE_AXIS, method="median"
    )
    cache = sync_scalar_cache(
        checkpointer=checkpointer,
        task_name="plot2_median",
        metric_name=metric_name,
        model_name=model_name,
        cache_root=CACHE_ROOT,
        compute_video_scalar=reduction_fn,
        method_version="magnitude_median",
    )
    return np.median(list(cache.values()))




# ============================================================================
# PLOT 2 -- pass B (raw value distribution -> histogram)
# ============================================================================
#
# BEFORE:
#
#   for model_name, video_results in pose_estimator_results.items():
#       model_values = []
#       for metric_result in video_results.values():
#           values = metric_result.values
#           flattened_valid_clipped_vals = self._flatten_clip_validate(values)
#           model_values.extend(np.abs(flattened_valid_clipped_vals.flatten()))
#       distribution = self._compute_distribution(model_values, bin_edges)
#
# AFTER:
#
# FINE_BIN_EDGES should be finer than any display bucketing you'll ever want
# (e.g. width 0.1 if bin_edges/bin_labels are width-2). Define once, reuse
# across runs -- changing FINE_BIN_EDGES later requires a full cache rebuild
# (bump method_version), changing display buckets does not.


FINE_BIN_EDGES = list(np.arange(0, 50.1, 0.1))  # adjust range to your data


def plot2_pass_b_per_model(self, metric_name, model_name, checkpointer, bin_edges, bin_labels):
    reduction_fn = make_raw_histogram_reduction(
        bin_edges=FINE_BIN_EDGES,
        flatten_clip_validate_fn=self._flatten_clip_validate,
    )
    cache = sync_scalar_cache(
        checkpointer=checkpointer,
        task_name="plot2_histogram",
        metric_name=metric_name,
        model_name=model_name,
        cache_root=CACHE_ROOT,
        compute_video_scalar=reduction_fn,
        method_version="raw_hist_v1",
    )
    fine_counts = merge_histograms(cache)
    # display_edges must land exactly on FINE_BIN_EDGES values
    display_pcts = rebin_to_display_buckets(fine_counts, FINE_BIN_EDGES, bin_edges)
    return display_pcts  # {"0-2": pct, "2-4": pct, ...} -- matches bin_labels order




# ============================================================================
# SUMMARY
# ============================================================================
#
# BEFORE (the preprocessing step -- converts every video to magnitude,
# consuming ALL raw JSON every run):
#
#   for metric_name in ["Velocity", "Acceleration", "Jerk"]:
#       if metric_name in metric_results.keys():
#           for model_name, video_results in metric_results[metric_name].items():
#               for video_name, metric_result in video_results.items():
#                   magnitude_values = metric_result.aggregate([COORDINATE_AXIS], method='vector_magnitude')
#                   metric_results[metric_name][model_name][video_name] = magnitude_values
#       return metric_results
#
#   aggregated_results = aggregate_results_over_all_videos(metric_results)
#   ...
#   data[metric_name] = [aggregated_results[metric_name][estimator] for estimator in pose_estimators]
#
# AFTER -- this is the SAME reduction as plot2 pass A (magnitude -> scalar),
# so it reuses the SAME cache (task_name="plot2_median") rather than
# recomputing it. If summary's cross-video combine method ever differs from
# plot2's (e.g. mean instead of median), give it its own task_name/method_version.


def generate_result_table(self, metric_names:list[str], model_names: list[str], checkpointer:Checkpointer):

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
