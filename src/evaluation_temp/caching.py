"""
  - "summary" task: one scalar per video (e.g. median magnitude) -> combine
    with np.median(list(cache.values())).
  - "plotting" task: one fixed-length bin-count vector per video (a fine-
    grained histogram) -> combine by SUMMING vectors elementwise across
    videos, then re-bin into whatever coarse display buckets you want at
    plot time. Changing display buckets later costs nothing, since the fine
    histogram already has more resolution than any coarse display bucketing
    would need.


Cache layout on disk (small -- reductions only, never raw arrays):
    cache_root/<task_name>/<metric_name>/<model_name>.json
    {
        "method_version": "v1",
        "values": {"video_0": 0.1234, "video_1": 0.5678, ...}       # scalar case
        # or
        "values": {"video_0": [3,1,0,7,...], "video_1": [0,2,5,...]}  # vector case
    }


Usage from within a task:


    def compute_video_scalar(result):
        magnitude = result.aggregate([COORDINATE_AXIS], method="vector_magnitude")
        return float(magnitude.aggregate_all(method="median"))


    cache = sync_scalar_cache(
        checkpointer=self.checkpointer,
        task_name="velocity_median_task",
        metric_name="Velocity",
        model_name="modelA",
        cache_root=Path("path/to/scalar_cache"),
        compute_video_scalar=compute_video_scalar,
        method_version="v1",
        force_rerun=config.force_rerun,   # your manual toggle
    )
    metric_median_values["modelA"] = np.median(list(cache.values()))
"""


import json
from pathlib import Path
from typing import Callable


import numpy as np
import numpy.ma as ma


def _load_cache(cache_path: Path) -> dict:
    '''
        values: scalar or vector values per video
        method_version: ???
        meta: ???
    '''
    if not cache_path.exists():
        return {"method_version": None, "meta": {}, "values": {}}
    try:
        cache = json.loads(cache_path.read_text())
        cache.setdefault("meta", {})
        return cache
    except (json.JSONDecodeError, OSError):
        # corrupt/partial cache file -- treat as empty, safer than crashing
        return {"method_version": None, "meta": {}, "values": {}}

# TODO DO WE NEED THIS
def make_unit_meta_factory(checkpointer, metric_name: str, model_name: str) -> Callable:
    """
    meta_factory for capturing `.unit` once, only when a cache is built from
    scratch (empty/stale) -- never re-derived on later runs, per-metric,
    matching the assumption that all videos of a metric share one unit.
    Reads exactly one video (whichever exists) to get it.
    """
    def factory() -> dict:
        json_dir = checkpointer.evaluation_dir / metric_name / model_name
        video_paths = list(json_dir.glob("*.json"))
        if not video_paths:
            return {}
        sample_video_name = video_paths[0].stem
        sample = checkpointer.load_evaluation_result(metric_name, model_name, sample_video_name)
        return {"unit": getattr(sample, "unit", None)}
    return factory

def read_cache_meta(cache_root: Path, task_name: str, metric_name: str, model_name: str) -> dict:
    """Read just the `meta` dict for a cache, without touching values/videos."""
    cache_path = cache_root / task_name / metric_name / f"{model_name}.json"
    return _load_cache(cache_path).get("meta", {})

# def sync_scalar_cache(
def cache_result(
    checkpointer: Checkpointer,
    task_name: str, # Ex: Kinematic_Distribution
    metric_name: str, # Ex: Velocity
    model_name: str, # Ex: MediaPipePoseEstimator
    reduction_function: Callable,
    force_rerun: bool = False,
    meta_factory: "Callable[[], dict] | None" = None,
) -> dict:
    """
    Returns {video_name: reduction} for all videos currently present on disk,
    reusing cached values and only computing reductions for new videos.
    `compute_video_scalar` may return a float (summary task) or a list of
    numbers, e.g. histogram bin counts (plotting task) -- both are plain
    JSON-serializable, so this function works unchanged for either shape.


    - New videos (on disk, not in cache): computed and added.
    - Removed videos (in cache, not on disk): dropped from the result.
    - method_version mismatch or force_rerun: cache ignored, full recompute.


    `meta_factory`, if given, is called ONCE whenever the cache is (re)built
    from scratch (stale/empty), and its return value is persisted alongside
    `values` as `meta` -- for things that should be captured once and reused
    across runs without needing a live object every time (e.g. `.unit`, or a
    frozen histogram bin width). Read back later via `read_cache_meta(...)`.
    """
    json_dir = checkpointer.evaluation_dir / metric_name / model_name
    current_videos = {p.stem for p in json_dir.glob("*.json")}

    visualization_cache_dir = checkpointer.visualization_cache_dir
    result_cache_path = visualization_cache_dir / task_name / metric_name / f"{model_name}.json"
    result_cache_path.mkdir(parents=True, exist_ok=True)
    # cache_path.parent.mkdir(parents=True, exist_ok=True)

    result = _load_cache(result_cache_path)

    values = {} if force_rerun else dict(result.get("values", {}))
    meta = dict(result.get("meta", {})) if not force_rerun else {}
    cached_videos = set(values.keys()) # videos for which results already exist

    to_add = sorted(current_videos - cached_videos) # if user added videos
    to_remove = cached_videos - current_videos # if user removed videos

    if not to_add and not to_remove and not force_rerun:
        print(f"  [up to date] {task_name}/{metric_name}/{model_name}: "
              f"{len(current_videos)} videos, no changes")
        return values

    print(f"  [sync:{force_rerun}] {task_name}/{metric_name}/{model_name}: "
          f"+{len(to_add)} -{len(to_remove)} "
          f"(cache had {len(cached_videos)}, disk has {len(current_videos)})")

    if force_rerun and meta_factory is not None:
        meta = meta_factory()

    for video_name in to_remove:
        values.pop(video_name, None)

    # read and cache result once
    for video_name in to_add:
        try:
            result = checkpointer.load_evaluation_result(
                metric_name, model_name, video_name
            )
            values[video_name] = reduction_function(result)
        except Exception as e:
            print(f"    [warn] failed to process {video_name}: {e}")

    result_cache_path.write_text(json.dumps(
        {"meta": meta, "values": values}
    ))
    return values

# ---------------------------------------------------------------------------
# Sparse (dict-based) histogram helpers -- no upper-range guess needed.
#
# Bin index = floor(abs(value) / bin_width), stored as {bin_index: count}.
# Unbounded range for free: an outlier just adds one dict key, nothing is
# silently dropped. Only `bin_width` needs choosing -- see
# `choose_fine_bin_width`, which derives it once from `kinematic_limit`
# (itself computed from the existing median cache) and freezes it via
# `sync_scalar_cache`'s `meta_factory`.
# ---------------------------------------------------------------------------

'''KINEMATIC DISTRIBUTION PLOT'''
def choose_fine_bin_width(kinematic_limit: float, n_bins: int, resolution_factor: int = 20) -> float:
    """
    Derive a fine bin width automatically from the same kinematic_limit the
    original code already computes (median-based). resolution_factor=20
    means the fine grid is 20x finer than the n_bins display buckets.
    """
    return kinematic_limit / (n_bins * resolution_factor)

def compute_video_sparse_histogram(values, bin_width: float) -> dict:
    """
    Per-video reduction: mask/NaN-clean, take abs, bucket into a sparse fine
    histogram. No clipping here -- clipping-equivalent behavior is applied
    later at display time in `rebin_sparse_to_display_buckets`, using
    whatever kinematic_limit is current at that point.
    """
    if isinstance(values, ma.MaskedArray):
        valid = values[~values.mask].data
    else:
        valid = np.asarray(values)
    valid = valid[~np.isnan(valid)]
    abs_vals = np.abs(valid).flatten()


    if abs_vals.size == 0:
        return {}


    bin_indices = np.floor(abs_vals / bin_width).astype(int)
    unique, counts = np.unique(bin_indices, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}

def merge_sparse_histograms(cache: dict) -> dict:
    """Sum sparse {bin_index_str: count} dicts across all videos."""
    merged: dict = {}
    for entry in cache.values():
        for k, v in entry.items():
            merged[k] = merged.get(k, 0) + v
    return merged

def rebin_sparse_to_display_buckets(
    merged_sparse: dict, bin_width: float, bin_edges: np.ndarray, bin_labels: list
) -> np.ndarray:
    """
    Collapse a sparse fine histogram into the original code's display
    buckets, replicating clip-then-histogram: any fine bin whose lower edge
    falls at or beyond the second-to-last coarse edge lands in the final
    open-ended ("> X") bucket -- exactly matching what
    `np.clip(values, -kinematic_limit, kinematic_limit)` followed by
    `np.histogram(..., bin_edges)` produces, but without ever discarding
    data at ingestion time.
    """
    coarse_counts = np.zeros(len(bin_labels))
    last_regular_edge = bin_edges[-2]


    for k, count in merged_sparse.items():
        idx = int(k)
        lo = idx * bin_width
        if lo >= last_regular_edge:
            coarse_counts[-1] += count
            continue
        bucket = np.searchsorted(bin_edges, lo, side="right") - 1
        bucket = min(max(bucket, 0), len(bin_labels) - 1)
        coarse_counts[bucket] += count


    total = coarse_counts.sum()
    return (coarse_counts / total * 100) if total > 0 else coarse_counts

# ---------------------------------------------------------------------------
# Task-specific per-video reduction functions, matching the existing
# MetricResult.aggregate(...) / aggregate_all(...) interface.
#
# Each takes a loaded MetricResult and returns a small, JSON-serializable
# payload -- a scalar, or a {"data": [...], "mask": [...]} vector -- that
# gets cached per video_name via sync_scalar_cache.
# ---------------------------------------------------------------------------

def _masked_values_to_payload(values):
    """Shared helper: MetricResult.values (np.ndarray or ma.MaskedArray) ->
    JSON-serializable {"data": [...], "mask": [...]}."""
    if isinstance(values, ma.MaskedArray):
        data = values.filled(np.nan)
        mask = ma.getmaskarray(values)
    else:
        data = np.asarray(values, dtype=np.float64)
        mask = np.isnan(data)
    return {"data": data.tolist(), "mask": mask.tolist()}

def make_plot1_video_reduction(coordinate_axis, frame_axis, person_axis,
                                convert_to_magnitude=True, method="median"):
    """
    Plot 1: optional magnitude over coordinate_axis, then reduce
    [frame_axis, person_axis] with `method` -> per-keypoint vector, masked.


    `convert_to_magnitude` and `method` are baked into the closure -- pass a
    matching method_version string (e.g. f"magnitude={convert_to_magnitude}_{method}")
    into sync_scalar_cache so changing either config invalidates the cache.
    """
    def compute(result: MetricResult):
        if convert_to_magnitude:
            result = result.aggregate([coordinate_axis], method="vector_magnitude")
        reduced = result.aggregate([frame_axis, person_axis], method=method)
        return _masked_values_to_payload(reduced.values)
    return compute


# def make_magnitude_median_reduction(coordinate_axis, method="median"):
def create_magnitude_values(coordinate_axis, method="median"):
    """
    Used by Kinematic Distribution Plot and Summary Table 
    Vector magnitude over coordinate_axis, then a single
    scalar over all axis.
    """
    def compute(metric_result):
        magnitude = metric_result.aggregate([coordinate_axis], method="vector_magnitude")
        return float(magnitude.aggregate_all(method=method))
    return compute

def make_raw_histogram_reduction(bin_width: float):
    """
    Plot 2 pass B equivalent: RAW (non-magnitude) values -> mask/NaN-clean,
    abs, sparse fine histogram. No clipping/kinematic_limit dependency here
    -- that's applied later at display time via rebin_sparse_to_display_buckets,
    so this reduction never needs to change even if kinematic_limit shifts.
    """
    def compute(metric_result):
        return compute_video_sparse_histogram(metric_result.values, bin_width)
    return compute

def combine_masked_vectors(cache: dict, method="median") -> np.ma.MaskedArray:
    """
    Combine cached {"data":[...], "mask":[...]} vectors across videos
    (Plot 1's cross-video step: ma.median(np.stack(model_values), axis=0)).
    """
    arrays = [
        ma.array(entry["data"], mask=entry["mask"])
        for entry in cache.values()
    ]
    stacked = ma.stack(arrays, axis=0)
    if method == "median":
        return ma.median(stacked, axis=0)
    elif method == "mean":
        return ma.mean(stacked, axis=0)
    else:
        raise ValueError(f"unsupported combine method: {method}")
