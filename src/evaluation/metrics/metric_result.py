from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.ma as ma

# Standard axis names (but any string is allowed)
FRAME_AXIS = 'frame'
PERSON_AXIS = 'person'
KEYPOINT_AXIS = 'keypoint'


class MetricResult:
    """
    Stores the result of a metric computation for one video.
    
    Supports n-dimensional results (e.g. frame x person x keypoint) with named axes
    and flexible aggregation methods. Uses masked arrays to handle variable numbers
    of persons or missing data.
    """
    
    def __init__(
        self,
        values: Union[np.ndarray, ma.MaskedArray],
        axis_names: List[str],
        metric_name: str,
        video_name: str,
        model_name: Optional[str] = None,
    ):
        """
        Initialize a metric result.
        
        Args:
            values: N-dimensional array of metric values. If not masked, will be converted
                   to masked array. Shape must match the number of axis_names provided.
            axis_names: List of names for each dimension (e.g. ['frame', 'person', 'keypoint'])
            metric_name: Name of the metric (e.g. 'mpjpe', 'acceleration_error')
            video_name: Name of the video this result is for
            model_name: Optional name of the model that produced these results
        """
        # Convert to masked array if needed
        if not isinstance(values, ma.MaskedArray):
            values = ma.array(values)
            
        if len(axis_names) != values.ndim:
            raise ValueError(f"Number of axis names ({len(axis_names)}) must match "
                           f"number of dimensions in values ({values.ndim})")
            
        self.values = values
        self.axis_names = axis_names
        self.metric_name = metric_name
        self.video_name = video_name
        self.model_name = model_name
        
        # Create axis name to dimension mapping for easier lookup
        self.axis_name_to_dim = {name: i for i, name in enumerate(axis_names)}
        
    def aggregate(
        self,
        dims: Union[str, List[str]],
        method: str = 'mean'
    ) -> 'MetricResult':
        """
        Aggregate the metric values along specified dimensions.
        
        Args:
            dims: Dimension name(s) to aggregate over (e.g. 'frame' or ['frame', 'person'])
            method: Aggregation method ('mean', 'rmse','median', 'sum', 'min', 'max')
            
        Returns:
            New MetricResult with aggregated values.
        """
        if isinstance(dims, str):
            dims = [dims]
            
        # Convert dimension names to axis numbers
        axes = [self.axis_name_to_dim[dim] for dim in dims]
        
        # Get remaining axes after aggregation
        remaining_axes = [
            name for i, name in enumerate(self.axis_names)
            if i not in axes
        ]
        
        if method == 'mean':
            new_values = ma.mean(self.values, axis=tuple(axes))
        elif method == 'rmse':
            new_values = ma.sqrt(ma.mean(self.values**2, axis=tuple(axes)))
        elif method == 'median':
            new_values = ma.median(self.values, axis=tuple(axes))
        elif method == 'sum':
            new_values = ma.sum(self.values, axis=tuple(axes))
        elif method == 'min':
            new_values = ma.min(self.values, axis=tuple(axes))
        elif method == 'max':
            new_values = ma.max(self.values, axis=tuple(axes))
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
            
        return MetricResult(
            values=new_values,
            axis_names=remaining_axes,
            metric_name=self.metric_name,
            video_name=self.video_name,
            model_name=self.model_name,
        )
    
    def aggregate_all(self) -> float:
        """
        Get a single scalar value by averaging over all dimensions.
        Useful for getting an overall score for the video.
        """
        return float(ma.mean(self.values))
    
    def get_values_aggregated_to_axis(self, axis_name: str) -> np.ndarray:
        """
        Get the values aggregated over all dimensions except the specified one.
        Useful for plotting metrics over time (frames) or analyzing per-keypoint errors.
        """
        other_dims = [name for name in self.axis_names if name != axis_name]
        return self.aggregate(other_dims).values
