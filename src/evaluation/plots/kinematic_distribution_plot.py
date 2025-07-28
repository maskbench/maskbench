from typing import Dict, List, Tuple
from itertools import cycle

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from evaluation.metrics.metric_result import COORDINATE_AXIS, MetricResult
from .plot import Plot


class KinematicDistributionPlot(Plot):
    """Plot class for visualizing kinematic distributions (velocity, acceleration, jerk) for different models."""
    
    def __init__(self, metric_name: str, kinematic_limit: float = None):
        """
        Initialize the kinematic distribution plot.
        
        Args:
            metric_name: Name of the kinematic metric ('Velocity', 'Acceleration', or 'Jerk')
            kinematic_limit: Optional limit for the kinematic values. All values greater than this value will land in one bucket.
        """
        if metric_name not in ['Velocity', 'Acceleration', 'Jerk']:
            raise ValueError(f"Metric name must be one of ['Velocity, 'Acceleration', 'Jerk']")
            
        super().__init__(
            name=f"{metric_name}Distribution",
            config={
                'title': f'{metric_name} Keypoint Distribution',
                'xlabel': f'{metric_name}',
                'ylabel': 'Percentage',
            }
        )
        
        self.metric_name = metric_name
        self.unit = None
        self.n_bins = 10
        self.kinematic_limit = kinematic_limit
        
        # Define a variety of marker shapes for different models
        # o: circle, s: square, ^: triangle up, v: triangle down, 
        # D: diamond, p: pentagon, h: hexagon, 8: octagon,
        # *: star, P: plus filled
        self.markers = ['^', '*', 'h', 's', 'D', 'o', 'p', 'h', '8', 'P']

    def _flatten_clip_validate(self, values: np.ndarray) -> np.ndarray:
        """
        Process input values by:
        1. Removing masked and NaN values
        2. Flattening the array
        3. Clipping values to the kinematic limit
        
        Args:
            values: Input numpy masked array with potential NaN values
            
        Returns:
            Flattened array of valid values clipped to kinematic limit
        """
        # Handle masked values if it's a masked array
        if isinstance(values, ma.MaskedArray):
            valid_values = values[~values.mask].data
        else:
            valid_values = values
            
        valid_values = valid_values[~np.isnan(valid_values)] # Remove NaN values
        flattened_values = valid_values.flatten()
        clipped_values = np.clip(flattened_values, -self.kinematic_limit, self.kinematic_limit)
        return clipped_values

    def _compute_distribution(self, values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(values, bins=bin_edges)
        return (hist / len(values)) * 100
    
    def _create_bin_edges_and_labels(self) -> Tuple[np.ndarray, List[str]]:
        """
        Create bin edges and corresponding labels for the kinematic distribution.
        
        Returns:
            Tuple containing:
                - np.ndarray: Bin edges for histogram computation
                - List[str]: Human-readable labels for the bins
        """
        diff = self.kinematic_limit / self.n_bins
        bin_edges = np.linspace(0, self.kinematic_limit + diff, self.n_bins + 2).astype(int)
        
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            if i == len(bin_edges) - 2:
                bin_labels.append(f'> {bin_edges[i]}')
            else:
                bin_labels.append(f'[{bin_edges[i]},{bin_edges[i+1]}]')
                
        return bin_edges, bin_labels

    def _round_to_nearest_magnitude(self, value: float) -> float:
        """Round a value to the nearest magnitude based on its range.
        
        For values:
        - Between 0-100: Round to nearest 10
        - Between 100-1000: Round to nearest 100 
        - Between 1000-10000: Round to nearest 1000
        And so on up to 1,000,000
        """
        if value <= 0:
            return 0
            
        magnitude = 10 ** (len(str(int(value))) - 1)
        if magnitude < 10:
            magnitude = 10
        return np.ceil(value / magnitude) * magnitude
    
    def draw(
        self,
        results: Dict[str, Dict[str, Dict[str, MetricResult]]],
        add_title: bool = True,
    ) -> Tuple[plt.Figure, str]:

        # First pass: compute the magnitude of the kinematic values and average over videos
        pose_estimator_results = results[self.metric_name]
        pose_estimator_medians = {} # store the average for each pose estimator over all videos
        pose_estimator_magnitude_results = {} # store the magnitude results for each pose estimator

        for pose_estimator_name, video_results in pose_estimator_results.items():
            self.unit = next(iter(video_results.values())).unit if self.unit is None else self.unit

            video_magnitudes = []
            pose_estimator_magnitude_results[pose_estimator_name] = {}
            for video_name, metric_result in video_results.items():
                magnitude_result = metric_result.aggregate([COORDINATE_AXIS], method='vector_magnitude')

                video_magnitudes.append(magnitude_result.aggregate_all())
                pose_estimator_magnitude_results[pose_estimator_name][video_name] = magnitude_result

            pose_estimator_medians[pose_estimator_name] = np.median(video_magnitudes)

        if self.kinematic_limit is None:
            # Calculate the maximum average magnitude of all pose estimators to set the bounds of the plot
            # This maximum value is increased by 20%
            raw_limit = max(pose_estimator_medians.values()) * 1.20
            self.kinematic_limit = self._round_to_nearest_magnitude(raw_limit)
        
        if self.unit:
            self.config['xlabel'] = f'{self.metric_name} ({self.unit})'
        fig = self._setup_figure(add_title=add_title)

        # Store lines for updating legend later
        lines = []
        labels = []
        marker_cycle = cycle(self.markers)
        bin_edges, bin_labels = self._create_bin_edges_and_labels()
        
        # Create x positions that span the full width
        x_positions = np.linspace(0, 1, len(bin_labels))
        
        # Second pass: flatten and clip the values
        for model_name, video_results in pose_estimator_results.items():
            model_values = []
            for metric_result in video_results.values():
                values = metric_result.values
                flattened_valid_clipped_vals = self._flatten_clip_validate(values)
                model_values.extend(np.abs(flattened_valid_clipped_vals.flatten()))
                
            distribution = self._compute_distribution(model_values, bin_edges)
            
            marker = next(marker_cycle)
            plt.plot(x_positions, distribution, 
                    marker=marker,
                    markersize=6,
            )
            
            scatter = plt.scatter([], [], 
                                marker=marker,
                                s=36,
                                label=model_name,
                                color=plt.gca().lines[-1].get_color())
            lines.append(scatter)
            labels.append(model_name)
        
        self._finish_label_grid_axes_styling(x_positions, bin_labels, lines, labels)
        return fig, f"{self.metric_name.lower()}_distribution"

    def _finish_label_grid_axes_styling(self, x_positions: np.ndarray, bin_labels: List[str], lines: List[plt.Line2D], labels: List[str]):
        # Configure x-axis
        plt.xlim(-0.01, 1.01)
        plt.xticks(x_positions, bin_labels, rotation=45)
        
        # Configure y-axis
        y_ticks = np.arange(0, 90, 20)
        plt.yticks(y_ticks, [f'{x:.2f} %' for x in y_ticks])
        plt.ylim(0, 90)
        
        # Configure grid
        plt.grid(True, axis='y', alpha=0.3, linestyle='-', color='gray')
        plt.grid(False, axis='x')
        
        # Ensure all text is black and properly sized
        plt.tick_params(colors='black', which='both')
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_color('black')
        
        plt.legend(lines, labels)
