from typing import Dict, List, Tuple
from itertools import cycle

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from evaluation.metrics.metric_result import COORDINATE_AXIS, MetricResult
from .plot import Plot


class KinematicDistributionPlot(Plot):
    """Plot class for visualizing kinematic distributions (velocity, acceleration, jerk) for different models."""
    
    def __init__(self, metric_name: str, kinematic_limit: float):
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
        # First handle masked values if it's a masked array
        if isinstance(values, ma.MaskedArray):
            valid_values = values[~values.mask].data
        else:
            valid_values = values
            
        # Remove NaN values
        valid_values = valid_values[~np.isnan(valid_values)]
        
        # Flatten the array
        flattened_values = valid_values.flatten()
        
        # Finally clip the values
        clipped_values = np.clip(flattened_values, -self.kinematic_limit, self.kinematic_limit)
        
        # print("Original shape:", values.shape)
        # print("Valid (non-masked, non-NaN) values shape:", valid_values.shape)
        # print("Final clipped values shape:", clipped_values.shape)
        
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
        bin_edges = np.linspace(0, self.kinematic_limit, self.n_bins + 1).astype(int)
        
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            if i == len(bin_edges) - 2:
                bin_labels.append(f'> {bin_edges[i]}')
            elif i == 0:
                bin_labels.append(f'< {bin_edges[i+1]}')
            else:
                bin_labels.append(f'[{bin_edges[i]},{bin_edges[i+1]}]')
                
        return bin_edges, bin_labels
    
    def draw(
        self,
        results: Dict[str, Dict[str, Dict[str, MetricResult]]],
    ) -> Tuple[plt.Figure, str]:
        pose_estimator_results = results[self.metric_name]
        
        # First pass: collect all kinematic values
        all_values = []
        for metric_results in pose_estimator_results.values():
            self.unit = next(iter(metric_results.values())).unit if self.unit is None else self.unit

            for metric_result in metric_results.values():
                magnitude_result = metric_result.aggregate([COORDINATE_AXIS], method='vector_magnitude')
                flattened_valid_clipped_vals = self._flatten_clip_validate(magnitude_result.values)
                all_values.extend(np.abs(flattened_valid_clipped_vals))
        
        if self.unit:
            self.config['xlabel'] = f'{self.metric_name} ({self.unit})'
        fig = self._setup_figure()


        # Store lines for updating legend later
        lines = []
        labels = []
        marker_cycle = cycle(self.markers)
        bin_edges, bin_labels = self._create_bin_edges_and_labels()
        
        # Create x positions that span the full width
        x_positions = np.linspace(0, 1, len(bin_labels))
        
        for model_name, metric_results in pose_estimator_results.items():
            model_values = []
            for metric_result in metric_results.values():
                values = metric_result.values
                flattened_valid_clipped_vals = self._flatten_clip_validate(values)
                model_values.extend(np.abs(flattened_valid_clipped_vals.flatten()))
                
            distribution = self._compute_distribution(model_values, bin_edges)
            
            marker = next(marker_cycle)
            
            plt.plot(x_positions, distribution, 
                    marker=marker,
                    markersize=6,
            )
            # plt.fill_between(x_positions, distribution, 
            #                alpha=0.2)
            
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
