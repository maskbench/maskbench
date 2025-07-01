from typing import Dict, List, Tuple
import os
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.metrics.metric_result import MetricResult, FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS
from .plot import Plot


class AccelerationDistributionPlot(Plot):
    """Plot class for visualizing acceleration distribution for different models over all videos of the dataset."""
    
    def __init__(self):
        # Configure plot settings through parent class
        super().__init__(
            name="AccelerationDistribution",
            config={
                'xlabel': 'Acceleration Error',
                'ylabel': 'Percentage',
            }
        )
        
        self.n_bins = 10
        self.accel_limit = 1000  # Limit for acceleration values
        
        # Define a variety of marker shapes for different models
        # o: circle, s: square, ^: triangle up, v: triangle down, 
        # D: diamond, p: pentagon, h: hexagon, 8: octagon,
        # *: star, P: plus filled
        self.markers = ['^', '*', 'h', 's' 'D', 'o', 'p', 'h', '8', 'P']

    def _clip_accelerations(self, accelerations: np.ndarray) -> np.ndarray:
        return np.clip(accelerations, -self.accel_limit, self.accel_limit)

    def _compute_distribution(self, accelerations: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(accelerations, bins=bin_edges)
        return (hist / len(accelerations)) * 100
    
    def _create_bin_edges_and_labels(self) -> Tuple[np.ndarray, List[str]]:
        """
        Create bin edges and corresponding labels for the acceleration distribution.
        
        Returns:
            Tuple containing:
                - np.ndarray: Bin edges for histogram computation
                - List[str]: Human-readable labels for the bins
        """
        bin_edges = np.linspace(0, self.accel_limit, self.n_bins + 1).astype(int)
        
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
    ) -> None:
        if 'Acceleration' not in results:
            raise ValueError("No acceleration metric found in results")
            
        acceleration_results = results['Acceleration']
        self._setup_figure()
        
        # First pass: collect all acceleration values
        all_accelerations = []
        for video_results in acceleration_results.values():
            for video_result in video_results.values():
                accelerations = video_result.values
                clipped_accels = self._clip_accelerations(accelerations)
                all_accelerations.extend(np.abs(clipped_accels.flatten()))
        
        bin_edges, bin_labels = self._create_bin_edges_and_labels()
        
        marker_cycle = cycle(self.markers)
        
        # Store lines for updating legend later
        lines = []
        labels = []
        
        # Create x positions that span the full width
        x_positions = np.linspace(0, 1, len(bin_labels))
        
        for model_name, video_results in acceleration_results.items():
            model_accelerations = []
            for video_result in video_results.values():
                accelerations = video_result.values
                clipped_accels = self._clip_accelerations(accelerations)
                model_accelerations.extend(np.abs(clipped_accels.flatten()))
                
            distribution = self._compute_distribution(model_accelerations, bin_edges)
            
            marker = next(marker_cycle)
            
            plt.plot(x_positions, distribution, 
                    marker=marker,
                    markersize=6,
            )
            plt.fill_between(x_positions, distribution, 
                           alpha=0.2)
            
            scatter = plt.scatter([], [], 
                                marker=marker,
                                s=36,
                                label=model_name,
                                color=plt.gca().lines[-1].get_color())
            lines.append(scatter)
            labels.append(model_name)
        
        self._finish_label_grid_axes_styling(x_positions, bin_labels, lines, labels)
        self._save_plot("acceleration_distribution.png") 


    def _finish_label_grid_axes_styling(self, x_positions: np.ndarray, bin_labels: List[str], lines: List[plt.Line2D], labels: List[str]):
        # Configure x-axis
        plt.xlim(0, 1.00)
        plt.xticks(x_positions, bin_labels, rotation=45)
        
        # Configure y-axis
        y_ticks = np.arange(0, 81, 20)
        plt.yticks(y_ticks, [f'{x:.2f} %' for x in y_ticks])
        plt.ylim(0, 80)
        
        # Configure grid
        plt.grid(True, axis='y', alpha=0.3, linestyle='-', color='gray')
        plt.grid(False, axis='x')
        
        # Ensure all text is black and properly sized
        plt.tick_params(colors='black', which='both')
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_color('black')
        
        plt.legend(lines, labels)
