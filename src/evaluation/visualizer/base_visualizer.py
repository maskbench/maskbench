from abc import abstractmethod
import os
from typing import Dict

from matplotlib import pyplot as plt

from evaluation.metrics import MetricResult
from evaluation.plots import AccelerationDistributionPlot, CocoKeypointPlot, generate_result_table
from checkpointer import Checkpointer


class Visualizer:
    def __init__(self, checkpointer: Checkpointer):
        """
        Base class for all visualizers.
        
        Args:
            checkpointer: Checkpointer instance to handle saving plots
        """
        self.checkpointer = checkpointer
        self.plots_dir = os.path.join(self.checkpointer.checkpoint_dir, "plots")
        
    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        """Save a matplotlib figure to the plots directory."""
        output_path = os.path.join(self.plots_dir, filename)
        fig.savefig(output_path, bbox_inches='tight', dpi=300)

    @abstractmethod
    def generate_all_plots(self, pose_results: Dict[str, Dict[str, Dict[str, MetricResult]]]):
        """
        Generate and save all plots.
        Args:
            pose_results: Dictionary containing pose results for each metric, model, and video.
        """
        pass