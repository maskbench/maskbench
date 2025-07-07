import os
from typing import Dict

from matplotlib import pyplot as plt

from evaluation.metrics import MetricResult
from evaluation.plots import AccelerationDistributionPlot, AccelerationOverTimePlot
from checkpointer import Checkpointer


class Visualizer:
    def __init__(self, checkpointer: Checkpointer):
        """
        Initialize the Visualizer.
        
        Args:
            checkpointer: Checkpointer instance to handle saving plots
        """
        self.checkpointer = checkpointer
        self.plots_dir = os.path.join(self.checkpointer.checkpoint_dir, "plots")
        
    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        """Save a matplotlib figure to the plots directory."""
        output_path = os.path.join(self.plots_dir, filename)
        fig.savefig(output_path, bbox_inches='tight', dpi=300)

    def save_plots(self, pose_results: Dict[str, Dict[str, Dict[str, MetricResult]]]):
        """Generate and save all plots."""
        os.makedirs(self.plots_dir, exist_ok=True)

        acceleration_distribution_plot = AccelerationDistributionPlot()
        fig, filename = acceleration_distribution_plot.draw(pose_results)
        self._save_plot(fig, filename)