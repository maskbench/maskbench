import os
from typing import Dict

from matplotlib import pyplot as plt

from evaluation.metrics import MetricResult
from evaluation.plots import AccelerationDistributionPlot, CocoKeypointPlot, generate_result_table
from checkpointer import Checkpointer
from .base_visualizer import Visualizer


class MaskBenchVisualizer(Visualizer):
    """
    This class contains specific plots and tables for the MaskBench project evaluation. 
    """
        
    def generate_all_plots(self, pose_results: Dict[str, Dict[str, Dict[str, MetricResult]]]):
        os.makedirs(self.plots_dir, exist_ok=True)

        if "Acceleration" in pose_results.keys():
            acceleration_distribution_plot = AccelerationDistributionPlot()
            fig, filename = acceleration_distribution_plot.draw(pose_results)
            self._save_plot(fig, filename)

        if "Euclidean Distance" in pose_results.keys() and "Acceleration" in pose_results.keys():
            coco_keypoint_plot = CocoKeypointPlot(metric_names=['Euclidean Distance', 'Acceleration'])
            figures_and_names = coco_keypoint_plot.draw(pose_results)
            for fig, filename in figures_and_names:
                self._save_plot(fig, filename)

        generate_result_table(pose_results)
