import os
from typing import Dict

from matplotlib import pyplot as plt

from evaluation.metrics import MetricResult
from evaluation.plots import KinematicDistributionPlot, CocoKeypointPlot, generate_result_table
from checkpointer import Checkpointer
from .base_visualizer import Visualizer


class MaskBenchVisualizer(Visualizer):
    """
    This class contains specific plots and tables for the MaskBench project evaluation. 
    """
        
    def generate_all_plots(self, pose_results: Dict[str, Dict[str, Dict[str, MetricResult]]]):
        os.makedirs(self.plots_dir, exist_ok=True)

        if "Velocity" in pose_results.keys():
            velocity_distribution_plot = KinematicDistributionPlot(metric_name="Velocity", kinematic_limit=500)
            fig, filename = velocity_distribution_plot.draw(pose_results)
            self._save_plot(fig, filename)

        if "Acceleration" in pose_results.keys():
            acceleration_distribution_plot = KinematicDistributionPlot(metric_name="Acceleration", kinematic_limit=3000)
            fig, filename = acceleration_distribution_plot.draw(pose_results)
            self._save_plot(fig, filename)

            coco_keypoint_plot = CocoKeypointPlot(metric_name="Acceleration")
            fig, filename = coco_keypoint_plot.draw(pose_results)
            self._save_plot(fig, filename)

        if "Jerk" in pose_results.keys():
            jerk_distribution_plot = KinematicDistributionPlot(metric_name="Jerk", kinematic_limit=30000)
            fig, filename = jerk_distribution_plot.draw(pose_results)
            self._save_plot(fig, filename)

        if "Euclidean Distance" in pose_results.keys():
            coco_keypoint_plot = CocoKeypointPlot(metric_name="Euclidean Distance")
            fig, filename = coco_keypoint_plot.draw(pose_results)
            self._save_plot(fig, filename)

        table = generate_result_table(pose_results)
        self._save_table(table, "result_table.txt")
