from typing import Dict

from evaluation.metrics import MetricResult
from evaluation.plots import AccelerationDistributionPlot, AccelerationOverTimePlot


class Visualizer:
    def save_plots(self, pose_results: Dict[str, Dict[str, Dict[str, MetricResult]]]):
        acceleration_plot = AccelerationOverTimePlot()
        acceleration_plot.draw(pose_results)

        acceleration_distribution_plot = AccelerationDistributionPlot()
        acceleration_distribution_plot.draw(pose_results)