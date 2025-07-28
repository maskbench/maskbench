import os
from typing import Dict

from matplotlib import pyplot as plt

from evaluation.metrics import MetricResult
from evaluation.plots import KinematicDistributionPlot, CocoKeypointPlot, generate_result_table, InferenceTimePlot
from checkpointer import Checkpointer
from evaluation.metrics.metric_result import COORDINATE_AXIS
from .base_visualizer import Visualizer


class MaskBenchVisualizer(Visualizer):
    """
    This class contains specific plots and tables for the MaskBench project evaluation. 
    """
        
    def generate_all_plots(self, pose_results: Dict[str, Dict[str, Dict[str, MetricResult]]]):
        os.makedirs(self.plots_dir, exist_ok=True)

        if "Velocity" in pose_results.keys():
            velocity_distribution_plot = KinematicDistributionPlot(metric_name="Velocity")
            fig, filename = velocity_distribution_plot.draw(pose_results, add_title=False)
            self._save_plot(fig, filename)

        if "Acceleration" in pose_results.keys():
            acceleration_distribution_plot = KinematicDistributionPlot(metric_name="Acceleration")
            fig, filename = acceleration_distribution_plot.draw(pose_results, add_title=False)
            self._save_plot(fig, filename)

            coco_keypoint_plot = CocoKeypointPlot(metric_name="Acceleration")
            fig, filename = coco_keypoint_plot.draw(pose_results, add_title=False)
            self._save_plot(fig, filename)

        if "Jerk" in pose_results.keys():
            jerk_distribution_plot = KinematicDistributionPlot(metric_name="Jerk")
            fig, filename = jerk_distribution_plot.draw(pose_results, add_title=False)
            self._save_plot(fig, filename)

        if "Euclidean Distance" in pose_results.keys():
            coco_keypoint_plot = CocoKeypointPlot(metric_name="Euclidean Distance")
            fig, filename = coco_keypoint_plot.draw(pose_results, add_title=False)
            self._save_plot(fig, filename)

        inference_times = self.checkpointer.load_inference_times()
        if inference_times:
            inference_times = self.set_maskanyone_ui_inference_times(inference_times)
            inference_times = self.sort_inference_times_pose_estimator_order(inference_times, pose_results)
            inference_time_plot = InferenceTimePlot()
            fig, filename = inference_time_plot.draw(inference_times)
            self._save_plot(fig, filename)

        pose_results = self.calculate_kinematic_magnitudes(pose_results)
        table_df = generate_result_table(pose_results)
        self._save_table(table_df, "result_table.csv")

        
    def set_maskanyone_ui_inference_times(self, inference_times: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Set the inference times for MaskAnyoneUI to be equal to the corresponding MaskAnyoneAPI models.
        """
        # Create a copy to avoid modifying the original
        mapped_times = inference_times.copy()
        
        # Define the mapping pairs
        ui_to_api_mapping = {
            'MaskAnyoneUI-MediaPipe': 'MaskAnyoneAPI-MediaPipe',
            'MaskAnyoneUI-OpenPose': 'MaskAnyoneAPI-OpenPose'
        }
        
        # For each UI model, set its times to the corresponding API model
        for ui_model, api_model in ui_to_api_mapping.items():
            if ui_model in inference_times and api_model in inference_times:
                mapped_times[ui_model] = mapped_times[api_model].copy()
                    
        return mapped_times

    def calculate_kinematic_magnitudes(self, pose_results: Dict[str, Dict[str, Dict[str, MetricResult]]]) -> Dict[str, Dict[str, Dict[str, MetricResult]]]:
        """
        Calculate the magnitude of the kinematic metrics.
        """
        for metric_name in ["Velocity", "Acceleration", "Jerk"]:
            if metric_name in pose_results.keys():
                for model_name, video_results in pose_results[metric_name].items():
                    for video_name, metric_result in video_results.items():
                        magnitude_values = metric_result.aggregate([COORDINATE_AXIS], method='vector_magnitude')
                        pose_results[metric_name][model_name][video_name] = magnitude_values
        return pose_results

    def sort_inference_times_pose_estimator_order(self, inference_times: Dict[str, Dict[str, float]], pose_results: Dict[str, Dict[str, Dict[str, MetricResult]]]) -> Dict[str, Dict[str, float]]:
        """
        Sort the inference times according to the order in pose_results.
        
        Args:
            inference_times: Dictionary containing inference times for each pose estimator
            pose_results: Dictionary containing pose estimation results, used to determine the order
            
        Returns:
            Dictionary containing sorted inference times
        """
        # Get the list of pose estimators from any metric in pose_results
        first_metric = next(iter(pose_results))
        pose_estimator_order = list(pose_results[first_metric].keys())
        
        sorted_inference_times = {}
        for pose_estimator in pose_estimator_order:
            if pose_estimator in inference_times:
                sorted_inference_times[pose_estimator] = inference_times[pose_estimator]
        return sorted_inference_times

        
