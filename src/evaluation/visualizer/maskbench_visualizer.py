import os
from typing import Dict

from evaluation.plots import KinematicDistributionPlot, CocoKeypointPlot, generate_result_table, InferenceTimePlot
from .base_visualizer import Visualizer

class MaskBenchVisualizer(Visualizer):
    """
    This class contains specific plots and tables for the MaskBench project evaluation. 
    """
        
    def generate_all_plots(self):
        os.makedirs(self.plots_dir, exist_ok=True)

        velocity_distribution_plot = KinematicDistributionPlot(metric_name="Velocity", checkpointer=self.checkpointer)
        fig, filename = velocity_distribution_plot.draw(add_title=False)
        self._save_plot(fig, filename)

        acceleration_distribution_plot = KinematicDistributionPlot(metric_name="Acceleration", checkpointer=self.checkpointer)
        fig, filename = acceleration_distribution_plot.draw(add_title=False)
        self._save_plot(fig, filename)

        coco_keypoint_plot = CocoKeypointPlot(metric_name="Acceleration", checkpointer=self.checkpointer)
        fig, filename = coco_keypoint_plot.draw(add_title=False)
        self._save_plot(fig, filename)

        jerk_distribution_plot = KinematicDistributionPlot(metric_name="Jerk", checkpointer=self.checkpointer)
        fig, filename = jerk_distribution_plot.draw(add_title=False)
        self._save_plot(fig, filename)

        inference_times = self.checkpointer.load_inference_times()
        if inference_times:
            inference_times = self.set_maskanyone_ui_inference_times(inference_times)
            inference_times = self.sort_inference_times_pose_estimator_order(inference_times)
            inference_time_plot = InferenceTimePlot()
            fig, filename = inference_time_plot.draw(inference_times)
            self._save_plot(fig, filename)

        table_df = generate_result_table(self.checkpointer)
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

    def sort_inference_times_pose_estimator_order(self, inference_times: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Sort the inference times according to the order in metric_results.
        
        Args:
            inference_times: Dictionary containing inference times for each pose estimator
            metric_results: Dictionary containing pose estimation results, used to determine the order
            
        Returns:
            Dictionary containing sorted inference times
        """
        
        sorted_inference_times = {}
        for pose_estimator in inference_times:
            sorted_inference_times[pose_estimator] = inference_times[pose_estimator]
        return sorted_inference_times

        
