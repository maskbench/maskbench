import os
import logging
from typing import Dict
import time
import numpy as np
from matplotlib import pyplot as plt

from evaluation.metrics import MetricResult
from evaluation.plots import KinematicDistributionPlot, CocoKeypointPlot, generate_result_table, InferenceTimePlot
from checkpointer import Checkpointer
from metric_result_class import COORDINATE_AXIS
from .base_visualizer import Visualizer


class MaskBenchVisualizer(Visualizer):
    """
    This class contains specific plots and tables for the MaskBench project evaluation. 
    """
        
    def generate_all_plots(self):
        '''
        lets confirm this but at this point i only care about the values, not name etc

        so what if i create one file, and then read line by line or smth
        '''
        os.makedirs(self.plots_dir, exist_ok=True)
        evaluated_metric_names = [d.name for d in self.checkpointer.evaluation_dir.iterdir() if d.is_dir()]

        start_time = time.time()
        # this is needed by kinematic plots and result table. so we generate it once
        metric_magnitude_values = self.calculate_kinematic_magnitudes(evaluated_metric_names)
        mid_time = time.time()
        print('took', mid_time - start_time)
        table_df = generate_result_table(metric_magnitude_values)
        self._save_table(table_df, "result_table.csv")
        print('final took', time.time() - mid_time)

        # metric_to_visualization_mapping = {
        #     'Velocity': [KinematicDistributionPlot],
        #     'Acceleration': [KinematicDistributionPlot, CocoKeypointPlot],
        #     'Jerk': [KinematicDistributionPlot],
        # }

        # for metric, visualizations in metric_to_visualization_mapping.items():
        #     for visualization in visualizations:
        #         plot = visualization(metric_name=metric)
        #         fig, filename = plot.draw(None)
        #         self._save_plot(fig, filename)

        # metric_results = self.checkpointer.load_all_evaluation_results()
        # if metric_results is None:
        #     logging.error(f'Metric Results not Found. Skipping Plot Generation')
        #     return

        '''
            # if "Velocity" in metric_results.keys():
            #     velocity_distribution_plot = KinematicDistributionPlot(metric_name="Velocity")
            #     fig, filename = velocity_distribution_plot.draw(metric_results, add_title=False)
            #     self._save_plot(fig, filename)
            # if "Acceleration" in metric_results.keys():
            #     acceleration_distribution_plot = KinematicDistributionPlot(metric_name="Acceleration")
            #     fig, filename = acceleration_distribution_plot.draw(metric_results, add_title=False)
            #     self._save_plot(fig, filename)

            #     coco_keypoint_plot = CocoKeypointPlot(metric_name="Acceleration")
            #     fig, filename = coco_keypoint_plot.draw(metric_results, add_title=False)
            #     self._save_plot(fig, filename)

            # if "Jerk" in metric_results.keys():
            #     jerk_distribution_plot = KinematicDistributionPlot(metric_name="Jerk")
            #     fig, filename = jerk_distribution_plot.draw(metric_results, add_title=False)
            #     self._save_plot(fig, filename)
        '''

        # inference_times = self.checkpointer.load_inference_times()
        # if inference_times:
        #     inference_times = self.set_maskanyone_ui_inference_times(inference_times)
        #     inference_times = self.sort_inference_times_pose_estimator_order(inference_times, metric_results)
        #     inference_time_plot = InferenceTimePlot()
        #     fig, filename = inference_time_plot.draw(inference_times)
        #     self._save_plot(fig, filename)
        '''
        - all need magnitude per video
        - aggregate returns a scalar value
        - 
        coco keypoint:
            for video_name, metric_result in video_results.items():
                    if self.convert_to_magnitude:
                        metric_result= provided magnitude value
                    median_video_keypoint_values = metric_result.aggregate([FRAME_AXIS, PERSON_AXIS], method='median').values
                    model_values.append(median_video_keypoint_values)
                median_model_keypoint_values = ma.median(np.stack(model_values, axis=0), axis=0)
        kinematic:
        for video_name, metric_result in video_results.items():
        #         magnitude_result = provided

        #         video_magnitudes.append(magnitude_result.aggregate_all(method='median'))
        #     pose_estimator_medians[pose_estimator_name] = np.median(video_magnitudes)
        result_table:
        for video_name in evaluated_video_names:
                magnitude_values = result.aggregate([COORDINATE_AXIS], method='vector_magnitude')
                metric_magnitude_values.setdefault(metric_name, {}).setdefault(model_name, {})[video_name] = magnitude_values
        '''
        
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

    # def calculate_kinematic_magnitudes(self, metric_results: Dict[str, Dict[str, Dict[str, MetricResult]]]) -> Dict[str, Dict[str, Dict[str, MetricResult]]]:
    def calculate_kinematic_magnitudes(self, evaluated_metric_names) -> Dict[str, Dict[str, Dict[str, MetricResult]]]:
        """
        Calculate the vector magnitude with respect to COORINATE AXIS per video, per model, per metric.
        Metric Result Values lose 1 dimesion, ex: (380, 1, 33, 2) -> (380, 1, 33) 
        """
        # valid_metrics = ["Velocity", "Acceleration", "Jerk"]
        valid_metrics = ["Velocity"]
        metric_magnitude_values = {}
        for metric_name in valid_metrics: # TODO hard-coded for now
            if metric_name not in evaluated_metric_names:
                continue

            metric_dir = self.checkpointer.evaluation_dir / metric_name
            print('Traversing metric dir:', metric_dir)
            evaluated_model_names = [d.name for d in metric_dir.iterdir() if d.is_dir()]

            for model_name in evaluated_model_names:
                model_dir = metric_dir / model_name
                evaluated_video_names = list(model_dir.glob('*.json'))
                print('Traversing model dir:', model_dir)
                print('found ', len(evaluated_video_names))

                video_magnitudes = []   
                for video_name in evaluated_video_names:
                    video_name = video_name.stem
                    if not self.checkpointer.exists_evaluation_result(metric_name, model_name, video_name):
                        continue

                    result = self.checkpointer.load_evaluation_result(metric_name, model_name, video_name)
                    magnitude_values = result.aggregate([COORDINATE_AXIS], method='vector_magnitude')
                    print(result.values.shape)
                    print(magnitude_values.values.shape)
                    metric_magnitude_values.setdefault(metric_name, {}).setdefault(model_name, {})[video_name] = magnitude_values
                    print(metric_magnitude_values)

                    break

        return metric_magnitude_values

    def sort_inference_times_pose_estimator_order(self, inference_times: Dict[str, Dict[str, float]], metric_results: Dict[str, Dict[str, Dict[str, MetricResult]]]) -> Dict[str, Dict[str, float]]:
        """
        Sort the inference times according to the order in metric_results.
        
        Args:
            inference_times: Dictionary containing inference times for each pose estimator
            metric_results: Dictionary containing pose estimation results, used to determine the order
            
        Returns:
            Dictionary containing sorted inference times
        """
        # Get the list of pose estimators from any metric in metric_results
        first_metric = next(iter(metric_results))
        pose_estimator_order = list(metric_results[first_metric].keys())
        
        sorted_inference_times = {}
        for pose_estimator in pose_estimator_order:
            if pose_estimator in inference_times:
                sorted_inference_times[pose_estimator] = inference_times[pose_estimator]
        return sorted_inference_times

        
