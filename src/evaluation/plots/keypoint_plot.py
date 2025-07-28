from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from evaluation.metrics.metric_result import COORDINATE_AXIS, FRAME_AXIS, PERSON_AXIS, MetricResult
from keypoint_pairs import COCO_KEYPOINT_NAMES
from .plot import Plot


class CocoKeypointPlot(Plot):
    """Plot class for visualizing COCO keypoint metrics for different models. Assumes that all metrics have the same number of keypoints."""
    
    def __init__(self, metric_name: str):
        super().__init__(
            name=f"CocoKeypointPlot",
            config={
                'title': f'Coco Keypoint Plot',
                'xlabel': 'Keypoint',
                'ylabel': 'Median Value',
                'style': 'white',
                'figsize': (18, 6),
            }
        )
        self.metric_name = metric_name
        self.convert_to_magnitude = False
        if self.metric_name in ["Velocity", "Acceleration", "Jerk"]:
            self.convert_to_magnitude = True
    
    def draw(
        self,
        results: Dict[str, Dict[str, Dict[str, MetricResult]]],
        add_title: bool = True,
    ) -> Tuple[plt.Figure, str]:
        """
        Draw keypoint plot for the given metric. Assumes that all metric results for all pose estimators have the same number of keypoints.
        
        Args:
            results: Dictionary mapping:
                    metric_name -> model_name -> video_name -> MetricResult
            add_title: Whether to add the title to the plot (default: True)
                    
        Returns:
            Tuple[plt.Figure, str]: The figure and suggested filename.
        """
        if self.metric_name not in results:
            raise ValueError(f"Metric {self.metric_name} not found in results.")
        
        metric_results = results[self.metric_name]
        plot_data = []
        
        for model_name, video_results in metric_results.items():
            model_values = []
            unit = next(iter(video_results.values())).unit # get the unit of the first video result

            for video_name, metric_result in video_results.items():
                if self.convert_to_magnitude:
                    metric_result = metric_result.aggregate([COORDINATE_AXIS], method='vector_magnitude')
                avg_video_keypoint_values = metric_result.aggregate([FRAME_AXIS, PERSON_AXIS], method='mean').values
                model_values.append(avg_video_keypoint_values)
            median_model_keypoint_values = np.median(np.stack(model_values, axis=0), axis=0)
            
            for keypoint_idx, value in enumerate(median_model_keypoint_values):
                if keypoint_idx not in COCO_KEYPOINT_NAMES.keys():
                    continue

                plot_data.append({
                    'Model': model_name,
                    'Keypoint': f"{COCO_KEYPOINT_NAMES[keypoint_idx]} ({keypoint_idx})",
                    'Median Value': value
                })
        
        df = pd.DataFrame(plot_data)
        
        self.config['title'] = f'{self.metric_name} by Keypoint and Model'
        self.config['style'] = 'grid'
        if unit:
            self.config['ylabel'] = f'Median {self.metric_name} ({unit})'
        fig = self._setup_figure(add_title=add_title)
        
        sns.barplot(
            data=df,
            x='Keypoint',
            y='Median Value',
            hue='Model',
        )

        plt.xticks(rotation=45, ha='right')

        filename = f'keypoint_plot_{self.metric_name.lower().replace(" ", "_")}'
        
        return (fig, filename)
            


            
