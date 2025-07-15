from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from evaluation.metrics.metric_result import KEYPOINT_AXIS, PERSON_AXIS, MetricResult
from .plot import Plot


class KeypointPlot(Plot):
    """Plot class for visualizing keypoint metrics for different models. Assumes that all metrics have the same number of keypoints."""
    
    def __init__(self, metric_names: List[str]):
        super().__init__(
            name=f"CocoKeypointPlot",
            config={
                'title': f'Coco Keypoint Plot',
                'xlabel': 'Keypoint Index',
                'ylabel': 'Average Value',
                'style': 'white',
                'figsize': (12, 6),
            }
        )
        self.metric_names = metric_names
    
    def draw(
        self,
        results: Dict[str, Dict[str, Dict[str, MetricResult]]],
    ) -> List[Tuple[plt.Figure, str]]:
        """
        Draw keypoint plot for the given metrics. Assumes that all metrics have the same number of keypoints.
        
        Args:
            results: Dictionary mapping:
                    model_name -> video_name -> MetricResult
                    
        Returns:
            List[Tuple[plt.Figure, str]]: List of tuples containing the figure and suggested filename.
        """
        figures_and_names = []
        
        for metric in self.metric_names:
            if metric not in results:
                raise ValueError(f"Metric {metric} not found in results")
            
            metric_results = results[metric]
            plot_data = []
            
            for model_name, video_results in metric_results.items():
                model_values = []  # List to store values for current model
                for video_name, metric_result in video_results.items():
                    avg_video_keypoint_values = metric_result.aggregate([PERSON_AXIS, KEYPOINT_AXIS], method='mean').values
                    model_values.append(avg_video_keypoint_values)
                
                # Stack all values for this model into a single array
                avg_model_keypoint_values = np.mean(np.stack(model_values, axis=0), axis=0)
                
                # Add data points for this model directly to plot_data
                for keypoint_idx, value in enumerate(avg_model_keypoint_values[:17]):
                    plot_data.append({
                        'Model': model_name,
                        'Keypoint Index': keypoint_idx,
                        'Average Value': value
                    })
            
            # Create DataFrame from collected data
            df = pd.DataFrame(plot_data)
            
            # Set up figure with standard styling
            self.config['title'] = f'{metric} by Keypoint and Model'
            fig = self._setup_figure()
            
            # Create the plot
            sns.barplot(
                data=df,
                x='Keypoint Index',
                y='Average Value',
                hue='Model',
                palette='rocket'
            )
            
            figures_and_names.append((fig, f'keypoint_plot_{metric}'))
            
        return figures_and_names
            


            
