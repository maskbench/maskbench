from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from evaluation.metrics.metric_result import KEYPOINT_AXIS, PERSON_AXIS, MetricResult
from keypoint_pairs import COCO_KEYPOINT_NAMES
from .plot import Plot


class CocoKeypointPlot(Plot):
    """Plot class for visualizing COCO keypoint metrics for different models. Assumes that all metrics have the same number of keypoints."""
    
    def __init__(self, metric_names: List[str]):
        super().__init__(
            name=f"CocoKeypointPlot",
            config={
                'title': f'Coco Keypoint Plot',
                'xlabel': 'Keypoint',
                'ylabel': 'Average Value',
                'style': 'white',
                'figsize': (18, 6),
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
                    metric_name -> model_name -> video_name -> MetricResult
                    
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
                model_values = []
                for video_name, metric_result in video_results.items():
                    avg_video_keypoint_values = metric_result.aggregate([PERSON_AXIS, KEYPOINT_AXIS], method='mean').values
                    model_values.append(avg_video_keypoint_values)
                avg_model_keypoint_values = np.mean(np.stack(model_values, axis=0), axis=0)
                
                for keypoint_idx, value in enumerate(avg_model_keypoint_values):
                    if keypoint_idx not in COCO_KEYPOINT_NAMES.keys():
                        continue

                    plot_data.append({
                        'Model': model_name,
                        'Keypoint': f"{COCO_KEYPOINT_NAMES[keypoint_idx]} ({keypoint_idx})",
                        'Average Value': value
                    })
            
            df = pd.DataFrame(plot_data)
            
            self.config['title'] = f'{metric} by Keypoint and Model'
            self.config['style'] = 'grid'
            fig = self._setup_figure()
            
            sns.barplot(
                data=df,
                x='Keypoint',
                y='Average Value',
                hue='Model',
            )

            plt.xticks(rotation=45, ha='right')
            
            figures_and_names.append((fig, f'keypoint_plot_{metric}'))
            
        return figures_and_names
            


            
