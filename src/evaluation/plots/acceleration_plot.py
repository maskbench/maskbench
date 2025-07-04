from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.metrics.metric_result import MetricResult, FRAME_AXIS, PERSON_AXIS, KEYPOINT_AXIS
from .plot import Plot


class AccelerationOverTimePlot(Plot):
    """Plot class for visualizing acceleration over time for different models."""
    
    def __init__(self):
        super().__init__(
            name="AccelerationOverTime",
            config={
                'title': 'Acceleration Over Time',
                'xlabel': 'Frames',
                'ylabel': 'Acceleration (pixels/second²)',
                'style': 'whitegrid'
            }
        )
    
    def draw(
        self,
        results: Dict[str, Dict[str, Dict[str, MetricResult]]],
    ) -> List[Tuple[plt.Figure, str]]:
        """
        Draw acceleration over time plot for each model.
        
        Args:
            results: Dictionary mapping:
                    metric_name -> model_name -> video_name -> MetricResult
                    
        Returns:
            List[Tuple[plt.Figure, str]]: List of tuples containing the figure and suggested filename
        """
        if 'Acceleration' not in results:
            raise ValueError("No acceleration metric found in results")
            
        acceleration_results = results['Acceleration']
        video_groupings = self._group_by_video(acceleration_results)
        
        figures_and_names = []
        for video_name, model_results in video_groupings.items():
            fig = self._setup_figure()
            
            plt.title(f"{self.config['title']} - {video_name}", pad=20)
            
            for model_name, metric_result in model_results.items():
                frame_accel = metric_result.aggregate([PERSON_AXIS, KEYPOINT_AXIS], method='mean')
                frames = np.arange(len(frame_accel.values))
                sns.lineplot(x=frames, y=frame_accel.values, label=model_name, linewidth=2)
            
            if self.config.get('legend', True):
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
            
            filename = f"acceleration_seaborn2_{video_name}.png"
            figures_and_names.append((fig, filename))
            
        return figures_and_names 