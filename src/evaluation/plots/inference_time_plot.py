from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .plot import Plot


class InferenceTimePlot(Plot):
    """Plot class for visualizing inference times for different models."""
    
    def __init__(self):
        super().__init__(
            name="InferenceTimePlot",
            config={
                'title': 'Average Inference Time by Model',
                'xlabel': 'Model',
                'ylabel': 'Inference Time (seconds)',
                'figsize': (12, 6),
            }
        )
    
    def draw(
        self,
        inference_times: Dict[str, Dict[str, float]],
    ) -> Tuple[plt.Figure, str]:
        """
        Draw inference time plot for each model.
        
        Args:
            inference_times: Dictionary mapping:
                    model_name -> video_name -> inference_time
                    
        Returns:
            Tuple[plt.Figure, str]: The figure and suggested filename.
        """
        plot_data = []
        
        for model_name, video_times in inference_times.items():
            avg_time = np.mean(list(video_times.values()))
            plot_data.append({
                'Model': model_name,
                'Time': avg_time
            })
        
        df = pd.DataFrame(plot_data)
        
        fig = self._setup_figure()
        
        # Create the bar plot
        ax = sns.barplot(
            data=df,
            x='Model',
            y='Time',
            hue='Model',
        )

        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of each bar
        for i, bar in enumerate(ax.patches):
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                value,
                f'{value:.2f}',
                ha='center',
                va='bottom'
            )
        
        return (fig, 'inference_time_plot')
