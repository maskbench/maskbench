import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from evaluation.metrics.metric_result import MetricResult


class Plot(ABC):
    """Base class for all plots in MaskBench."""
    
    def __init__(self, name: str, config: Optional[Dict[str, any]] = None):
        """
        Initialize a plot.
        
        Args:
            name: Unique name of the plot
            config: Optional configuration dictionary for the plot
                   Common config options:
                   - style: str for seaborn style (default: white)
                   - figsize: tuple for figure size (default: (10, 5))
                   - dpi: int for figure resolution (default: 300)
                   - title: str for plot title
                   - xlabel: str for x-axis label
                   - ylabel: str for y-axis label
                   - legend: bool for showing legend
        """
        self.name = name
        self.output_path = "/output/plots"
        os.makedirs(self.output_path, exist_ok=True)
        
        self.config = config or {}
        if 'figsize' not in self.config:
            self.config['figsize'] = (10, 5)
        if 'dpi' not in self.config:
            self.config['dpi'] = 300
        if 'style' not in self.config:
            self.config['style'] = 'white'
        
        sns.set_style(self.config['style'])
        sns.color_palette("rocket")
        sns.set_context("paper")
        
    @abstractmethod
    def draw(
        self,
        results: Dict[str, Dict[str, Dict[str, MetricResult]]],
    ) -> None:
        """
        Draw the plot using the provided results.
        
        Args:
            results: Dictionary mapping:
                    metric_name -> model_name -> video_name -> MetricResult
            
        The plot is saved to the output directory.
        """
        pass
    
    def _setup_figure(self) -> None:
        """Set up the figure with standard configuration."""
        plt.figure(figsize=self.config['figsize'], dpi=self.config['dpi'])
        plt.tight_layout()

        # Remove plot edges
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        
        if 'title' in self.config:
            plt.title(self.config['title'])
        if 'xlabel' in self.config:
            plt.xlabel(self.config['xlabel'], labelpad=10)
        if 'ylabel' in self.config:
            plt.ylabel(self.config['ylabel'], labelpad=10)
    
    def _save_plot(self, filename: str) -> None:
        """Save the plot to the output directory with the given filename."""
        output_file = os.path.join(self.output_path, filename)
        plt.savefig(output_file, bbox_inches='tight', dpi=self.config['dpi'])
        plt.close() 

    def _group_by_video(self, results: Dict[str, Dict[str, Dict[str, MetricResult]]]) -> Dict[str, Dict[str, Dict[str, MetricResult]]]:
        """Group the results by video."""
        video_to_models = {}
        for model_name, video_results in results.items():
            for video_name, metric_result in video_results.items():
                if video_name not in video_to_models:
                    video_to_models[video_name] = {}
                video_to_models[video_name][model_name] = metric_result
        return video_to_models