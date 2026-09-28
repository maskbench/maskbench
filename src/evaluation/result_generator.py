from checkpointer import Checkpointer
from typing import List, Dict

class ResultGenerator:
    def __init__(self, checkpointer: Checkpointer, test:bool = False):
        self.checkpointer = checkpointer
        self.metric_results_meta = self.get_evaluation_result_meta(checkpointer, test)

    def get_evaluation_result_meta(self, checkpointer:Checkpointer, test:bool)-> Dict[str, Dict[str, List[str]]]:
        metric_results_meta = {}
        evaluation_dir = checkpointer.evaluation_dir
        evaluated_metric_names = [d.name for d in evaluation_dir.iterdir() if d.is_dir()]

        for metric_name in evaluated_metric_names:
            metric_dir = evaluation_dir / metric_name
            evaluated_model_names = [d.name for d in metric_dir.iterdir() if d.is_dir()]

            for model_name in evaluated_model_names:
                model_dir = metric_dir / model_name
                evaluated_video_result_paths = list(video_path.stem for video_path in model_dir.glob('*.json'))
                print(f'Found {len(evaluated_video_result_paths)} for Estimator {model_name} for Metric {metric_name}')

                if test: # test with 5 videos per model per metric
                    evaluated_video_result_paths = evaluated_video_result_paths[:5]
                    print('Testing with 5 videos')

                for video_name in evaluated_video_result_paths:
                    if checkpointer.exists_evaluation_result(metric_name, model_name, video_name):
                        metric_results_meta.setdefault(metric_name, {}).setdefault(model_name, []).append(video_name)
                        
        return metric_results_meta
        
    def return_metric_names(self):
        return list(self.metric_results_meta.keys())
    def return_model_names_for_metric(self, metric_name:str):
        return list(self.metric_results_meta.get(metric_name, {}).keys())

    def iter_with_metric_model(self, metric_name: str, model_name: str):
        video_names = self.metric_results_meta.get(metric_name, {}).get(model_name, [])
        for video_name in video_names:
            result = self.checkpointer.load_evaluation_result(metric_name, model_name, video_name)
            yield video_name, result

    def iter_with_metric(self, metric_name: str):
        model_names = self.return_model_names_for_metric(metric_name)
        for model_name in model_names:
            yield model_name, self.iter_with_metric_model(metric_name, model_name)