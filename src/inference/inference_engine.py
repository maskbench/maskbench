import time
from checkpointer import Checkpointer
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import tqdm

class InferenceEngine:
    """Class responsible for running the pose estimators on the videos and saving the results in the `poses` folder."""
    
    def __init__(self, dataset: dict, pose_estimators: list, checkpointer: Checkpointer, execute_processing: bool):
        self.dataset = dataset
        self.pose_estimators = pose_estimators
        self.checkpointer = checkpointer
        self.execute_processing = execute_processing
    
    def run_parallel_tasks(self, max_workers: int = None) -> None:
        if not self.execute_processing:
            logging.info("Skipping inference as per configuration.")
            print("Skipping inference as per configuration.")
            return
        num_estimator = len(self.pose_estimators)
        if num_estimator == 0:
            raise ValueError("No pose estimators provided. Please provide at least one pose estimator to run the inference engine.")
        
        if max_workers is None:
            max_workers = num_estimator
        max_workers= min(max_workers, num_estimator)  # Max is number of estimators to avoid model runtime conflicts. 

        print('=' * 50)
        print(f"Running {num_estimator} pose estimators with max_workers={max_workers}")
        print(f"Total videos to process: {len(self.dataset)}")
        print('=' * 50)
        logging.info(f"Processing {len(self.dataset)} videos with max_workers={max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_estimator = {
                executor.submit(self.estimate_pose_keypoints, estimator): estimator
                for estimator in self.pose_estimators
            }
            for future in as_completed(future_to_estimator):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Faced Exception: {e} while processing estimator: {future_to_estimator[future].name}")
                    print(f"Faced Exception: {e} while processing estimator: {future_to_estimator[future].name}")
        
        return

    def estimate_pose_keypoints(self, estimator) -> None:
        """
        Run the pose estimators on the videos and save the results in the `poses` folder.
        If a checkpoint name is provided in the configuration file, the inference engine will load the results from the checkpoint and skip the inference for the videos that already have results.
        This allows to resume the inference process in case it fails or to skip the inference entirely and only evaluate the metrics.

        Returns:
            None
        """
        progress = tqdm.tqdm(total=len(self.dataset), desc=f"Processing videos with {estimator.name}", unit="videos")
        print()
        for video in self.dataset:
            progress.update(1)
            print(progress.__str__())
            
            if self.checkpointer.exists(estimator.name, video.get_filename()):
                print(f"Skipping already processed video {video.get_filename()} for estimator {estimator.name}")
                continue # if results already exist, skip inference

            logging.info(f"Running estimator '{estimator.name}' on video {video.path}")

            start_time = time.time()
            try:
                video_pose_result = estimator.estimate_pose(video.path)
                self.checkpointer.save_video_pose_result(video_pose_result, estimator.name)
                self.checkpointer.save_inference_time(estimator.name, video.get_filename(), time.time() - start_time)
            except Exception as e:
                print(f"Error processing video {video.get_filename()} with estimator {estimator.name}: {e}")
                logging.error(f"Faced Exception: {e} on Video: {video.get_filename()} with Estimator: {estimator.name}")
                continue

        progress.close()
        print()
        
        logging.info(f"Completed estimator '{estimator.name}'")
        return