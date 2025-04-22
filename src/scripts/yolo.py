import argparse
import os
from ultralytics import YOLOE, settings
import cv2


def main():
    parser = argparse.ArgumentParser(description="Run YOLOE on a video file.")
    parser.add_argument("--video_path", type=str, help="The path to the input video file relative to the dataset directory configured in the .env file.")
    args = parser.parse_args()

    # These are locations within the docker container (mounted directories)
    video_path = os.path.join("/datasets", args.video_path)
    video_file_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = "/output"

    settings.update({"weights_dir": "/weights"})
    model = YOLOE("yoloe-11l-seg-pf.pt")

    # Predict with person class and confidence threshold
    results = model.track(video_path, conf=0.85, classes=[2163], stream=True)  # Class 2163 corresponds to 'person' (see model.names)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    writer = None
    for result in results:
        frame = result.plot()

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_path = os.path.join(out_dir, f"tracked_{video_file_name}.mp4")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        writer.write(frame)

    if writer:
        writer.release()
        print("Video saved to", output_path)

if __name__ == "__main__":
    main()
