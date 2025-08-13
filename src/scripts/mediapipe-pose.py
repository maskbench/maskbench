import cv2
import csv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Pose Landmarker on a video file."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        help="The path to the input video file relative to the dataset directory configured in the .env file.",
    )
    parser.add_argument(
        "--model_path", type=str, help="The path to the pose landmarker model file."
    )
    args = parser.parse_args()

    video_path = os.path.join("/datasets", args.video_path)
    model_path = os.path.join("/weights", args.model_path)
    out_dir = "/output"
    csv_dir = "/mediapipe_csv"
    video_file_name = os.path.splitext(os.path.basename(args.video_path))[0]
    video_output_path = os.path.join(out_dir, f"{video_file_name}_output.mp4")
    csv_path = os.path.join(csv_dir, f"{video_file_name}_landmarks.csv")

    print(f"video_path: {video_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_segmentation_masks=False,
    )

    landmarker = PoseLandmarker.create_from_options(options)

    csv_data = []
    csv_data.append(["frame", "landmark", "x", "y", "z", "visibility"])

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, int(frame_number * 1000 / fps))

        if result.pose_landmarks:
            for person_landmarks in result.pose_landmarks:

                for landmark_idx, landmark in enumerate(person_landmarks):
                    x_px = int(landmark.x * width)
                    y_px = int(landmark.y * height)
                    cv2.circle(frame, (x_px, y_px), 4, (0, 255, 0), -1)

                    csv_data.append(
                        [
                            frame_number,
                            f"Landmark_{landmark_idx}",
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            landmark.visibility,
                        ]
                    )

        out.write(frame)

        frame_number += 1
        print(f"Processed frame {frame_number}/{total_frames}", end="\r")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"\nDone! Output saved to:\n- Video: {video_output_path}\n- CSV: {csv_path}")


if __name__ == "__main__":
    main()
