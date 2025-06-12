import argparse
import os
import tempfile
import pickle
import requests
import cv2
import json


def draw_pose(frame, keypoints, confidence_threshold=0.1):
    # keypoints: list of (x, y, confidence) tuples
    # Example skeleton connections (OpenPose BODY_25 format)
    skeleton = [
        (1, 8),
        (1, 2),
        (1, 5),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (8, 9),
        (9, 10),
        (10, 11),
        (8, 12),
        (12, 13),
        (13, 14),
        (1, 0),
        (0, 15),
        (15, 17),
        (0, 16),
        (16, 18),
        (14, 19),
        (19, 20),
        (14, 21),
        (11, 22),
        (22, 23),
        (11, 24),
        # (2, 17), (5, 18)
    ]

    for pair in skeleton:
        part_from, part_to = pair
        if (
            keypoints[part_from][2] > confidence_threshold
            and keypoints[part_to][2] > confidence_threshold
        ):
            x1, y1 = int(keypoints[part_from][0]), int(keypoints[part_from][1])
            x2, y2 = int(keypoints[part_to][0]), int(keypoints[part_to][1])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for x, y, conf in keypoints:
        if conf > confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)


def main():
    parser = argparse.ArgumentParser(description="Run OpenPose on a video file.")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file relative to /datasets.",
    )
    args = parser.parse_args()

    video_path = os.path.join("/datasets", args.video_path)
    video_file_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = "/output"

    openpose_url = "http://openpose:8000/openpose/estimate-pose-on-video"

    data = {"options": {"model_pose": "BODY_25", "face": False, "hand": False}}

    print("Uploading video to OpenPose container and starting pose estimation...")
    # Upload video to OpenPose container
    with open(video_path, "rb") as f:
        files = {"video": ("input.mp4", f, "video/mp4")}
        options = {"model_pose": "BODY_25", "face": False, "hand": False}
        data = {"options": json.dumps(options)}
        response = requests.post(openpose_url, files=files, data=data)

    if response.status_code != 200:
        raise Exception(
            f"OpenPose container error: {response.status_code} {response.text}"
        )

    # Load pose data
    pose_data = pickle.loads(
        response.content
    )  # Expecting {frame_idx: list of keypoints}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(out_dir, f"openpose_{video_file_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(pose_data):
            keypoints_list = pose_data[frame_idx]
        else:
            keypoints_list = []

        draw_pose(frame, keypoints_list["pose_keypoints"])

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print("Video saved to", output_path)


if __name__ == "__main__":
    main()
