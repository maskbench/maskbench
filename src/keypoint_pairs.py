from mediapipe.python.solutions.pose import PoseLandmark

# Depending on the source you look at, there are a different number of keypoints for COCO.
# We aggreed to follow the YOLO standard and ordering: https://docs.ultralytics.com/tasks/pose/
# This does not have a neck keypoint, which is the interpolation of the two shoulders and is mainly used for drawing.
# The order is different from other COCO sources, see also issue: https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1560
COCO_KEYPOINT_NAMES = {
    0: "Nose",
    1: "L. Eye",
    2: "R. Eye",
    3: "L. Ear",
    4: "R. Ear",
    5: "L. Shoulder",
    6: "R. Shoulder",
    7: "L. Elbow",
    8: "R. Elbow",
    9: "L. Wrist",
    10: "R. Wrist",
    11: "L. Hip",
    12: "R. Hip",
    13: "L. Knee",
    14: "R. Knee",
    15: "L. Ankle",
    16: "R. Ankle",
}

# stores the indices of keypoints required in coco format
COCO_TO_MEDIAPIPE = [
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_EYE,
    PoseLandmark.RIGHT_EYE,
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE
]


COCO_TO_OPENPOSE_BODY25B = [
    0,  # Nose
    1,  # Left Eye
    2,  # Right Eye
    3,  # Left Ear
    4,  # Right Ear
    5,  # Left Shoulder
    6,  # Right Shoulder
    7,  # Left Elbow
    8,  # Right Elbow
    9,  # Left Wrist
    10, # Right Wrist
    11, # Left Hip
    12, # Right Hip
    13, # Left Knee
    14, # Right Knee
    15, # Left Ankle
    16, # Right Ankle
]

COCO_TO_OPENPOSE_BODY25 = [
    0,  # Nose
    16, # Left Eye
    15, # Right Eye
    18, # Left Ear
    17, # Right Ear
    5,  # Left Shoulder
    2,  # Right Shoulder
    6,  # Left Elbow
    3,  # Right Elbow
    7,  # Left Wrist
    4,  # Right Wrist
    12, # Left Hip
    9,  # Right Hip
    13, # Left Knee
    10, # Right Knee
    14, # Left Ankle
    11, # Right Ankle
]

MEDIAPIPE_KEYPOINT_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 19), (15, 17), (17, 19), (15, 21),
    (12, 14), (14, 16), (16, 20), (16, 18), (18, 20), (11, 23), (12, 24), (16, 22),
    (23, 25), (24, 26), (25, 27), (26, 28), (23, 24), (28, 30), (28, 32), (30, 32),
    (27, 29), (27, 31), (29, 31)
]

OPENPOSE_BODY25_KEYPOINT_PAIRS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9),
    (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (1, 0), (0, 15),
    (15, 17), (0, 16), (16, 18), (14, 19), (19, 20),
    (14, 21), (11, 22), (22, 23), (11, 24)
    # (2, 17), (5, 18)
]

OPENPOSE_BODY25B_KEYPOINT_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16), (15, 19), (19, 20),
    (15, 21), (16, 22), (22, 23), (16, 24), (5, 17), (6, 17), (11, 12), (17, 18),
    (5, 6),
]

# Yolo and Coco have the same keypoint pairs since Yolo stores keypoints in Coco format
COCO_KEYPOINT_PAIRS = [
    (15, 13), (16, 14), (13, 11), (12, 14), (11, 12), (11, 5),
    (12, 6), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (0, 1), (0, 2), (1, 3), (2, 4)
]
