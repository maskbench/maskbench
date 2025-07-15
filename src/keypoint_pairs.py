from mediapipe.python.solutions.pose import PoseLandmark

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

MEDIAPIPE_KEYPOINT_PAIRS = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 7),
            (0, 4),
            (4, 5),
            (5, 6),
            (6, 8),
            (9, 10),
            (11, 12),
            (11, 13),
            (13, 15),
            (15, 19),
            (15, 17),
            (17, 19),
            (15, 21),
            (12, 14),
            (14, 16),
            (16, 20),
            (16, 18),
            (18, 20),
            (11, 23),
            (12, 24),
            (16, 22),
            (23, 25),
            (24, 26),
            (25, 27),
            (26, 28),
            (23, 24),
            (28, 30),
            (28, 32),
            (30, 32),
            (27, 29),
            (27, 31),
            (29, 31),
        ]

COCO_TO_OPENPOSE = [
    0,   # Nose
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

OPENPOSE_KEYPOINT_PAIRS = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16),
            (15, 19),
            (19, 20),
            (15, 21),
            (16, 22),
            (22, 23),
            (16, 24),
            (5, 17),
            (6, 17),
            (11, 12),
            (17, 18),
            (5, 6),
        ]

# Yolo and Coco have the same keypoint pairs since Yolo stores keypoints in Coco format
COCO_KEYPOINT_PAIRS = [
            (15, 13),
            (16, 14),
            (13, 11),
            (12, 14),
            (11, 12),
            (11, 5),
            (12, 6),
            (5, 6),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
        ]