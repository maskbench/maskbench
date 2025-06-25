FROM python:3.12

RUN apt-get update && apt-get install -y libgl1
RUN pip install --upgrade pip && pip install poetry

# set the current working directory inside the container
WORKDIR /

# download model weights
RUN mkdir -p /weights/pre_built

# Mediapipe
RUN curl -L -o /weights/pre_built/pose_landmarker_lite.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
RUN curl -L -o /weights/pre_built/pose_landmarker_full.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
RUN curl -L -o /weights/pre_built/pose_landmarker_heavy.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

# Yolo
RUN curl -L -o /weights/pre_built/yolo11n-pose.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
RUN curl -L -o /weights/pre_built/yolo11s-pose.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt"
RUN curl -L -o /weights/pre_built/yolo11m-pose.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt"
RUN curl -L -o /weights/pre_built/yolo11l-pose.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt"
RUN curl -L -o /weights/pre_built/yolo11x-pose.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt"

# Copy dependency files
COPY pyproject.toml poetry.lock* ./
# Avoid creating a virtualenv in a container
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

WORKDIR /src
COPY src/ /src/

# Default command when the container starts
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

