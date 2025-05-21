FROM python:3.12

RUN apt-get update && apt-get install -y libgl1
RUN pip install --upgrade pip && pip install poetry

# set the current working directory inside the container
WORKDIR /

# download model weights
RUN mkdir -p /weights
RUN curl -L -o /weights/mediapipe_pose.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
RUN curl -L -o /weights/yolo11n-pose.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"

# Copy dependency files
COPY pyproject.toml poetry.lock* ./
# Avoid creating a virtualenv in a container
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

WORKDIR /src
COPY src/ /src/

# Default command when the container starts
CMD ["python3", "main.py"]

