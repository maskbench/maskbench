FROM python:3.12

RUN apt-get update && apt-get install -y libgl1 ffmpeg
RUN pip install --upgrade pip && pip install poetry

# set the current working directory inside the container
WORKDIR /

# download model weights
RUN mkdir -p /weights/pre_built

# Mediapipe
RUN curl -L -o /weights/pre_built/pose_landmarker_lite.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
RUN curl -L -o /weights/pre_built/pose_landmarker_full.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
RUN curl -L -o /weights/pre_built/pose_landmarker_heavy.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

# Mediapipe Hand Landmark
RUN curl -L -o /weights/pre_built/mediapipe_hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

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

