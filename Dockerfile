FROM python:3.12

RUN apt-get update && apt-get install -y libgl1
RUN pip install --upgrade pip && pip install poetry

# set the current working directory inside the container
WORKDIR /

# Copy dependency files
COPY pyproject.toml poetry.lock* ./
# Avoid creating a virtualenv in a container
RUN poetry config virtualenvs.create false \
    && poetry install --no-root


WORKDIR /src
COPY src/ /src/

# Default command when the container starts
CMD ["python3", "main.py"]

