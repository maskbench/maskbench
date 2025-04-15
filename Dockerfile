FROM python:3.12

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
COPY datasets/ /datasets/
COPY output/ /output/

# Default command when the container starts
CMD ["python3", "main.py"]

