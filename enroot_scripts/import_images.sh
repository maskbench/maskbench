#!/bin/bash
set -e

echo "=== Importing images to Enroot ==="

# Source environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Import service images
echo "Importing sam2..."
enroot import docker://ghcr.io/maskanyone/maskanyone/sam2:${MASK_ANYONE_VERSION}

echo "Importing maskanyone_api..."
enroot import docker://ghcr.io/maskanyone/maskanyone/api:${MASK_ANYONE_VERSION}

echo "Importing openpose..."
enroot import docker://ghcr.io/maskanyone/maskanyone/openpose:${MASK_ANYONE_VERSION}

echo "Importing runner..."
enroot import docker://ghcr.io/shaddahmed19/maskbench_runner:latest

echo "Images are stored in: ~/.local/share/enroot/"
