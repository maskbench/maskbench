#!/bin/bash

set -e

# Source environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Directory to store PIDs
PID_DIR=./enroot_pids
mkdir -p $PID_DIR

# Function to start a service in background
start_service() {
    local name=$1
    local image=$2
    shift 2
    local extra_args="$@"
    
    echo "Starting ${name}..."
    
    # Start container in background
    # Note: Adjust the image filename based on what enroot actually creates
    nohup enroot start \
        --rw \
        ${extra_args} \
        ${image} \
        > ${PID_DIR}/${name}.log 2>&1 &
    
    echo $! > ${PID_DIR}/${name}.pid
    echo "${name} started (PID: $!)"
}

# Clean up old PIDs
rm -f ${PID_DIR}/*.pid

# Start SAM2
start_service "sam2" \
    "ghcr.io+maskanyone+maskanyone+sam2-${MASK_ANYONE_VERSION}.sqsh" \
    "--env NVIDIA_VISIBLE_DEVICES=${MASKBENCH_GPU_ID_3}" \
    "--env SAM2_OFFLOAD_VIDEO_TO_CPU=false" \
    "--env SAM2_OFFLOAD_STATE_TO_CPU=false"

echo "Waiting 90 seconds for sam2 to initialize..."
sleep 90

# Start MaskAnyone API
start_service "maskanyone_api" \
    "ghcr.io+maskanyone+maskanyone+api-${MASK_ANYONE_VERSION}.sqsh" \
    "--env NVIDIA_VISIBLE_DEVICES=${MASKBENCH_GPU_ID_3}"

echo "Waiting 90 seconds for maskanyone_api to initialize..."
sleep 90

# Start OpenPose
start_service "openpose" \
    "ghcr.io+maskanyone+maskanyone+openpose-${MASK_ANYONE_VERSION}.sqsh" \
    "--env NVIDIA_VISIBLE_DEVICES=${MASKBENCH_GPU_ID_2}" \
    "--env OPENPOSE_MODEL_DIR=/workspace/openpose/models"

echo "Waiting 90 seconds for openpose to initialize..."
sleep 90

echo ""
echo "=== All services started ==="
echo "Service logs are in: ${PID_DIR}/"
