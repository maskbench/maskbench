#!/bin/bash

set -e

env_file_path="../.env"

if [ -f "$env_file_path" ]; then
        set -a
        source "$env_file_path
        set +a
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
        --env-file ${env_file_path} \
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
    "./maskanyone+maskanyone+sam2+${MASK_ANYONE_VERSION}.sqsh" \
    "--env NVIDIA_VISIBLE_DEVICES=${MASKBENCH_GPU_ID_3}"

echo "Waiting 90 seconds for sam2 to initialize..."
sleep 90

# Start OpenPose
start_service "openpose" \
    "./maskanyone+maskanyone+openpose+${MASK_ANYONE_VERSION}.sqsh" \
    "--env NVIDIA_VISIBLE_DEVICES=${MASKBENCH_GPU_ID_2}"

echo "Waiting 90 seconds for openpose to initialize..."
sleep 90

# Start MaskAnyone API
start_service "maskanyone_api" \
    "./maskanyone+maskanyone+api+${MASK_ANYONE_VERSION}.sqsh" \
    "--env NVIDIA_VISIBLE_DEVICES=${MASKBENCH_GPU_ID_3}"

echo ""
echo "=== All services started ==="
echo "Service logs are in: ${PID_DIR}/"