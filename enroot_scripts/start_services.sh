#!/bin/bash

set -e

env_file_path="../.env"

if [ -f "$env_file_path" ]; then
        set -a
        source "$env_file_path"
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
        if [ ! -f "$image" ]; then
        echo "ERROR: IMAGE file $image not found!"
        exit 1
        fi

    echo "Starting ${name}..."
    # Start container in background
    # Note: Adjust the image filename based on what enroot actually creates
    nohup enroot start \
        --rw \
        ${extra_args} \
        ${image} \
        > ${PID_DIR}/${name}.log 2>&1 &

    echo $! > ${PID_DIR}/${name}.pid
}

# Clean up old PIDs
rm -f ${PID_DIR}/*.pid

# Start SAM2
 start_service "sam2" \
    "./shaddahmed14+sam2+test.sqsh" \
    "--env NVIDIA_VISIBLE_DEVICES=${MASKBENCH_GPU_ID_3}" \
        "--env SAM2_OFFLOAD_STATE_TO_CPU=${SAM2_OFFLOAD_STATE_TO_CPU}" \
        "--env SAM2_OFFLOAD_VIDEO_TO_CPU=${SAM2_OFFLOAD_VIDEO_TO_CPU}" \
        "--env SAM2_PORT=${SAM2_PORT}" \
        "--env SAM2_HOST=${SAM2_HOST}"
echo "${SAM2_PORT} ${SAM2_HOST}"
echo "Waiting 60 seconds for sam2 to initialize..."
echo "$SAM2_PORT"
#sleep 60

# Start OpenPose
start_service "openpose" \
    "./shaddahmed14+openpose+test.sqsh" \
    "--env NVIDIA_VISIBLE_DEVICES=${MASKBENCH_GPU_ID_2}" \
        "--env OPENPOSE_MODEL_DIR=${OPENPOSE_MODEL_DIR}" \
        "--env OPENPOSE_PORT=${OPENPOSE_PORT}" \
        "--env OPENPOSE_HOST=${OPENPOSE_HOST}"
echo "${OPENPOSE_PORT} ${OPENPOSE_HOST}"
echo "$OPENPOSE_PORT $OPENPOSE_HOST"
echo "Waiting 60 seconds for openpose to initialize..."
sleep 60

# Start MaskAnyone API
 start_service "maskanyone_api" \
    "./shaddahmed14+maskanyone_api+test.sqsh" \
    "--env NVIDIA_VISIBLE_DEVICES=${MASKBENCH_GPU_ID_3}" \
        "--env SAM2_PORT=${SAM2_PORT}" \
        "--env SAM2_HOST=${SAM2_HOST}" \
        "--env OPENPOSE_PORT=${OPENPOSE_PORT}" \
        "--env OPENPOSE_HOST=${OPENPOSE_HOST}" \
        "--env WORKER_PORT=${WORKER_PORT}" \
        "--env WORKER_HOST=${WORKER_HOST}"
echo "${WORKER_PORT} ${WORKER_HOST}"