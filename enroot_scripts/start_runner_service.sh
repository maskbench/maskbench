#!/bin/bash
# Start the runner container interactively

set -e

env_file_path="../.env"

if [ -f "$env_file_path" ]; then
        set -a
        source "$env_file_path"
        set +a
fi

echo "=== Starting runner container ==="
echo ""

# Check if services are running
PID_DIR=./enroot_pids
all_running=true

for service in sam2 maskanyone_api openpose; do
    if [ -f ${PID_DIR}/${service}.pid ]; then
        pid=$(cat ${PID_DIR}/${service}.pid)
        if ! ps -p $pid > /dev/null 2>&1; then
            echo "WARNING: ${service} is not running!"
            all_running=false
        fi
    else
        echo "WARNING: ${service} was never started!"
        all_running=false
    fi
done

if [ "$all_running" = false ]; then
    echo ""
    echo "Some services are not running. Start them with: start_services.sh"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start runner interactively
# Note: Adjust the image filename based on what enroot creates
enroot start \
    --env NVIDIA_VISIBLE_DEVICES=${MASKBENCH_GPU_ID_1} \
    --env SAM2_PORT=${SAM2_PORT} \
    --env SAM2_HOST=${SAM2_HOST} \
    --env OPENPOSE_PORT=${OPENPOSE_PORT} \
    --env OPENPOSE_HOST=${OPENPOSE_HOST} \
    --env WORKER_PORT=${WORKER_PORT} \
    --env WORKER_HOST=${WORKER_HOST} \
    --env MASKBENCH_CONFIG_FILE=${MASKBENCH_CONFIG_FILE} \
    --mount ../src:/src \
    --mount ../poetry.lock:/poetry.lock \
    --mount ../pyproject.toml:/pyproject.toml \
    --mount ../config:/config \
    --mount ${MASKBENCH_DATASET_DIR}:/datasets \
    --mount ${MASKBENCH_OUTPUT_DIR}:/output \
    --rw \
    shaddahmed14+maskbench_runner+test.sqsh

# When you exit the runner, this script ends
echo ""
echo "Runner exited."