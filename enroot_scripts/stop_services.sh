#!/bin/bash
# Stop all running services

PID_DIR=./enroot_pids

echo "=== Stopping services ==="

for service in sam2 maskanyone_api openpose; do
    if [ -f ${PID_DIR}/${service}.pid ]; then
        pid=$(cat ${PID_DIR}/${service}.pid)
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping ${service} (PID: ${pid})..."
            kill $pid
            sleep 2
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo "  Force killing ${service}..."
                kill -9 $pid
            fi
        else
            echo "${service} not running (stale PID)"
        fi
        rm -f ${PID_DIR}/${service}.pid
    fi
done

echo ""
echo "All services stopped."
echo "Logs preserved in: ${PID_DIR}/"