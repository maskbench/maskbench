#!/bin/bash
# Check status of running services

PID_DIR=./enroot_pids

echo "=== Service Status ==="
echo ""

for service in sam2 maskanyone_api openpose maskbench-runner; do
    if [ -f ${PID_DIR}/${service}.pid ]; then
        pid=$(cat ${PID_DIR}/${service}.pid)
        if ps -p $pid > /dev/null 2>&1; then
            echo "✓ ${service} is RUNNING (PID: ${pid})"
        else
            echo "✗ ${service} is STOPPED (stale PID: ${pid})"
        fi
        
        # Show last few log lines
        if [ -f ${PID_DIR}/${service}.log ]; then
            echo "  Last log lines:"
            tail -n 3 ${PID_DIR}/${service}.log | sed 's/^/    /'
        fi
        echo ""
    else
        echo "✗ ${service} - no PID file found"
        echo ""
    fi
done

echo "To view full logs: tail -f ${PID_DIR}/<service>.log"