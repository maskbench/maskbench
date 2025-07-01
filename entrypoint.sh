#!/bin/bash

# Initialize trap handlers
cleanup() {
  echo "Fixing permissions..."
  chmod -R a+rw /output
}

handle_sigterm() {
  kill -TERM "$child" 2>/dev/null
  wait "$child"
  cleanup
  exit 0
}

handle_sigint() {
  kill -INT "$child" 2>/dev/null
  wait "$child"
  cleanup
  exit 0
}

# Set up signal handlers
trap handle_sigterm SIGTERM
trap handle_sigint SIGINT
trap cleanup EXIT

# Start the Python process in the background and get its PID
python3 main.py &
child=$!

# Wait for the Python process to complete
wait "$child"