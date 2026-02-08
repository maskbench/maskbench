#!/bin/bash

cleanup() {
  echo "Fixing permissions..."
  chmod -R a+rw /output
  echo "Cleanup complete."
}

trap cleanup EXIT

python3 -u main.py
exit_code=$?

echo "Process completed with exit code $exit_code"
exit $exit_code
