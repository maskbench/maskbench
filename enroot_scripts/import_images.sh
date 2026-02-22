#!/bin/bash
set -e

echo "=== Importing images to Enroot ==="

# Source environment variables
env_file_path="../.env"

if [ -f "$env_file_path" ]; then
        set -a
        source "$env_file_path"
        set +a
fi

echo "MASKANYONE VERSION: ${MASK_ANYONE_VERSION}"

import_image() {
    local name=$1
    local image=$2
    local sqsh_file="${name}.sqsh"

    if [ -f "$sqsh_file" ]; then
        echo "Skipping ${name}, already exists (${sqsh_file})"
        echo "Delete file to redownload it"
    else
        echo "Importing ${name}..."
        enroot import docker://${image}
    fi
}

import_image "shaddahmed14+sam2+test" "ghcr.io/shaddahmed14/sam2:test"
import_image "shaddahmed14+maskanyone_api+test" "ghcr.io/shaddahmed14/maskanyone_api:test"
import_image "shaddahmed14+openpose+test" "ghcr.io/shaddahmed14/openpose:test"
import_image "shaddahmed14+maskbench_runner+test" "ghcr.io/shaddahmed14/maskbench_runner:test"

echo "Done."