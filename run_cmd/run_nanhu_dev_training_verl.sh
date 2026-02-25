#!/bin/bash
set -e

cd /mnt/seed17/001688/shenyichong/verl/run_cmd

echo "=== Step 1: Starting Ray cluster ==="
bash start_ray_cluster.sh
RAY_EXIT=$?

if [ $RAY_EXIT -ne 0 ]; then
    echo "ERROR: start_ray_cluster.sh failed with exit code $RAY_EXIT"
    exit 1
fi

echo ""
echo "=== Step 2: Waiting for Ray to be fully ready ==="
sleep 5

# Verify Ray is running
if ! ray status > /dev/null 2>&1; then
    echo "ERROR: Ray cluster is not running"
    exit 1
fi
echo "Ray cluster is ready"

echo ""
echo "=== Step 3: Starting training job ==="
bash qwen3-8b_grpo_nodes.sh