#!/bin/bash
echo "RANK=$RANK, MASTER_ADDR=$MASTER_ADDR, WORLD_SIZE=$WORLD_SIZE"
echo $(date '+%Y-%m-%d %H:%M:%S')

WORK_DIR="/mnt/seed17/001688/shenyichong/verl"
cd $WORK_DIR/run_cmd

# Check RANK environment variable to determine role
if [ "$RANK" -eq 0 ]; then
    # === MASTER NODE (RANK=0) ===
    echo "$(date '+%Y-%m-%d-%H-%M-%S'), Starting Ray head node..."
    ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8872 --disable-usage-stats
    
    if [ "${WORLD_SIZE:-1}" -ge 2 ]; then
        echo "$(date '+%Y-%m-%d-%H-%M-%S'), Waiting 60s for ${WORLD_SIZE} worker nodes to connect..."
        sleep 60
        echo "$(date '+%Y-%m-%d-%H-%M-%S'), Wait time up."
    else
        sleep 5
    fi
    
    ray status
    echo "$(date '+%Y-%m-%d-%H-%M-%S'), Starting training..."
    
    # Run training script (nanhu version uses direct python, not ray job submit)
    # bash $WORK_DIR/run_cmd/qwen3-8b_grpo_nodes_nanhu_pp4.sh 2>&1
    # bash $WORK_DIR/run_cmd/qwen3-30b-021.sh 2>&1
    # bash $WORK_DIR/run_cmd/qwen3-30b-021-clip-normal-test.sh 2>&1
    # bash $WORK_DIR/run_cmd/qwen3-30b-021-noclip-test.sh 2>&1
    # bash $WORK_DIR/run_cmd/qwen3-30b-021-assign-logp.sh 2>&1
    bash $WORK_DIR/run_cmd/qwen3-30b-021-precision-test.sh 2>&1
    
    echo "$(date '+%Y-%m-%d-%H-%M-%S'), Training done."
else
    # === WORKER NODE (RANK>0) ===
    echo "$(date '+%Y-%m-%d-%H-%M-%S'), Worker node RANK=$RANK, waiting 10s for head node..."
    sleep 10
    echo "$(date '+%Y-%m-%d-%H-%M-%S'), Connecting to Ray head at $MASTER_ADDR:6379..."
    
    ray start --address="$MASTER_ADDR:6379"
    ray status
    
    # Keep worker alive while training runs on master
    sleep 360000
fi