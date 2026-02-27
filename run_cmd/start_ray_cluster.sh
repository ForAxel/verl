#!/bin/bash
# Start Ray head on current (master) node, then discover worker nodes and start Ray on them.
#
# Default: discover hosts from ~/.ssh/config (grep "Host "), exclude current node, start workers via SSH.
#
# Override discovery (first match wins):
#   - Slurm: SLURM_JOB_NODELIST (workers via srun).
#   - Env:   RAY_WORKER_NODES="worker-0,worker-1" (comma-separated).
#   - File:  RAY_NODE_FILE=/path/to/file (first line = head, rest = workers).
#
# Optional: RAY_PORT=65379, RAY_DASHBOARD_PORT=8872, RAY_HEAD_IP=<ip>.

set -e

RAY_PORT="${RAY_PORT:-65379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8872}"
CURRENT_HOST=$(hostname -s)

# ---------------------------------------------------------------------------
# 1. Start Ray head on current node
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Stopping any existing Ray on this node..."
echo "=========================================="
ray stop --force 2>/dev/null || true
sleep 2

echo ""
echo "=========================================="
echo "Starting Ray HEAD on $(hostname) (${CURRENT_HOST})"
echo "=========================================="
ray start --head \
  --port="${RAY_PORT}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT}"

# Get head node IP (this machine)
HEAD_IP=$(hostname -I | awk '{print $1}')
if [ -n "${RAY_HEAD_IP}" ]; then
  HEAD_IP="${RAY_HEAD_IP}"
fi

RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
echo "Ray head address: ${RAY_ADDRESS}"
echo "Dashboard: http://${HEAD_IP}:${RAY_DASHBOARD_PORT}"

# ---------------------------------------------------------------------------
# 2. Discover worker nodes (default: from ~/.ssh/config)
# ---------------------------------------------------------------------------
WORKER_NODES=()
USE_SLURM=false

if [ -n "${SLURM_JOB_NODELIST}" ]; then
  mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  for n in "${ALL_NODES[@]:1}"; do
    WORKER_NODES+=("$n")
  done
  USE_SLURM=true
  echo "Discovered ${#WORKER_NODES[@]} worker(s) from SLURM_JOB_NODELIST"
elif [ -n "${RAY_WORKER_NODES}" ]; then
  IFS=',' read -ra WORKER_NODES <<< "${RAY_WORKER_NODES}"
  echo "Using RAY_WORKER_NODES: ${#WORKER_NODES[@]} worker(s)"
elif [ -n "${RAY_NODE_FILE}" ] && [ -f "${RAY_NODE_FILE}" ]; then
  while read -r line; do
    line=$(echo "$line" | sed 's/#.*//;s/^[[:space:]]*//;s/[[:space:]]*$//')
    [ -z "$line" ] && continue
    [ "$line" != "$CURRENT_HOST" ] && WORKER_NODES+=("$line")
  done < <(tail -n +2 "${RAY_NODE_FILE}")
  echo "Using RAY_NODE_FILE ${RAY_NODE_FILE}: ${#WORKER_NODES[@]} worker(s)"
else
  # Default: parse ~/.ssh/config for "Host " entries, SSH to each and start worker (skip master/current host)
  SSH_CONFIG="${HOME}/.ssh/config"
  if [ -f "${SSH_CONFIG}" ]; then
    echo "Parsing ~/.ssh/config for worker nodes..."
    SEEN_REMOTE=""
    ALL_HOSTS=$(grep -E "^[[:space:]]*Host " "${SSH_CONFIG}" | sed 's/^[[:space:]]*Host[[:space:]]*//' | tr ' ' '\n' | sort -u)
    while read -r name; do
      [ -z "$name" ] && continue
      [[ "$name" == *"*"* ]] && continue
      [[ "$name" == "localhost" ]] && continue
      # Skip hosts with "master" in the name (common pattern)
      [[ "$name" == *"master"* ]] && echo "  Skipping master host: $name" && continue
      
      # Try to verify this is a different machine (optional check)
      remote_host=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes "$name" hostname -s 2>&1)
      ssh_exit=$?
      if [ $ssh_exit -ne 0 ]; then
        echo "  Warning: Cannot SSH to $name (will still try to start Ray worker): ${remote_host}"
        # Still add it - SSH might fail but Ray start might work
        WORKER_NODES+=("$name")
        continue
      fi
      
      if [ -z "$remote_host" ] || [ "$remote_host" = "$CURRENT_HOST" ]; then
        echo "  Skipping $name (same host as master: ${remote_host:-unknown})"
        continue
      fi
      
      # One worker per distinct machine (skip if we already have this remote_host)
      if [[ " ${SEEN_REMOTE} " == *" ${remote_host} "* ]]; then
        echo "  Skipping $name (duplicate of ${remote_host})"
        continue
      fi
      SEEN_REMOTE="${SEEN_REMOTE} ${remote_host}"
      echo "  Found worker: $name (hostname: ${remote_host})"
      WORKER_NODES+=("$name")
    done <<< "$ALL_HOSTS"
    echo "Discovered ${#WORKER_NODES[@]} worker(s) from ~/.ssh/config"
  else
    echo "~/.ssh/config not found, skipping worker discovery"
  fi
fi

# ---------------------------------------------------------------------------
# 3. Start Ray on each worker (Slurm: srun; else: SSH)
# ---------------------------------------------------------------------------
if [ ${#WORKER_NODES[@]} -eq 0 ]; then
  echo "No worker nodes configured. Ray head only (single node)."
  ray status
  exit 0
fi

echo ""
echo "=========================================="
echo "Starting Ray workers (address=${RAY_ADDRESS})"
echo "=========================================="

for node in "${WORKER_NODES[@]}"; do
  echo ""
  echo "--- Worker: ${node} ---"
  if [ "$USE_SLURM" = true ]; then
    srun --nodes=1 --ntasks=1 -w "$node" bash -c "
      ray stop --force 2>/dev/null || true
      sleep 2
      ray start --address='${RAY_ADDRESS}' \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${RAY_DASHBOARD_PORT}
      echo \"Ray worker started on \$(hostname)\"
    "
  else
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${node}" bash -s << EOF
      ray stop --force 2>/dev/null || true
      sleep 2
      ray start --address='${RAY_ADDRESS}' \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${RAY_DASHBOARD_PORT}
      echo "Ray worker started on \$(hostname)"
EOF
  fi
  if [ $? -eq 0 ]; then
    echo "  OK ${node}"
  else
    echo "  FAILED ${node}"
  fi
done

# Wait for workers to register with head (Ray needs a moment to discover nodes)
if [ ${#WORKER_NODES[@]} -gt 0 ]; then
  echo ""
  echo "Waiting for workers to register with head node..."
  EXPECTED_NODES=$((1 + ${#WORKER_NODES[@]}))  # 1 head + N workers
  MAX_WAIT=30
  WAITED=0
  while [ $WAITED -lt $MAX_WAIT ]; do
    ACTIVE_NODES=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
    if [ "$ACTIVE_NODES" -ge "$EXPECTED_NODES" ]; then
      echo "All ${EXPECTED_NODES} node(s) registered!"
      break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo "  Waiting... (${WAITED}s, found ${ACTIVE_NODES}/${EXPECTED_NODES} nodes)"
  done
  if [ $WAITED -ge $MAX_WAIT ]; then
    echo "  Warning: Timeout waiting for all nodes (found ${ACTIVE_NODES}/${EXPECTED_NODES})"
  fi
fi

echo ""
echo "=========================================="
echo "Ray cluster status"
echo "=========================================="
ray status
