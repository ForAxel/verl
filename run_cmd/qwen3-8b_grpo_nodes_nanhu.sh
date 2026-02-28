set -x

# Use Model_PATH environment variable to select model
# Options: "Qwen3-8B" or "Qwen3-8B-Base"
Model_PATH=${Model_PATH:-"Qwen3-8B"}
# Control whether to resume from an RL checkpoint.
# 1: use LOAD_CKPT_DIR if valid; 0: always start from pretrained init (DIST_CKPT_PATH/HF).
ENABLE_CKPT_RESUME=${ENABLE_CKPT_RESUME:-0}
# Explicit RL checkpoint path to load (should be a global_step_* directory).
LOAD_CKPT_DIR=${LOAD_CKPT_DIR:-""}

# Set model paths based on Model_PATH
if [ "$Model_PATH" = "Qwen3-8B-Base" ]; then
    HF_MODEL_PATH='/mnt/seed17/001688/shenyichong/models/Qwen3-8B-Base'
    DIST_CKPT_PATH='/mnt/seed17/001688/zhaoping/LLMs/MCORE/Qwen3-8B-Base'
    MODEL_NAME="Qwen3-8B-Base"
else
    HF_MODEL_PATH='/mnt/seed17/001688/zhaoping/LLMs/Qwen3-8B'
    DIST_CKPT_PATH='/mnt/seed17/001688/zhaoping/LLMs/MCORE/Qwen3-8B'
    MODEL_NAME="Qwen3-8B"
fi

# Generate unique experiment name with timestamp for separate ray logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_TAG=${EXPERIMENT_TAG:-"megatron_sglang_32k_tp4"}
# Allow overriding experiment name for deterministic resume behavior.
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"${MODEL_NAME}_${EXPERIMENT_TAG}_${TIMESTAMP}"}
PROJECT_NAME="verl_grpo_dapo"
# Save checkpoint path: always save under this directory.
CKPT_BASE_DIR="/mnt/seed17/001688/shenyichong/verl/checkpoints"
SAVE_CKPT_DIR="${CKPT_BASE_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# Resume policy:
# - If ENABLE_CKPT_RESUME=1 and LOAD_CKPT_DIR is valid -> resume from LOAD_CKPT_DIR
# - Otherwise -> train from pretrained init (DIST_CKPT_PATH/HF)
mkdir -p "${SAVE_CKPT_DIR}"
if [ "${ENABLE_CKPT_RESUME}" = "1" ] && [ -n "${LOAD_CKPT_DIR}" ] && [ -d "${LOAD_CKPT_DIR}" ] && [[ "${LOAD_CKPT_DIR}" == *global_step_* ]]; then
    TRAINER_RESUME_MODE="resume_path"
    TRAINER_RESUME_FROM_PATH="${LOAD_CKPT_DIR}"
    echo "Resume enabled: ${TRAINER_RESUME_FROM_PATH}"
    RESUME_ARGS=(
        trainer.resume_mode="${TRAINER_RESUME_MODE}"
        trainer.resume_from_path="${TRAINER_RESUME_FROM_PATH}"
    )
else
    TRAINER_RESUME_MODE="disable"
    TRAINER_RESUME_FROM_PATH=""
    if [ "${ENABLE_CKPT_RESUME}" = "1" ] && [ -n "${LOAD_CKPT_DIR}" ]; then
        echo "LOAD_CKPT_DIR is invalid (${LOAD_CKPT_DIR}). Start from pretrained init via DIST_CKPT_PATH/HF_MODEL_PATH."
    elif [ "${ENABLE_CKPT_RESUME}" = "1" ]; then
        echo "LOAD_CKPT_DIR is not provided. Start from pretrained init via DIST_CKPT_PATH/HF_MODEL_PATH."
    else
        echo "Checkpoint lookup disabled (ENABLE_CKPT_RESUME=0). Start from pretrained init via DIST_CKPT_PATH/HF_MODEL_PATH."
    fi
    RESUME_ARGS=(
        trainer.resume_mode="${TRAINER_RESUME_MODE}"
    )
fi

echo "=== Experiment Configuration ==="
echo "Model_PATH: $Model_PATH"
echo "HF_MODEL_PATH: $HF_MODEL_PATH"
echo "DIST_CKPT_PATH: $DIST_CKPT_PATH"
echo "EXPERIMENT_TAG: $EXPERIMENT_TAG"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "SAVE_CKPT_DIR: $SAVE_CKPT_DIR"
echo "ENABLE_CKPT_RESUME: $ENABLE_CKPT_RESUME"
echo "LOAD_CKPT_DIR: $LOAD_CKPT_DIR"
echo "TRAINER_RESUME_MODE: $TRAINER_RESUME_MODE"
if [ -n "$TRAINER_RESUME_FROM_PATH" ]; then
    echo "TRAINER_RESUME_FROM_PATH: $TRAINER_RESUME_FROM_PATH"
fi
echo "================================"

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
export OMP_NUM_THREADS=4

VERL_PATH=/mnt/seed17/001688/shenyichong/verl


DATASET_PATH="/mnt/seed17/001688/zhaoping/Data/AM-Thinking-v1-RL-Dataset"
train_files=$DATASET_PATH/math_train.parquet
test_files=$DATASET_PATH/math_test.parquet

train_files=/mnt/seed17/001688/shenyichong/verl-musa-patch/examples/data/dapo_train_16k.parquet
test_files=/mnt/seed17/001688/shenyichong/verl-musa-patch/examples/data/dapo_val_1k.parquet


#train_files=/mnt/seed17/001688/shenyichong/verl-musa-patch/examples/data/lighteval-MATH-preprocessed/train.parquet
#train_files=/mnt/seed17/001688/shenyichong/verl-musa-patch/examples/data/lighteval-MATH-preprocessed/test2.parquet


# 需要指定到 Verl 中对应config路径
CONFIG_PATH=$VERL_PATH/verl/trainer/config


# # 解决保存问题
# export TORCH_SAFE_SERIALIZATION=1

# export TORCH_NCCL_BLOCKING_WAIT=1
#export TORCH_MCCL_BLOCKING_WAIT=1
#export MCCL_TIMEOUT=600000  # 单位：毫秒（600000ms = 10分钟）
#export TORCH_MCCL_TRACE_BUFFER_SIZE=1048576  # 启用NCCL详细日志（如日志提示）

# 输出详细报错信息
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_CPP_LOG_LEVEL=INFO
# export MCCL_DEBUG=INFO
#export MUSA_LAUNCH_BLOCKING=0 # MUSA 操作同步，用于定位错误

# export MCCL_ASYNC_ENABLE=0           # 禁用异步操作
# export MCCL_BUFFSIZE=16777216       # 调整缓冲区大小
# export MCCL_TIMEOUT=180
# export MCCL_RETRIES=3

#export MUSA_ERROR_DUMP_VERBOSE=1

use_dynamic_bsz=True

max_prompt_length=512
# Keep total length within each model's context window.
# Qwen3-8B-Base: 32768 max_position_embeddings -> max_response_length must be <= 32256 when prompt is 512.
if [ "$Model_PATH" = "Qwen3-8B-Base" ]; then
    max_response_length=32255
else
    max_response_length=32768
fi
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))

export PYTHONUNBUFFERED=1 
export MUSA_LAUNCH_BLOCKING=0
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH
# env PYTHONPATH="$PYTHONPATH" \
#     ACCELERATOR_BACKEND="$ACCELERATOR_BACKEND" \
#     PYTHONUNBUFFERED=1 \
#     RAY_LOGGING_LEVEL=WARNING \
#     RAY_DEDUP_LOGS=0 \
#     MUSA_ERROR_DUMP_VERBOSE=1 \
#     RAY_ADDRESS="localhost:65379" \
runtime_env=/mnt/seed17/001688/shenyichong/verl/run_cmd/runtime_env.yaml

# Submit job and capture output to extract job ID
JOB_OUTPUT=$(RAY_ADDRESS='http://127.0.0.1:8872' ray job submit \
    --runtime-env=$runtime_env \
    --no-wait \
    -- python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='ppo_megatron_trainer_demo.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=64 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.prompt_key=source_prompt \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.batch_p2p_comm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.no_gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_softmax_in_fp32=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.accumulate_allreduce_grads_in_fp32=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.no_masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.no_bias_swiglu_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=12 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_k=100 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=50 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${SAVE_CKPT_DIR}" \
    "${RESUME_ARGS[@]}" \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.nnodes=2 \
    trainer.save_freq=5 \
    trainer.test_freq=1000 \
    trainer.total_epochs=10 \
    2>&1)

echo "$JOB_OUTPUT"

# Extract job ID and follow logs to keep platform job active
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP "raysubmit_[a-zA-Z0-9]+" | head -n1)
if [ -n "$JOB_ID" ]; then
    echo "=========================================="
    echo "Job submitted: $JOB_ID"
    echo "Following logs (this keeps the platform job active)..."
    echo "=========================================="
    RAY_ADDRESS='http://127.0.0.1:8872' ray job logs "$JOB_ID" --follow 2>&1 | tee ../logs/${EXPERIMENT_NAME}.log
else
    echo "ERROR: Failed to extract job ID from output"
    echo "$JOB_OUTPUT"
    exit 1
fi
