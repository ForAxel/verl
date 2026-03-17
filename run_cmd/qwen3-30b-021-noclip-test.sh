set -x

# 直接使用下载的模型参数和mcore参数
# HF_MODEL_PATH='/mnt/seed17/001688/zhaoping/LLMs/Qwen3-30B-A3B'
# DIST_CKPT_PATH='/mnt/seed17/001688/zhaoping/LLMs/MCORE/Qwen3-30B-A3B'
HF_MODEL_PATH='/mnt/seed17/001688/rl_models/Qwen3-30B-A3B-Instruct-2507'
DIST_CKPT_PATH='/mnt/seed17/001688/rl_models/MCORE/Qwen3-30B-A3B-Instruct-2507'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME=${MODEL_NAME:-"Qwen3-30B-A3B-Instruct-2507"}
EXPERIMENT_NAME="${MODEL_NAME}_megatron_sglang_32k_tp4_pp4_${TIMESTAMP}"

# export MUSA_VISIBLE_DEVICES='0,1'
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# export MUSA_EXECUTION_TIMEOUT=30000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0

# export MCCL_IB_GID_INDEX=3
# export MUSA_BLOCK_SCHEDULE_MODE=1
# export MCCL_ALGOS=1
# export MCCL_BUFFSIZE=20480000


# export ACCELERATE_USE_FSDP=1
# export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export VERL_LOGGING_LEVEL=INFO #INFO
export HYDRA_FULL_ERROR=1
#export MUSA_USERQ=1

export MUSA_PATCH_PATH=/home/megatron-lm-musa-patch
export MEGATRON_PATH=/home/Megatron-LM
#export VERL_PATH=/home/verl
export VERL_PATH=/mnt/seed17/001688/shenyichong/verl
export PYTHONPATH=${MEGATRON_PATH}:${VERL_PATH}:${MUSA_PATCH_PATH}:$PYTHONPATH


DATASET_PATH="/mnt/seed17/001688/rl_models/data"
train_files=$DATASET_PATH/amt_math17k_d_qweninstruct.parquet
test_files="['$DATASET_PATH/aime_2024.parquet','$DATASET_PATH/aime_2025.parquet']"
# test_files=$DATASET_PATH/aime_2024.parquet


# DATASET_PATH="/mnt/seed17/001688/shenyichong/verl-musa-patch/examples/data/"
# train_files=$DATASET_PATH/dapo_train_16k.parquet
# test_files=$DATASET_PATH/dapo_val_1k.parquet

# 需要指定到 Verl 中对应config路径
CONFIG_PATH=$VERL_PATH/verl/trainer/config


# # 解决保存问题
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_SAFE_SERIALIZATION=1

export VLLM_PATCH_MUSA_CUSTOM_OPS=1



use_dynamic_bsz=True

max_prompt_length=2048
# max_response_length=4096
# max_response_length=16384
# max_response_length=65536
max_response_length=32768
#max_response_length=1024
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
# actor_ppo_max_token_len=32768
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
# infer_ppo_max_token_len=32768
#infer_ppo_max_token_len=49152
max_response_length_=1024
#    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=18 \
env PYTHONPATH="$PYTHONPATH" \
    MUSA_VISIBLE_DEVICES="$MUSA_VISIBLE_DEVICES" \
    ACCELERATOR_BACKEND="$ACCELERATOR_BACKEND" \
    RAY_LOGGING_LEVEL=WARNING \
    RAY_DEDUP_LOGS=1 \
    RAY_ADDRESS="localhost:65379" \
    VLLM_PATCH_MUSA_CUSTOM_OPS=1 \
    #+actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=False \
    #+actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False \
runtime_env=/mnt/seed17/001688/shenyichong/verl/run_cmd/qwen3_30b_a3b_runtime_env.yaml
JOB_OUTPUT=$(RAY_ADDRESS='http://localhost:8872' ray job submit \
    --runtime-env=$runtime_env \
    --no-wait \
    -- python3 -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='ppo_megatron_trainer_demo.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.prompt_key=prompt \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=4e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.profiler.enable=False \
    actor_rollout_ref.actor.megatron.use_mbridge=False \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.batch_p2p_comm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=48 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=6 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="alltoall" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.clip_ratio_low=0.0 \
    actor_rollout_ref.actor.clip_ratio_high=0.0 \
    actor_rollout_ref.actor.clip_ratio_c=1.001 \
    actor_rollout_ref.actor.policy_loss.loss_mode="gpg" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.expert_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.ref.megatron.sequence_parallel=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_grpo_qwen3_30b_021' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.nnodes=4 \
    trainer.save_freq=150 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    # > logs/$EXPERIMENT_NAME.log 2>&1)
    2>&1)

echo "$JOB_OUTPUT"


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
