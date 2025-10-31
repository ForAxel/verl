# 单GPU启动，使用已有的CKPT数据
# 模型、数据可以正常加载，卡在fit阶段

set -x

# 直接使用下载的模型参数和mcore参数
HF_MODEL_PATH='/home/dist/zhaoping/LLMs/Qwen3-1.7B'
DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/Qwen3-1.7B-mcore'

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
export OMP_NUM_THREADS=4
# export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_VISIBLE_DEVICES='0'
export MUSA_EXECUTION_TIMEOUT=3200000
# export MUSA_KERNEL_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0

# export MCCL_IB_GID_INDEX=3
# export MUSA_BLOCK_SCHEDULE_MODE=1
# export MCCL_ALGOS=1
# export MCCL_BUFFSIZE=20480000


# export ACCELERATE_USE_FSDP=1
# export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export VERL_LOGGING_LEVEL=WARNING #INFO
export HYDRA_FULL_ERROR=1
#export MUSA_USERQ=1

export MUSA_PATCH_PATH=/home/dist/zhaoping/Code/verl-musa-patch
export MEGATRON_PATH=/home/dist/zhaoping/Code/musa_patch/Megatron-LM
export VERL_PATH=/home/dist/zhaoping/Code/verl-musa-patch/verl
export PYTHONPATH=${VERL_PATH}:${MEGATRON_PATH}:${MUSA_PATCH_PATH}:$PYTHONPATH


DATASET_PATH="/home/dist/zhaoping/Data/AM-Thinking-v1-RL-Dataset"
train_files=$DATASET_PATH/math_train.parquet
test_files=$DATASET_PATH/math_test.parquet

# 需要指定到 Verl 中对应config路径
CONFIG_PATH="/home/dist/zhaoping/Code/verl-musa-patch/verl/verl/trainer/config"


# ======= DEBUG ======
# 增加NCCL超时设置
export MCCL_BLOCKING_WAIT=1
export MCCL_ASYNC_ERROR_HANDLING=1
# 调整网络绑定
export GLOO_SOCKET_IFNAME=bond0
# 减少通信量
export MCCL_MAX_NCHANNELS=1  # 确保已设置

# 减少Ray工作进程数
export RAY_task_retry_delay_ms=5000
export RAY_max_task_retries=3
export RAY_num_cpus=32  # 根据实际CPU核心数调整
# 禁用Python的子进程fork
export PYTHON_DISABLE_FORK=1

# 改成单batch看下效果  data.train_batch_size=1 actor_rollout_ref.actor.ppo_mini_batch_size=1
# ======

env PYTHONPATH="$PYTHONPATH" \
    MUSA_VISIBLE_DEVICES="$MUSA_VISIBLE_DEVICES" \
    ACCELERATOR_BACKEND="$ACCELERATOR_BACKEND" \
    RAY_LOGGING_LEVEL=WARNING \
    RAY_DEDUP_LOGS=0 \
    RAY_ADDRESS="10.18.33.9:65379" \
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='ppo_megatron_trainer_demo.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=2 \
    data.max_prompt_length=256 \
    data.max_response_length=32 \
    data.filter_overlong_prompts=True \
    data.prompt_key=prompt \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='Qwen3_1.7b_megatron_sglang' \
    trainer.n_gpus_per_node=1 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=1000 \
    trainer.total_epochs=10 $@ \
| tee ../logs/demo_run.log 2>&1
