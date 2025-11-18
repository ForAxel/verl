# 单GPU启动，使用已有的CKPT数据
# 模型、数据可以正常加载，卡在fit阶段

set -x

# 直接使用下载的模型参数和mcore参数
HF_MODEL_PATH='/home/dist/zhaoping/LLMs/Qwen3-1.7B'
DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/Qwen3-1.7B-mcore'

export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# export MUSA_VISIBLE_DEVICES='7'
export MUSA_EXECUTION_TIMEOUT=3200000
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
export RAY_BACKEND_LOG_LEVEL=debug
export HYDRA_FULL_ERROR=1
#export MUSA_USERQ=1

# export MUSA_PATCH_PATH=/home/dist/zhaoping/Code/verl-musa-patch
export MEGATRON_PATH=/home/dist/zhaoping/Code/musa_patch/Megatron-LM
export VERL_PATH=/home/dist/zhaoping/Code/verl-musa-patch/verl
export PYTHONPATH=${MEGATRON_PATH}:${VERL_PATH}:${MUSA_PATCH_PATH}:$PYTHONPATH


DATASET_PATH="/home/dist/zhaoping/Data/AM-Thinking-v1-RL-Dataset"
train_files=$DATASET_PATH/math_train.parquet
test_files=$DATASET_PATH/math_test.parquet

# 需要指定到 Verl 中对应config路径
CONFIG_PATH="/home/dist/zhaoping/Code/verl-musa-patch/verl/verl/trainer/config"


# # 解决保存问题
# export TORCH_SAFE_SERIALIZATION=1

export TOKENIZERS_PARALLELISM=false # 禁用 tokenizer并行化
# export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_MCCL_BLOCKING_WAIT=1
export MCCL_TIMEOUT=600000  # 单位：毫秒（600000ms = 10分钟）
export TORCH_MCCL_TRACE_BUFFER_SIZE=1048576  # 启用NCCL详细日志（如日志提示）

# 输出详细报错信息
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_CPP_LOG_LEVEL=INFO
# export MCCL_DEBUG=INFO
export MUSA_LAUNCH_BLOCKING=1 # MUSA 操作同步，用于定位错误

export MCCL_ASYNC_ENABLE=0           # 禁用异步操作
export MCCL_BUFFSIZE=16777216       # 调整缓冲区大小
export MCCL_TIMEOUT=180
export MCCL_RETRIES=3

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
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_k=100 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=50 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    algorithm.use_kl_in_reward=False \
    trainer.device='musa' \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='Qwen3_1.7b_megatron_sglang' \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=10 $@ \
| tee ../logs/run_ppo_multiGPU.log 2>&1
