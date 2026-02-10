set -x

# 直接使用下载的模型参数和mcore参数
HF_MODEL_PATH='/mnt/seed17/001688/zhaoping/LLMs/Qwen3-1.7B'
DIST_CKPT_PATH='/mnt/seed17/001688/zhaoping/LLMs/MCORE/Qwen3-1.7B-mcore'

export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0


export VERL_LOGGING_LEVEL=INFO #INFO
export HYDRA_FULL_ERROR=1
#export MUSA_USERQ=1

export MEGATRON_PATH=/home/Megatron-LM
export VERL_PATH=/home/verl
export PYTHONPATH=${MEGATRON_PATH}:${VERL_PATH}:${MUSA_PATCH_PATH}:$PYTHONPATH


DATASET_PATH="/mnt/seed17/001688/zhaoping/Data/AM-Thinking-v1-RL-Dataset"
train_files=$DATASET_PATH/math_train.parquet
test_files=$DATASET_PATH/math_test.parquet

# 需要指定到 Verl 中对应config路径
CONFIG_PATH=$VERL_PATH/verl/experimental/fully_async_policy/config

export VLLM_PATCH_MUSA_CUSTOM_OPS=1

# 禁用 PyTorch 的隐式初始化
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export NCCL_IB_DISABLE=1  # 如非 InfiniBand 环境

# 确保 Ray 使用独立端口
export RAY_BACKEND=auto

rollout_mode="async"
rollout_name="sglang" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*400)))
test_freq=10
staleness_threshold=0
trigger_parameter_sync_step=16
partial_rollout=False

env PYTHONPATH="$PYTHONPATH" \
    MUSA_VISIBLE_DEVICES="$MUSA_VISIBLE_DEVICES" \
    ACCELERATOR_BACKEND="$ACCELERATOR_BACKEND" \
    RAY_LOGGING_LEVEL=DEBUG \
    RAY_DEDUP_LOGS=0 \
    RAY_ADDRESS="localhost:65379" \
    VLLM_PATCH_MUSA_CUSTOM_OPS=1 \
    HYDRA_FULL_ERROR=1 \
python -m fully_async_main \
    --config-path="$CONFIG_PATH" \
    --config-name='fully_async_ppo_megatron_trainer_demo.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=16 \
    data.filter_overlong_prompts=True \
    data.prompt_key=prompt \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    +actor_rollout_ref.actor.strategy.detach_sync_backend="gloo" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
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
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='Qwen3_1.7b_megatron_sglang' \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=10 \
	data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.strategy=megatron \
    critic.strategy=megatron \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=4 \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.partial_rollout="${partial_rollout}" $@ \
    2>&1 | tee $VERL_PATH/logs/async/run_async_demo.log