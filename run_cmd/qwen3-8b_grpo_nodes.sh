# 单GPU启动，使用已有的CKPT数据
# 模型、数据可以正常加载，卡在fit阶段

set -x

# 直接使用下载的模型参数和mcore参数
HF_MODEL_PATH='/mnt/seed17/001688/zhaoping/LLMs/Qwen3-8B'
#HF_MODEL_PATH='/mnt/seed-program-nas/001688/kechun.wu/tmp0119/Qwen3-8B-Base'
DIST_CKPT_PATH='/mnt/seed17/001688/zhaoping/LLMs/MCORE/Qwen3-8B'

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_EXECUTION_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0

export MCCL_IB_GID_INDEX=3
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_ALGOS=1
export MCCL_BUFFSIZE=20480000


export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export VERL_LOGGING_LEVEL=INFO #INFO
export HYDRA_FULL_ERROR=1
#export MUSA_USERQ=1

export MUSA_PATCH_PATH=/home/megatron-lm-musa-patch/
export MEGATRON_PATH=/home/Megatron-LM
export VERL_PATH=/mnt/seed-program-nas/001688/kechun.wu/tmp0119/verl
export PYTHONPATH=${MEGATRON_PATH}:${VERL_PATH}:${MUSA_PATCH_PATH}:$PYTHONPATH


DATASET_PATH="/mnt/seed17/001688/zhaoping/Data/AM-Thinking-v1-RL-Dataset"
train_files=$DATASET_PATH/math_train.parquet
test_files=$DATASET_PATH/math_test.parquet

train_files=/mnt/seed-program-nas/001688/kechun.wu/verl-musa-patch/examples/data/dapo_train_16k.parquet
test_files=/mnt/seed-program-nas/001688/kechun.wu/verl-musa-patch/examples/data/dapo_val_1k.parquet


#train_files=/mnt/seed-program-nas/001688/kechun.wu/verl-musa-patch/examples/data/lighteval-MATH-preprocessed/train.parquet
#train_files=/mnt/seed-program-nas/001688/kechun.wu/verl-musa-patch/examples/data/lighteval-MATH-preprocessed/test2.parquet


# 需要指定到 Verl 中对应config路径
CONFIG_PATH=$VERL_PATH/verl/trainer/config


# # 解决保存问题
# export TORCH_SAFE_SERIALIZATION=1

export TOKENIZERS_PARALLELISM=false # 禁用 tokenizer并行化
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
max_response_length=4096
max_response_length=16384
max_response_length=32768
#max_response_length=32256
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))

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
runtime_env=/mnt/seed-program-nas/001688/kechun.wu/tmp0119/verl/run_cmd/runtime_env.yaml
RAY_ADDRESS='http://10.202.43.221:8265' ray job submit \
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
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
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
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=9 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=6 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
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
    trainer.project_name='verl_grpo_dapo' \
    trainer.experiment_name='Qwen3-8B_BASE_megatron_sglang_32k_pp4' \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.nnodes=2 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=10 #$@ \
    #2>&1 | tee ../logs/qwen3-8b_ppo_dp2pp2.log
