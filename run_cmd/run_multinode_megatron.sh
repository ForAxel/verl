# 提交任务到Ray上运行

set -x

HF_MODEL_PATH='/home/dist/zhaoping/LLMs/Qwen3-1.7B'
DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/Qwen3-1.7B-mcore'
# /home/dist/zhaoping/LLMs 存储了大量LLM模型参数
#python ../verl/scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# export MUSA_EXECUTION_TIMEOUT=30000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0

export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1
#export MUSA_USERQ=1
export MUSA_PATCH_PATH=/home/dist/zhaoping/Code/verl-musa-patch
export MEGATRON_PATH=/home/dist/zhaoping/Code/musa_patch/Megatron-LM
export VERL_PATH=/home/dist/zhaoping/Code/verl-musa-patch/verl
export PYTHONPATH=${VERL_PATH}:${MEGATRON_PATH}:${MUSA_PATCH_PATH}:$PYTHONPATH

# export VLLM_USE_V1='0'

export MUSA_LAUNCH_BLOCKING=1
export MUSA_MAX_CONCURRENT_LAUNCHES=1

DATASET_PATH="/home/dist/zhaoping/Data/AM-Thinking-v1-RL-Dataset"
train_files=$DATASET_PATH/math_train.parquet
test_files=$DATASET_PATH/math_test.parquet

CONFIG_PATH="/home/dist/zhaoping/Code/verl-musa-patch/verl/verl/trainer/config"


# ray job submit --address="10.18.32.9:65379" \
#     --no-wait\
#     -- \
#     python -c "import ray;ray.init();print('Test job')" # demo test

# RAY_ADDRESS="10.18.32.9:65379" python -c "import ray;ray.init();print('Test job direct connect Ray')"

# # success
# ray job submit --address="10.18.32.9:65379" \
#     --runtime-env=/home/dist/zhaoping/Code/verl-musa-patch/runtime_env.yaml \
#     --no-wait\
#     -- \
#     python -c "import ray;ray.init();print('Test job with runtime env')"

#  data.truncation='middle'

ray job submit --address="10.18.32.9:65379" \
    --runtime-env=/home/dist/zhaoping/Code/verl-musa-patch/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='ppo_megatron_trainer_demo.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=32 \
    data.filter_overlong_prompts=True \
    data.prompt_key=prompt \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
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
    trainer.experiment_name='Qwen3_1.7b_megatron_sglang_raySubmit_little' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=1000 \
    trainer.total_epochs=3 $@ \
 | tee ../logs/demo_little_multinode_megatron.log 2>&1
