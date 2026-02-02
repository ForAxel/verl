export MEGATRON_PATH=/home/Megatron-LM
export VERL_PATH=/home/verl
export PYTHONPATH=${MEGATRON_PATH}:${VERL_PATH}:$PYTHONPATH

# HF_MODEL_PATH='/home/dist/zhaoping/LLMs/Qwen3-1.7B'
# DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/Qwen3-1.7B-mcore'

# HF_MODEL_PATH='/home/dist/zhaoping/LLMs/Qwen2-7B-Instruct'
# DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/Qwen2-7B-Instruct'

# HF_MODEL_PATH='/home/dist/zhaoping/LLMs/Qwen3-30B-A3B'
# DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/Qwen3-30B-A3B'

# HF_MODEL_PATH='/home/dist/zhaoping/LLMs/DeepSeek-V2-Lite'
# DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/DeepSeek-V2-Lite'

# HF_MODEL_PATH='/home/dist/zhaoping/LLMs/Qwen3-8B'
# DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/Qwen3-8B'

# HF_MODEL_PATH='/mnt/seed17/001688/zhaoping/LLMs/DeepSeek-V2-Lite'
# DIST_CKPT_PATH='/mnt/seed17/001688/zhaoping/LLMs/MCORE/DeepSeek-V2-Lite'

HF_MODEL_PATH='/mnt/seed17/001688/zhaoping/LLMs/Qwen3-30B-A3B'
DIST_CKPT_PATH='/mnt/seed17/001688/zhaoping/LLMs/MCORE/Qwen3-30B-A3B'


# /home/dist/zhaoping/LLMs 存储了大量LLM模型参数
python -u ../scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH