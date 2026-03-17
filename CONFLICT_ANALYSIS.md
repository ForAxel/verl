# Git 合并冲突文件分类

## 冲突类型说明
- **UU (both modified)**: 两个分支都修改了同一文件，需要手动合并内容
- **AA (both added)**: 两个分支都添加了同名文件，但内容不同
- **UD (deleted by them)**: `mt/dev_260127` 分支删除了文件，但 `mt/dev_1031` 分支修改了它
- **UA (added by them)**: `mt/dev_260127` 分支添加了新文件/目录

---

## 1. 配置文件冲突 (UU - 双方修改)

### 根目录配置
- `.gitignore` - Git 忽略规则文件

### 脚本文件
- `scripts/converter_hf_to_mcore.py` - 模型转换脚本

### 运行时配置
- `runtime_env.yaml` - 运行时环境配置

---

## 2. 运行脚本冲突 (AA - 双方都添加了同名文件)

**位置**: `run_cmd/` 目录

这些脚本在两个分支中都存在但内容不同，需要决定保留哪个版本或手动合并：

- `run_cmd/convert_weight.sh`
- `run_cmd/dsv2-lite_ppo_ep4.sh`
- `run_cmd/dsv2-lite_ppo_ep8.sh`
- `run_cmd/qwen3-8b_ppo_dp2pp2.sh`
- `run_cmd/qwen3-8b_ppo_dp2tp2.sh`
- `run_cmd/qwen3-8b_ppo_single.sh`
- `run_cmd/qwen3-8b_ppo_tp4.sh`
- `run_cmd/run_ppo_demo.sh`
- `run_cmd/run_ppo_dp2tp2.sh`
- `run_cmd/run_ppo_dp2tp4.sh`
- `run_cmd/run_ppo_dp4.sh`
- `run_cmd/run_ppo_tp2.sh`
- `run_cmd/run_ppo_tp4.sh`
- `run_cmd/start_ray.sh`

---

## 3. 核心代码冲突 (UU - 双方修改)

### 3.1 实验性功能模块
- `verl/experimental/one_step_off_policy/ray_trainer.py`
- `verl/experimental/transfer_queue/main_ppo.py`

### 3.2 模型相关
- `verl/models/mcore/config_converter.py` - 配置转换器
- `verl/models/mcore/registry.py` - 模型注册表
- `verl/models/mcore/weight_converter.py` - 权重转换器
- `verl/models/weight_loader_registry.py` - 权重加载器注册表

### 3.3 单控制器模块
- `verl/single_controller/base/worker.py` - 基础工作器
- `verl/single_controller/ray/base.py` - Ray 基础模块

### 3.4 训练器配置
- `verl/trainer/config/_generated_ppo_megatron_trainer.yaml` - 生成的 PPO Megatron 训练器配置
- `verl/trainer/config/actor/actor.yaml` - Actor 配置
- `verl/trainer/config/engine/megatron.yaml` - Megatron 引擎配置
- `verl/trainer/config/ppo_megatron_trainer_demo.yaml` (AA - 双方都添加)

### 3.5 训练器核心代码
- `verl/trainer/fsdp_sft_trainer.py` - FSDP SFT 训练器
- `verl/trainer/main_ppo.py` - PPO 主程序
- `verl/trainer/ppo/ray_trainer.py` - PPO Ray 训练器

### 3.6 工具函数
- `verl/utils/attention_utils.py` - 注意力工具函数
- `verl/utils/dataset/rl_dataset.py` - RL 数据集工具
- `verl/utils/device.py` - 设备工具
- `verl/utils/flops_counter.py` - FLOPs 计数器
- `verl/utils/megatron/dist_checkpointing.py` - 分布式检查点
- `verl/utils/model.py` - 模型工具
- `verl/utils/profiler/profile.py` - 性能分析器
- `verl/utils/reward_score/__init__.py` - 奖励分数模块
- `verl/utils/torch_functional.py` - PyTorch 函数式工具

### 3.7 Worker 模块
- `verl/workers/actor/dp_actor.py` - 数据并行 Actor
- `verl/workers/actor/megatron_actor.py` - Megatron Actor
- `verl/workers/engine/megatron/transformer_impl.py` - Transformer 实现
- `verl/workers/fsdp_workers.py` - FSDP Workers
- `verl/workers/megatron_workers.py` - Megatron Workers
- `verl/workers/reward_model/megatron/reward_model.py` - Megatron 奖励模型
- `verl/workers/rollout/sglang_rollout/sglang_rollout.py` - SGLang Rollout
- `verl/workers/rollout/sglang_rollout/utils.py` - SGLang 工具函数
- `verl/workers/utils/padding.py` - 填充工具

---

## 4. 删除/修改冲突 (UD - 对方删除，我方修改)

这些文件在 `mt/dev_260127` 中被删除，但在 `mt/dev_1031` 中被修改了。需要决定：
- **保留文件**: `git add <file>` (保留你的修改)
- **删除文件**: `git rm <file>` (接受对方的删除)

- `recipe/prime/prime_ray_trainer.py` - Prime Ray 训练器
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` - VLLM Rollout SPMD
- `verl/workers/sharding_manager/fsdp_vllm.py` - FSDP VLLM 分片管理器
- `verl/workers/sharding_manager/megatron_vllm.py` - Megatron VLLM 分片管理器

---

## 5. 新增文件/目录 (UA - 对方添加)

- `recipe~origin_mt_dev_260127` - 这看起来是一个临时目录（可能是 Git 自动创建的）

---

## 解决建议

### 优先级 1: 核心代码文件 (UU)
这些是最重要的冲突，需要仔细审查和手动合并：
- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/workers/` 下的核心文件
- `verl/models/mcore/` 下的文件

### 优先级 2: 配置文件 (UU/AA)
- `.gitignore` - 通常可以合并两个版本的规则
- `runtime_env.yaml` - 需要检查环境差异
- `verl/trainer/config/` 下的 YAML 文件

### 优先级 3: 脚本文件 (AA)
- `run_cmd/` 下的脚本 - 可以比较两个版本，选择更合适的或合并

### 优先级 4: 删除冲突 (UD)
- 检查这些文件是否还在使用
- 如果 `mt/dev_260127` 删除了它们，可能是重构的一部分

---

## 快速解决命令参考

### 接受当前分支版本 (mt/dev_1031)
```bash
git checkout --ours <file>
git add <file>
```

### 接受合并分支版本 (mt/dev_260127)
```bash
git checkout --theirs <file>
git add <file>
```

### 删除文件（接受对方的删除）
```bash
git rm <file>
```

### 保留文件（拒绝对方的删除）
```bash
git add <file>
```

### 完成合并
```bash
git commit
```
