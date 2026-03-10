# TPRL: Token Pruning via Reinforcement Learning

基于强化学习的视觉token剪枝框架,用于大型视觉语言模型(LVLMs)的推理加速。

## 方法概述

TPRL将视觉token剪枝建模为序列决策过程,通过强化学习优化剪枝策略:

1. **Autoencoder (可选)**: 将视觉token压缩到低维表示
2. **Learning from Demonstrations (LfD)**: 使用启发式方法生成的轨迹预训练策略网络
3. **PPO训练**: 使用Proximal Policy Optimization微调策略,优化任务性能和计算效率

## 项目结构

```
LVLMs_pruner/
├── model/
│   ├── autoencoder.py          # Token-wise autoencoder (待实现)
│   ├── rl_networks.py          # Policy和Value网络
│   ├── llava_mllm.py          # LLaVA模型实现
│   └── qwen_mllm.py           # Qwen模型实现
├── pruner/
│   ├── rl_pruner.py           # RL-based pruner
│   ├── random_pruner.py       # Random pruner (用于生成演示)
│   └── mlp_pruner.py          # MLP pruner
├── train_lfd.py               # LfD训练脚本
├── train_ppo.py               # PPO训练脚本
├── main.py                    # 评估脚本
└── config.py                  # 配置文件
```

## 配置说明

在 [config.py](config.py) 中设置超参数:

### 模型配置
```python
MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # 使用LLaVA-1.5-7B
USE_AUTOENCODER = False  # 当前版本不使用autoencoder
```

### LfD配置
```python
LFD_NUM_DEMOS = 50000      # 生成50K演示轨迹
LFD_NUM_STEPS = 3          # 每个轨迹3步
LFD_NUM_EPOCHS = 5         # 训练5个epoch
LFD_LEARNING_RATE = 5e-5   # 学习率
LFD_BATCH_SIZE = 128       # Batch size
```

### PPO配置
```python
PPO_NUM_EPOCHS = 200           # 训练200个epoch
PPO_ROLLOUT_BATCH_SIZE = 64    # 每次收集64个rollout
PPO_MINI_BATCH_SIZE = 16       # Mini-batch size
PPO_MAX_STEPS = 3              # 每个rollout最多3步
PPO_LR_ACTOR = 1e-5            # Actor学习率
PPO_LR_CRITIC = 3e-5           # Critic学习率
PPO_GAMMA = 0.99               # 折扣因子
PPO_CLIP_EPSILON = 0.2         # PPO clip ratio
RL_STEP_DISCOUNT = 0.5         # 步骤折扣因子 (lambda_disc)
```

## 使用流程

### 1. Learning from Demonstrations (LfD)

首先使用启发式方法生成演示轨迹并预训练策略网络:

```bash
python train_lfd.py
```

这将:
- 使用RandomPruner生成50K演示轨迹
- 每个轨迹包含3个剪枝步骤
- 使用BCE损失训练策略网络模仿演示
- 保存checkpoint到 `logs/lfd_checkpoint_epoch5.pt`

### 2. PPO训练

使用PPO微调策略网络:

```bash
# 首先在config.py中设置LfD checkpoint路径
LFD_CHECKPOINT_PATH = "logs/lfd_checkpoint_epoch5.pt"

# 运行PPO训练
python train_ppo.py
```

这将:
- 加载LfD预训练的策略网络
- 收集rollout轨迹
- 使用PPO更新策略和价值网络
- 每10个epoch保存checkpoint

### 3. 评估

使用训练好的RL pruner进行评估:

```python
# 在main.py中修改pruner设置
from trainer.trainer import setup_rl_pruner

# 加载训练好的checkpoint
pruner = setup_rl_pruner(config, mllm)
checkpoint = torch.load('logs/ppo_checkpoint_epoch200.pt')
pruner.model.load_state_dict(checkpoint['model_state_dict'])

# 运行评估
python main.py
```

## 核心组件说明

### 1. RL Networks ([rl_networks.py](model/rl_networks.py))

- **SharedAttentionModule**: 共享的多头自注意力模块
- **PolicyNetwork**: 输出每个token的保留概率
- **ValueNetwork**: 估计状态价值
- **RLPruningAgent**: 完整的RL agent

### 2. RL Pruner ([rl_pruner.py](pruner/rl_pruner.py))

- `calculate_pruning_scores()`: 计算保留概率
- `prune_tokens_deterministic()`: 基于阈值的确定性剪枝(推理时使用)
- `prune_tokens()`: 基于top-k的剪枝(评估时使用)

### 3. 奖励函数

```python
reward = alpha * task_reward + beta * efficiency_reward

# 任务奖励: 性能变化
task_reward = Score(H_t) - Score(H_{t-1})

# 效率奖励: 压缩率
efficiency_reward = 1 - (K_t / K_{t-1})
```

### 4. PPO更新

使用GAE计算优势函数:
```python
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
```

PPO目标函数:
```python
L = -L_CLIP + c1 * L_VF - c2 * L_S
```

## 推理模式

训练完成后,有两种推理模式:

### 1. 阈值模式 (Threshold-based)
```python
pruner.forward(visual_features, query_embeddings, use_threshold=True)
# 保留概率 > threshold 的token
```

### 2. Top-k模式
```python
pruner.forward(visual_features, query_embeddings, target_ratio=0.5)
# 保留top 50%的token
```

## 预期结果

根据论文,TPRL应该能够:
- 移除最多66.7%的视觉token
- 减少54.2%的FLOPs
- 平均准确率下降仅0.7%

## 注意事项

1. **内存需求**: PPO训练需要大量GPU内存,建议使用多GPU
2. **训练时间**: 完整训练(LfD + PPO)可能需要数天
3. **数据集**: 当前使用ScreenSpot-Pro,可以修改为其他数据集
4. **Autoencoder**: 当前版本未实现autoencoder,直接使用原始视觉token

## 故障排查

### OOM (Out of Memory)
- 减少 `PPO_ROLLOUT_BATCH_SIZE`
- 减少 `PPO_MINI_BATCH_SIZE`
- 减少 `PPO_MAX_STEPS`

### 训练不稳定
- 降低学习率
- 增加 `PPO_CLIP_EPSILON`
- 调整奖励权重 `PPO_REWARD_ALPHA` 和 `PPO_REWARD_BETA`

### 性能不佳
- 增加LfD训练epoch
- 增加演示轨迹数量
- 调整 `RL_STEP_DISCOUNT`

## 下一步

- [ ] 实现autoencoder用于状态压缩
- [ ] 支持更多数据集和任务
- [ ] 添加更多评估指标
- [ ] 优化训练效率
