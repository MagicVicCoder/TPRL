# TPRL快速开始指南

## 概述

TPRL是一个基于强化学习的视觉token剪枝框架,用于加速大型视觉语言模型(LVLMs)的推理。

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch transformers pillow tqdm datasets

# 或使用requirements.txt
pip install -r requirements.txt
```

### 2. 配置模型

编辑 `config.py`:

```python
# 选择模型
MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # 或 "llava-hf/llava-1.5-13b-hf"

# RL设置
USE_AUTOENCODER = False  # 当前版本不使用autoencoder
RL_THRESHOLD = 0.5       # 推理时的保留概率阈值
RL_STEP_DISCOUNT = 0.5   # 步骤折扣因子

# LfD设置
LFD_NUM_DEMOS = 50000    # 演示轨迹数量(可以先用较小值如5000测试)
LFD_NUM_EPOCHS = 5       # 训练epoch数
LFD_BATCH_SIZE = 128     # Batch size

# PPO设置
PPO_NUM_EPOCHS = 200     # 训练epoch数(可以先用较小值如20测试)
PPO_ROLLOUT_BATCH_SIZE = 64
PPO_LR_ACTOR = 1e-5
PPO_LR_CRITIC = 3e-5
```

### 3. 训练流程

#### Step 1: Learning from Demonstrations

```bash
python train_lfd.py
```

这将:
- 生成演示轨迹(使用RandomPruner)
- 训练策略网络模仿演示
- 保存checkpoint到 `logs/lfd_checkpoint_epoch5.pt`

**预计时间**: 取决于演示数量,50K演示约需数小时

#### Step 2: PPO Fine-tuning

首先在 `config.py` 中设置LfD checkpoint:

```python
LFD_CHECKPOINT_PATH = "logs/lfd_checkpoint_epoch5.pt"
```

然后运行:

```bash
python train_ppo.py
```

这将:
- 加载LfD预训练的策略
- 收集rollout轨迹
- 使用PPO更新策略
- 每10个epoch保存checkpoint

**预计时间**: 200 epochs约需数天(取决于GPU)

#### Step 3: 评估

修改 `main.py` 使用RL pruner:

```python
from trainer.trainer import setup_rl_pruner
import torch

# 创建pruner
pruner = setup_rl_pruner(config, mllm)

# 加载训练好的checkpoint
checkpoint = torch.load('logs/ppo_checkpoint_epoch200.pt')
pruner.model.load_state_dict(checkpoint['model_state_dict'])

# 运行评估
python main.py
```

## 快速测试(小规模)

如果想快速测试流程,可以使用较小的配置:

```python
# config.py
LFD_NUM_DEMOS = 1000      # 减少到1K
LFD_NUM_EPOCHS = 2        # 减少到2 epochs
PPO_NUM_EPOCHS = 10       # 减少到10 epochs
PPO_ROLLOUT_BATCH_SIZE = 16  # 减少batch size
```

## 常见问题

### Q1: OOM (Out of Memory)

**解决方案**:
```python
# 减少batch size
LFD_BATCH_SIZE = 64  # 从128减少
PPO_ROLLOUT_BATCH_SIZE = 32  # 从64减少
PPO_MINI_BATCH_SIZE = 8  # 从16减少

# 减少模型大小
RL_HIDDEN_DIM = 256  # 从512减少
RL_NUM_LAYERS = 1    # 从2减少
```

### Q2: 训练不稳定

**解决方案**:
```python
# 降低学习率
PPO_LR_ACTOR = 5e-6   # 从1e-5降低
PPO_LR_CRITIC = 1e-5  # 从3e-5降低

# 增加clip范围
PPO_CLIP_EPSILON = 0.3  # 从0.2增加

# 调整奖励权重
PPO_REWARD_ALPHA = 0.5  # 减少任务奖励权重
PPO_REWARD_BETA = 0.2   # 增加效率奖励权重
```

### Q3: 性能不佳

**解决方案**:
1. 增加LfD训练时间
2. 增加演示轨迹质量(使用更好的启发式pruner)
3. 调整 `RL_STEP_DISCOUNT`
4. 增加PPO训练epoch数

### Q4: 训练太慢

**解决方案**:
1. 使用多GPU训练
2. 减少 `PPO_MAX_STEPS`
3. 减少 `PPO_ROLLOUT_BATCH_SIZE`
4. 使用更小的模型(7B而不是13B)

## 监控训练

### LfD训练

查看日志:
```bash
tail -f logs/run_*.log
```

关键指标:
- Loss: 应该逐渐下降
- Accuracy: 应该逐渐上升(目标>0.8)

### PPO训练

关键指标:
- Policy Loss: 应该稳定
- Value Loss: 应该下降
- Avg Reward: 应该上升
- Entropy: 应该保持在合理范围(0.3-0.7)

## 推理使用

训练完成后,有两种推理模式:

### 模式1: 阈值模式(推荐)

```python
# 使用固定阈值
pruned_features = pruner.forward(
    visual_features,
    query_embeddings,
    use_threshold=True
)
```

调整阈值:
```python
# 在config.py中
RL_THRESHOLD = 0.5  # 降低阈值保留更多token,提高阈值剪除更多token
```

### 模式2: Top-k模式

```python
# 保留固定比例的token
pruned_features = pruner.forward(
    visual_features,
    query_embeddings,
    target_ratio=0.5  # 保留50%
)
```

## 性能预期

根据论文,TPRL应该能够:
- 移除最多66.7%的视觉token
- 减少54.2%的FLOPs
- 平均准确率下降仅0.7%

实际效果取决于:
- 训练数据质量
- 超参数设置
- 模型大小
- 任务难度

## 下一步

1. 运行完整训练流程
2. 在不同数据集上评估
3. 调整超参数优化性能
4. 考虑添加autoencoder支持

## 获取帮助

- 查看 [TPRL_README.md](TPRL_README.md) 了解详细信息
- 查看 [TPRL_IMPLEMENTATION_SUMMARY.md](TPRL_IMPLEMENTATION_SUMMARY.md) 了解实现细节
- 运行 `python check_tprl.py` 验证安装

## 引用

如果使用此代码,请引用原始论文:
```
@article{tprl2024,
  title={TPRL: Token Pruning via Reinforcement Learning for Vision-Language Models},
  author={...},
  journal={...},
  year={2024}
}
```
