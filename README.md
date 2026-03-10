# TPRL: Token Pruning via Reinforcement Learning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

基于强化学习的视觉token剪枝框架，用于加速大型视觉语言模型(LVLMs)的推理。

## 🌟 特性

- ✅ **强化学习优化**: 使用PPO算法学习自适应剪枝策略
- ✅ **任务感知**: 直接优化下游任务性能
- ✅ **多步剪枝**: 渐进式token剪枝，学习最优轨迹
- ✅ **高效推理**: 移除最多66.7%的视觉token，减少54.2%的FLOPs
- ✅ **支持多模型**: LLaVA-1.5-7B/13B, Qwen2.5-VL等

## 📋 方法概述

TPRL将视觉token剪枝建模为马尔可夫决策过程(MDP):

1. **Learning from Demonstrations (LfD)**: 使用启发式方法生成演示轨迹，预训练策略网络
2. **PPO Fine-tuning**: 使用Proximal Policy Optimization微调策略，优化任务性能和计算效率
3. **Inference**: 一次性剪枝，保留最重要的视觉token

### 架构

```
视觉输入 → ViT → Projector → [TPRL剪枝] → LLM → 输出
```

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/MagicVicCoder/TPRL.git
cd TPRL

# 安装依赖
pip install -r requirements.txt
```

### 训练

#### Step 1: Learning from Demonstrations

```bash
python train_lfd.py
```

#### Step 2: PPO Training

```bash
# 在config.py中设置LfD checkpoint路径
python train_ppo.py
```

### 评估

```bash
python main.py
```



## 📁 项目结构

```
TPRL/
├── model/
│   ├── autoencoder.py      # Token压缩(可选)
│   ├── rl_networks.py      # Policy和Value网络
│   ├── llava_mllm.py       # LLaVA模型
│   └── qwen_mllm.py        # Qwen模型
├── pruner/
│   ├── rl_pruner.py        # RL-based pruner
│   ├── random_pruner.py    # Baseline
│   └── mlp_pruner.py       # MLP pruner
├── train_lfd.py            # LfD训练
├── train_ppo.py            # PPO训练
├── config.py               # 配置文件
└── main.py                 # 评估脚本
```



## 🎯 核心思想

### MDP Formulation

- **State**: (视觉token, 文本query)
- **Action**: 每个token的保留/剪枝决策
- **Reward**: 任务性能 + 计算效率

### 奖励函数

```python
reward = α × task_reward + β × efficiency_reward
```

- `task_reward`: 任务性能变化(IoU/准确率)
- `efficiency_reward`: 压缩率

## 🛠️ 依赖

- Python >= 3.8
- PyTorch >= 2.0
- Transformers >= 4.37.0
- 其他依赖见 `requirements.txt`



---

⭐ 如果觉得有用，请给个Star!
