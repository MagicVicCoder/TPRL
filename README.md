# Language-Guided Token Compression with Reinforcement Learning in Large Vision-Language Models

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A reinforcement-learning-based visual token pruning framework to accelerate inference of Large Vision Language Models (LVLMs).


## 📋 Method Overview


TPRL formulates visual token pruning as a Markov Decision Process (MDP):

1. **Learning from Demonstrations (LfD)**: Generate demonstration trajectories using heuristics and pretrain the policy network.
2. **PPO Fine-tuning**: Fine-tune the policy with Proximal Policy Optimization to jointly optimize task performance and computational efficiency.
3. **Inference**: One-shot pruning that retains the most important visual tokens.

### Architecture

```
visual input → ViT → Projector → [TPRL pruner] → LLM → output
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MagicVicCoder/TPRL.git
cd TPRL

# Install requirements
pip install -r requirements.txt
```

### Training

#### Step 1: Learning from Demonstrations

```bash
python train_lfd.py
```

#### Step 2: PPO Training

```bash
# Set the LfD checkpoint path in config.py first
python train_ppo.py
```

### Evaluation

```bash
python main.py
```

## 📁 Project Structure

```
TPRL/
├── model/
│   ├── autoencoder.py      # Token compression (optional)
│   ├── rl_networks.py      # Policy and value networks
│   ├── llava_mllm.py       # LLaVA model wrapper
│   └── qwen_mllm.py        # Qwen model wrapper
├── pruner/
│   ├── rl_pruner.py        # RL-based pruner
│   ├── random_pruner.py    # Baseline random pruner
│   └── mlp_pruner.py       # MLP-based pruner
├── train_lfd.py            # LfD training script
├── train_ppo.py            # PPO training script
├── config.py               # Configuration
└── main.py                 # Evaluation / inference script
```

## 🎯 Core Idea

### MDP Formulation

* **State**: (visual tokens, text query)
* **Action**: keep / prune decision for each token
* **Reward**: downstream task performance + computational efficiency

### Reward Function

```python
reward = alpha * task_reward + beta * efficiency_reward
```

* `task_reward`: change in task performance (e.g., IoU / accuracy)
* `efficiency_reward`: compression / efficiency metric

## 🛠️ Requirements

* Python >= 3.8
* PyTorch >= 2.0
* Transformers >= 4.37.0
* See `requirements.txt` for full dependency list

---

⭐ If you find this repository useful, please give it a Star!

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@misc{cao2026languageguidedtokencompressionreinforcement,
  title={Language-Guided Token Compression with Reinforcement Learning in Large Vision-Language Models},
  author={Sihan Cao and Jianwei Zhang and Pengcheng Zheng and Jiaxin Yan and Caiyan Qin and Yalan Ye and Wei Dong and Peng Wang and Yang Yang and Chaoning Zhang},
  year={2026},
  eprint={2603.13394},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.13394}
}

