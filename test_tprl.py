"""
快速测试脚本 - 验证TPRL组件是否正常工作
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_rl_networks():
    """测试RL网络组件"""
    print("\n=== Testing RL Networks ===")
    from model.rl_networks import RLPruningAgent

    batch_size = 2
    num_tokens = 10
    d_model = 512

    # 创建agent
    agent = RLPruningAgent(d_model=d_model, nhead=8, num_layers=2, hidden_dim=256)

    # 创建测试输入
    visual_tokens = torch.randn(batch_size, num_tokens, d_model)
    query_embedding = torch.randn(batch_size, 1, d_model)

    # 测试forward
    probs = agent(visual_tokens, query_embedding, return_value=False)
    print(f"✓ Policy output shape: {probs.shape}")
    assert probs.shape == (batch_size, num_tokens)

    # 测试with value
    probs, value = agent(visual_tokens, query_embedding, return_value=True)
    print(f"✓ Policy output shape: {probs.shape}")
    print(f"✓ Value output shape: {value.shape}")
    assert value.shape == (batch_size,)

    # 测试action sampling
    actions, log_probs = agent.sample_action(probs, step_discount=0.5)
    print(f"✓ Actions shape: {actions.shape}")
    print(f"✓ Log probs shape: {log_probs.shape}")

    # 测试entropy
    entropy = agent.get_entropy(probs, step_discount=0.5)
    print(f"✓ Entropy: {entropy.item():.4f}")

    print("✓ All RL network tests passed!")


def test_rl_pruner():
    """测试RL Pruner"""
    print("\n=== Testing RL Pruner ===")

    # 创建mock config
    class MockConfig:
        DEVICE = 'cpu'
        USE_AUTOENCODER = False
        RL_NHEAD = 4
        RL_NUM_LAYERS = 1
        RL_HIDDEN_DIM = 256
        RL_DROPOUT = 0.1
        RL_THRESHOLD = 0.5
        RL_STEP_DISCOUNT = 0.5
        PRUNING_TARGET_RATIO = 0.5

    # 创建mock MLLM
    class MockMLLM:
        device = 'cpu'
        feature_dim = 512

    config = MockConfig()
    mllm = MockMLLM()

    from pruner.rl_pruner import RLPruner

    # 创建pruner
    pruner = RLPruner(mllm, config)
    print(f"✓ RLPruner created")

    # 测试输入
    batch_size = 2
    num_tokens = 20
    hidden_dim = 512

    visual_features = torch.randn(batch_size, num_tokens, hidden_dim)
    query_embeddings = torch.randn(batch_size, 1, hidden_dim)

    # 测试calculate_pruning_scores
    scores = pruner.calculate_pruning_scores(visual_features, query_embeddings)
    print(f"✓ Pruning scores shape: {scores.shape}")
    assert scores.shape == (batch_size, num_tokens)

    # 测试prune_tokens (top-k mode)
    pruned_features = pruner.prune_tokens(visual_features, query_embeddings, target_ratio=0.5)
    print(f"✓ Pruned features shape: {pruned_features.shape}")
    assert pruned_features.shape[1] == int(num_tokens * 0.5)

    # 测试prune_tokens_deterministic (threshold mode)
    pruned_features = pruner.prune_tokens_deterministic(visual_features, query_embeddings, threshold=0.5)
    print(f"✓ Deterministic pruned features shape: {pruned_features.shape}")

    # 测试forward
    pruned_features = pruner.forward(visual_features, query_embeddings, target_ratio=0.5)
    print(f"✓ Forward output shape: {pruned_features.shape}")

    print("✓ All RL pruner tests passed!")


def test_demonstration_generation():
    """测试演示生成逻辑"""
    print("\n=== Testing Demonstration Generation Logic ===")

    # 模拟演示数据结构
    demo = {
        'visual_features': torch.randn(10, 512),
        'query_embedding': torch.randn(1, 512),
        'actions': torch.randint(0, 2, (10,)).float()
    }

    print(f"✓ Demo visual_features shape: {demo['visual_features'].shape}")
    print(f"✓ Demo query_embedding shape: {demo['query_embedding'].shape}")
    print(f"✓ Demo actions shape: {demo['actions'].shape}")
    print(f"✓ Actions: {demo['actions']}")

    # 测试collate function
    from train_lfd import collate_demonstrations

    batch = [demo, demo]
    collated = collate_demonstrations(batch)

    print(f"✓ Collated visual_features shape: {collated['visual_features'].shape}")
    print(f"✓ Collated query_embeddings shape: {collated['query_embeddings'].shape}")
    print(f"✓ Collated actions shape: {collated['actions'].shape}")
    print(f"✓ Collated masks shape: {collated['masks'].shape}")

    print("✓ Demonstration generation logic tests passed!")


def test_ppo_components():
    """测试PPO组件"""
    print("\n=== Testing PPO Components ===")

    from train_ppo import RolloutBuffer, compute_gae

    # 测试RolloutBuffer
    buffer = RolloutBuffer()

    for i in range(5):
        buffer.add(
            state_visual=torch.randn(10, 512),
            state_query=torch.randn(1, 512),
            action=torch.randint(0, 2, (10,)).float(),
            log_prob=torch.randn(10),
            reward=0.5,
            value=0.3,
            done=0.0,
            mask=torch.ones(10)
        )

    print(f"✓ Buffer size: {len(buffer.rewards)}")

    # 测试GAE计算
    rewards = [0.5, 0.3, 0.8, 0.2, 0.6]
    values = [0.4, 0.35, 0.7, 0.25, 0.5]
    dones = [0.0, 0.0, 0.0, 0.0, 1.0]

    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)

    print(f"✓ Advantages: {advantages}")
    print(f"✓ Returns: {returns}")
    assert len(advantages) == len(rewards)
    assert len(returns) == len(rewards)

    print("✓ All PPO component tests passed!")


def main():
    """运行所有测试"""
    print("=" * 80)
    print("TPRL Component Tests")
    print("=" * 80)

    try:
        test_rl_networks()
        test_rl_pruner()
        test_demonstration_generation()
        test_ppo_components()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\n下一步:")
        print("1. 运行 LfD 训练: python train_lfd.py")
        print("2. 运行 PPO 训练: python train_ppo.py")
        print("3. 评估模型: python main.py")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
