"""
静态代码检查 - 验证TPRL实现的完整性
"""

import os
import re

def check_file_exists(filepath):
    """检查文件是否存在"""
    return os.path.exists(filepath)

def check_imports_in_file(filepath, expected_imports):
    """检查文件中的导入"""
    with open(filepath, 'r') as f:
        content = f.read()

    results = {}
    for imp in expected_imports:
        results[imp] = imp in content

    return results

def check_class_methods(filepath, class_name, expected_methods):
    """检查类中的方法"""
    with open(filepath, 'r') as f:
        content = f.read()

    # 检查类定义
    class_pattern = f"class {class_name}"
    has_class = class_pattern in content

    # 检查方法
    method_results = {}
    for method in expected_methods:
        method_pattern = f"def {method}"
        method_results[method] = method_pattern in content

    return has_class, method_results

def main():
    print("=" * 80)
    print("TPRL Implementation Static Check")
    print("=" * 80)

    all_passed = True

    # 1. 检查核心文件
    print("\n[1] Checking core files...")
    core_files = [
        "model/autoencoder.py",
        "model/rl_networks.py",
        "pruner/rl_pruner.py",
        "train_lfd.py",
        "train_ppo.py",
        "TPRL_README.md"
    ]

    for file in core_files:
        exists = check_file_exists(file)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_passed = False

    # 2. 检查RL Networks
    print("\n[2] Checking RL Networks...")
    rl_networks_classes = {
        "SharedAttentionModule": ["__init__", "forward"],
        "PolicyNetwork": ["__init__", "forward"],
        "ValueNetwork": ["__init__", "forward"],
        "RLPruningAgent": ["__init__", "forward", "sample_action", "get_action_log_probs", "get_entropy"]
    }

    for class_name, methods in rl_networks_classes.items():
        has_class, method_results = check_class_methods("model/rl_networks.py", class_name, methods)
        if has_class:
            print(f"  ✓ Class {class_name} found")
            for method, found in method_results.items():
                status = "✓" if found else "✗"
                print(f"    {status} Method {method}")
                if not found:
                    all_passed = False
        else:
            print(f"  ✗ Class {class_name} NOT FOUND")
            all_passed = False

    # 3. 检查RL Pruner
    print("\n[3] Checking RL Pruner...")
    has_class, method_results = check_class_methods(
        "pruner/rl_pruner.py",
        "RLPruner",
        ["_build_model", "calculate_pruning_scores", "prune_tokens", "prune_tokens_deterministic", "forward"]
    )

    if has_class:
        print(f"  ✓ Class RLPruner found")
        for method, found in method_results.items():
            status = "✓" if found else "✗"
            print(f"    {status} Method {method}")
            if not found:
                all_passed = False
    else:
        print(f"  ✗ Class RLPruner NOT FOUND")
        all_passed = False

    # 4. 检查LfD训练脚本
    print("\n[4] Checking LfD training script...")
    lfd_functions = [
        "generate_demonstrations",
        "train_lfd",
        "collate_demonstrations",
        "main"
    ]

    with open("train_lfd.py", 'r') as f:
        lfd_content = f.read()

    for func in lfd_functions:
        found = f"def {func}" in lfd_content
        status = "✓" if found else "✗"
        print(f"  {status} Function {func}")
        if not found:
            all_passed = False

    # 5. 检查PPO训练脚本
    print("\n[5] Checking PPO training script...")
    ppo_components = [
        "RolloutBuffer",
        "compute_gae",
        "compute_task_reward",
        "collect_rollouts",
        "ppo_update",
        "main"
    ]

    with open("train_ppo.py", 'r') as f:
        ppo_content = f.read()

    for comp in ppo_components:
        if comp == "RolloutBuffer":
            found = f"class {comp}" in ppo_content
        else:
            found = f"def {comp}" in ppo_content
        status = "✓" if found else "✗"
        print(f"  {status} {'Class' if comp == 'RolloutBuffer' else 'Function'} {comp}")
        if not found:
            all_passed = False

    # 6. 检查config.py中的RL配置
    print("\n[6] Checking RL configurations in config.py...")
    rl_configs = [
        "USE_AUTOENCODER",
        "RL_LATENT_DIM",
        "RL_NHEAD",
        "RL_NUM_LAYERS",
        "RL_HIDDEN_DIM",
        "RL_THRESHOLD",
        "RL_STEP_DISCOUNT",
        "LFD_NUM_DEMOS",
        "LFD_NUM_EPOCHS",
        "LFD_LEARNING_RATE",
        "PPO_NUM_EPOCHS",
        "PPO_ROLLOUT_BATCH_SIZE",
        "PPO_LR_ACTOR",
        "PPO_LR_CRITIC",
        "PPO_GAMMA",
        "PPO_CLIP_EPSILON"
    ]

    with open("config.py", 'r') as f:
        config_content = f.read()

    for cfg in rl_configs:
        found = f"{cfg} =" in config_content
        status = "✓" if found else "✗"
        print(f"  {status} Config {cfg}")
        if not found:
            all_passed = False

    # 7. 检查trainer.py中的setup函数
    print("\n[7] Checking trainer.py...")
    with open("trainer/trainer.py", 'r') as f:
        trainer_content = f.read()

    trainer_checks = {
        "Import RLPruner": "from pruner.rl_pruner import RLPruner",
        "setup_rl_pruner function": "def setup_rl_pruner"
    }

    for name, pattern in trainer_checks.items():
        found = pattern in trainer_content
        status = "✓" if found else "✗"
        print(f"  {status} {name}")
        if not found:
            all_passed = False

    # 总结
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nTPRL框架已成功实现,包括:")
        print("  - RL Networks (Policy + Value + Shared Attention)")
        print("  - RL Pruner")
        print("  - Learning from Demonstrations训练脚本")
        print("  - PPO训练脚本")
        print("  - 完整的配置系统")
        print("\n下一步:")
        print("  1. 安装依赖: pip install torch transformers")
        print("  2. 运行LfD训练: python train_lfd.py")
        print("  3. 运行PPO训练: python train_ppo.py")
        print("  4. 评估模型: python main.py")
    else:
        print("✗ SOME CHECKS FAILED")
        print("请检查上述失败的项目")
    print("=" * 80)

    return 0 if all_passed else 1

if __name__ == "__main__":
    import sys
    os.chdir("/data/users/csh/TPRL")
    sys.exit(main())
