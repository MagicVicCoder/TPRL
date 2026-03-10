import os
import sys
import torch
from PIL import Image
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model.base_mllm import get_mllm
from pruner.random_pruner import RandomPruner

def test_llava_model(model_id):
    """
    测试LLaVA模型的完整流程：加载 -> 特征提取 -> 剪枝 -> 推理
    """
    print("=" * 80)
    print(f"Testing Model: {model_id}")
    print("=" * 80)

    # 1. 临时修改配置
    original_model_id = config.MODEL_ID
    config.MODEL_ID = model_id

    try:
        # 2. 加载模型
        print("\n[Step 1] Loading model...")
        mllm = get_mllm(config)
        print(f"✓ Model loaded successfully")
        print(f"  - Model type: {type(mllm).__name__}")
        print(f"  - Device: {mllm.device}")
        print(f"  - Feature dim: {mllm.feature_dim}")

        # 3. 创建测试图像和问题
        print("\n[Step 2] Creating test image and question...")
        # 创建一个简单的测试图像 (224x224 RGB)
        test_image = Image.new('RGB', (224, 224), color=(73, 109, 137))
        test_question = "What is in this image?"
        print(f"✓ Test data created")
        print(f"  - Image size: {test_image.size}")
        print(f"  - Question: {test_question}")

        # 4. 提取组件
        print("\n[Step 3] Extracting components...")
        components = mllm.get_components_for_env(test_image, test_question)

        if components is None:
            print("✗ Failed to extract components")
            return False

        print(f"✓ Components extracted successfully")
        print(f"  - Visual features shape: {components['original_visual_features'].shape}")
        print(f"  - Text part1 shape: {components['text_embeds_part1'].shape}")
        print(f"  - Text part2 shape: {components['text_embeds_part2'].shape}")
        print(f"  - Query embeddings shape: {components['query_embeddings'].shape}")
        print(f"  - Num patches: {components['current_num_patches']}")

        # 5. 初始化剪枝器
        print("\n[Step 4] Initializing pruner...")
        pruner = RandomPruner(mllm, config)
        print(f"✓ Pruner initialized")

        # 6. 执行剪枝
        print("\n[Step 5] Performing pruning...")
        original_visual_features = components["original_visual_features"]
        query_embeddings = components["query_embeddings"]
        target_ratio = config.PRUNING_TARGET_RATIO

        pruned_visual_features = pruner.forward(
            original_visual_features,
            query_embeddings,
            target_ratio
        )

        print(f"✓ Pruning completed")
        print(f"  - Original patches: {original_visual_features.shape[1]}")
        print(f"  - Pruned patches: {pruned_visual_features.shape[1]}")
        print(f"  - Compression ratio: {pruned_visual_features.shape[1] / original_visual_features.shape[1]:.2f}")

        # 7. 组合嵌入
        print("\n[Step 6] Combining embeddings...")
        combined_embeddings = torch.cat([
            components["text_embeds_part1"],
            pruned_visual_features,
            components["text_embeds_part2"]
        ], dim=1)
        attention_mask = torch.ones(
            (1, combined_embeddings.shape[1]),
            dtype=torch.long,
            device=mllm.device
        )
        print(f"✓ Embeddings combined")
        print(f"  - Combined shape: {combined_embeddings.shape}")
        print(f"  - Attention mask shape: {attention_mask.shape}")

        # 8. 生成答案
        print("\n[Step 7] Generating answer...")
        generated_answer = mllm.generate_answer(
            combined_embeddings,
            attention_mask,
            max_new_tokens=20
        )
        print(f"✓ Answer generated")
        print(f"  - Generated text: '{generated_answer}'")

        print("\n" + "=" * 80)
        print(f"✓ ALL TESTS PASSED for {model_id}")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 恢复原始配置
        config.MODEL_ID = original_model_id
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """
    主测试函数
    """
    print("\n" + "=" * 80)
    print("LLaVA Model Integration Test")
    print("=" * 80)

    # 测试的模型列表
    models_to_test = [
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/llava-1.5-13b-hf"
    ]

    results = {}

    for model_id in models_to_test:
        success = test_llava_model(model_id)
        results[model_id] = success
        print("\n")

    # 打印总结
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for model_id, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {model_id}")
    print("=" * 80)

    # 返回退出码
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
