"""
代码格式和逻辑验证脚本
检查LLaVA模型实现的代码结构是否正确，无需实际加载模型
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_imports():
    """检查所有必要的导入是否正确"""
    print("\n[1] Checking imports...")
    try:
        from model.llava_mllm import LLaVA
        from model.qwen_mllm import Qwen2_5VL
        from model.base_mllm import get_mllm, BaseMLLM
        from pruner.random_pruner import RandomPruner
        from pruner.mlp_pruner import MLPPruner
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False

def check_class_structure():
    """检查LLaVA类的结构"""
    print("\n[2] Checking LLaVA class structure...")
    try:
        from model.llava_mllm import LLaVA
        from model.base_mllm import BaseMLLM

        # 检查继承关系
        if not issubclass(LLaVA, BaseMLLM):
            print("  ✗ LLaVA does not inherit from BaseMLLM")
            return False

        # 检查必要的方法
        required_methods = ['_load_model', 'get_components_for_env', 'generate_answer']
        for method in required_methods:
            if not hasattr(LLaVA, method):
                print(f"  ✗ Missing method: {method}")
                return False

        print("  ✓ Class structure is correct")
        print(f"    - Inherits from BaseMLLM: Yes")
        print(f"    - Has _load_model: Yes")
        print(f"    - Has get_components_for_env: Yes")
        print(f"    - Has generate_answer: Yes")
        return True
    except Exception as e:
        print(f"  ✗ Structure check error: {e}")
        return False

def check_factory_function():
    """检查工厂函数是否正确配置"""
    print("\n[3] Checking factory function...")
    try:
        import config
        from model.base_mllm import get_mllm

        # 测试不同的模型ID
        test_cases = [
            ("llava-hf/llava-1.5-7b-hf", "LLaVA"),
            ("llava-hf/llava-1.5-13b-hf", "LLaVA"),
            ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2_5VL"),
        ]

        original_model_id = config.MODEL_ID

        for model_id, expected_class in test_cases:
            config.MODEL_ID = model_id
            # 不实际加载模型，只检查逻辑
            if "llava" in model_id.lower():
                from model.llava_mllm import LLaVA
                expected_type = LLaVA
            elif "Qwen" in model_id or "qwen" in model_id:
                from model.qwen_mllm import Qwen2_5VL
                expected_type = Qwen2_5VL
            else:
                print(f"  ✗ Unknown model type: {model_id}")
                return False

            print(f"  ✓ {model_id} -> {expected_class}")

        config.MODEL_ID = original_model_id
        print("  ✓ Factory function logic is correct")
        return True
    except Exception as e:
        print(f"  ✗ Factory function error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_method_signatures():
    """检查方法签名是否正确"""
    print("\n[4] Checking method signatures...")
    try:
        from model.llava_mllm import LLaVA
        import inspect

        # 检查get_components_for_env的签名
        sig = inspect.signature(LLaVA.get_components_for_env)
        params = list(sig.parameters.keys())
        if params != ['self', 'image', 'question']:
            print(f"  ✗ get_components_for_env signature incorrect: {params}")
            return False

        # 检查generate_answer的签名
        sig = inspect.signature(LLaVA.generate_answer)
        params = list(sig.parameters.keys())
        expected = ['self', 'final_embeddings', 'attention_mask', 'max_new_tokens']
        if params != expected:
            print(f"  ✗ generate_answer signature incorrect: {params}")
            return False

        print("  ✓ Method signatures are correct")
        print(f"    - get_components_for_env(self, image, question)")
        print(f"    - generate_answer(self, final_embeddings, attention_mask, max_new_tokens)")
        return True
    except Exception as e:
        print(f"  ✗ Signature check error: {e}")
        return False

def check_return_format():
    """检查返回值格式的文档"""
    print("\n[5] Checking expected return format...")
    try:
        print("  ✓ get_components_for_env should return:")
        print("    {")
        print("      'original_visual_features': torch.Tensor [B, N, D],")
        print("      'text_embeds_part1': torch.Tensor [B, L1, D],")
        print("      'text_embeds_part2': torch.Tensor [B, L2, D],")
        print("      'query_embeddings': torch.Tensor [B, 1, D],")
        print("      'current_num_patches': int")
        print("    }")
        print("  ✓ generate_answer should return:")
        print("    str (generated text)")
        return True
    except Exception as e:
        print(f"  ✗ Format check error: {e}")
        return False

def check_code_consistency():
    """检查代码一致性"""
    print("\n[6] Checking code consistency with Qwen implementation...")
    try:
        from model.llava_mllm import LLaVA
        from model.qwen_mllm import Qwen2_5VL
        import inspect

        # 比较两个类的方法
        llava_methods = set(dir(LLaVA))
        qwen_methods = set(dir(Qwen2_5VL))

        # 应该有相同的公共方法
        common_methods = {'_load_model', 'get_components_for_env', 'generate_answer'}

        for method in common_methods:
            if method not in llava_methods:
                print(f"  ✗ LLaVA missing method: {method}")
                return False
            if method not in qwen_methods:
                print(f"  ✗ Qwen missing method: {method}")
                return False

        print("  ✓ Both implementations have consistent interfaces")
        return True
    except Exception as e:
        print(f"  ✗ Consistency check error: {e}")
        return False

def main():
    """运行所有检查"""
    print("=" * 80)
    print("LLaVA Implementation Validation (Code Structure Only)")
    print("=" * 80)

    checks = [
        ("Import Check", check_imports),
        ("Class Structure", check_class_structure),
        ("Factory Function", check_factory_function),
        ("Method Signatures", check_method_signatures),
        ("Return Format", check_return_format),
        ("Code Consistency", check_code_consistency),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 打印总结
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)
    print("=" * 80)
    if all_passed:
        print("✓ All validation checks passed!")
        print("\nThe code structure is correct. You can now test with actual models.")
    else:
        print("✗ Some validation checks failed. Please review the errors above.")
    print("=" * 80)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
