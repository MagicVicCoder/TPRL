"""
静态代码检查脚本 - 不需要安装依赖
检查LLaVA模型实现的代码格式和结构
"""

import os
import re

def check_file_exists():
    """检查必要的文件是否存在"""
    print("\n[1] Checking file existence...")
    files = [
        "model/llava_mllm.py",
        "model/qwen_mllm.py",
        "model/base_mllm.py",
        "pruner/random_pruner.py",
        "pruner/mlp_pruner.py",
        "config.py"
    ]

    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} NOT FOUND")
            all_exist = False

    return all_exist

def check_llava_class_definition():
    """检查LLaVA类的定义"""
    print("\n[2] Checking LLaVA class definition...")

    with open("model/llava_mllm.py", "r") as f:
        content = f.read()

    checks = {
        "Class definition": r"class LLaVA\(BaseMLLM\):",
        "Import BaseMLLM": r"from \.base_mllm import BaseMLLM",
        "Import transformers": r"from transformers import.*LlavaForConditionalGeneration",
        "_load_model method": r"def _load_model\(self\):",
        "get_components_for_env method": r"def get_components_for_env\(self, image, question\):",
        "generate_answer method": r"def generate_answer\(self, final_embeddings, attention_mask",
    }

    all_passed = True
    for name, pattern in checks.items():
        if re.search(pattern, content):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} NOT FOUND")
            all_passed = False

    return all_passed

def check_llava_implementation_details():
    """检查LLaVA实现的关键细节"""
    print("\n[3] Checking LLaVA implementation details...")

    with open("model/llava_mllm.py", "r") as f:
        content = f.read()

    checks = {
        "Vision tower access": r"self\.model\.vision_tower",
        "Multi-modal projector": r"self\.model\.multi_modal_projector",
        "Language model embeddings": r"self\.model\.language_model\.get_input_embeddings",
        "Language model generate": r"self\.model\.language_model\.generate",
        "Image token index": r"self\.model\.config\.image_token_index",
        "Returns dict with keys": r"return \{",
        "original_visual_features": r"['\"]original_visual_features['\"]",
        "text_embeds_part1": r"['\"]text_embeds_part1['\"]",
        "text_embeds_part2": r"['\"]text_embeds_part2['\"]",
        "query_embeddings": r"['\"]query_embeddings['\"]",
        "current_num_patches": r"['\"]current_num_patches['\"]",
    }

    all_passed = True
    for name, pattern in checks.items():
        if re.search(pattern, content):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} NOT FOUND")
            all_passed = False

    return all_passed

def check_factory_function():
    """检查工厂函数"""
    print("\n[4] Checking factory function in base_mllm.py...")

    with open("model/base_mllm.py", "r") as f:
        content = f.read()

    checks = {
        "get_mllm function": r"def get_mllm\(config\):",
        "LLaVA import": r"from \.llava_mllm import LLaVA",
        "LLaVA condition": r"llava.*in.*config\.MODEL_ID\.lower\(\)",
        "Returns LLaVA": r"return LLaVA\(",
    }

    all_passed = True
    for name, pattern in checks.items():
        if re.search(pattern, content, re.IGNORECASE):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} NOT FOUND")
            all_passed = False

    return all_passed

def check_config_file():
    """检查配置文件"""
    print("\n[5] Checking config.py...")

    with open("config.py", "r") as f:
        content = f.read()

    checks = {
        "MODEL_ID variable": r"MODEL_ID\s*=",
        "LLaVA-7B mentioned": r"llava-1\.5-7b",
        "LLaVA-13B mentioned": r"llava-1\.5-13b",
    }

    all_passed = True
    for name, pattern in checks.items():
        if re.search(pattern, content, re.IGNORECASE):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} NOT FOUND")
            all_passed = False

    return all_passed

def check_code_flow():
    """检查代码流程的完整性"""
    print("\n[6] Checking code flow completeness...")

    with open("model/llava_mllm.py", "r") as f:
        content = f.read()

    # 检查get_components_for_env的流程
    flow_checks = {
        "Process inputs": r"self\.processor\(",
        "Get vision features": r"vision_tower\(pixel_values",
        "Project features": r"multi_modal_projector\(",
        "Get text embeddings": r"get_input_embeddings\(\)\(input_ids\)",
        "Find image tokens": r"torch\.where\(input_ids",
        "Split text embeddings": r"text_embeds_part1.*text_embeds_part2",
        "Create query embeddings": r"\.mean\(dim=1",
    }

    all_passed = True
    for name, pattern in flow_checks.items():
        if re.search(pattern, content):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} NOT FOUND")
            all_passed = False

    return all_passed

def compare_with_qwen():
    """对比LLaVA和Qwen的实现结构"""
    print("\n[7] Comparing LLaVA with Qwen implementation...")

    with open("model/llava_mllm.py", "r") as f:
        llava_content = f.read()

    with open("model/qwen_mllm.py", "r") as f:
        qwen_content = f.read()

    # 检查两者是否有相同的方法
    methods = ["_load_model", "get_components_for_env", "generate_answer"]

    all_passed = True
    for method in methods:
        llava_has = f"def {method}" in llava_content
        qwen_has = f"def {method}" in qwen_content

        if llava_has and qwen_has:
            print(f"  ✓ Both have {method}")
        else:
            print(f"  ✗ Method {method} - LLaVA: {llava_has}, Qwen: {qwen_has}")
            all_passed = False

    # 检查返回值结构是否一致
    return_keys = ["original_visual_features", "text_embeds_part1", "text_embeds_part2",
                   "query_embeddings", "current_num_patches"]

    for key in return_keys:
        llava_has = f'"{key}"' in llava_content or f"'{key}'" in llava_content
        qwen_has = f'"{key}"' in qwen_content or f"'{key}'" in qwen_content

        if llava_has and qwen_has:
            print(f"  ✓ Both return {key}")
        else:
            print(f"  ✗ Return key {key} - LLaVA: {llava_has}, Qwen: {qwen_has}")
            all_passed = False

    return all_passed

def main():
    """运行所有检查"""
    print("=" * 80)
    print("LLaVA Implementation Static Code Validation")
    print("=" * 80)

    # 切换到项目目录
    os.chdir("/data/users/csh/LVLMs_pruner")

    checks = [
        ("File Existence", check_file_exists),
        ("Class Definition", check_llava_class_definition),
        ("Implementation Details", check_llava_implementation_details),
        ("Factory Function", check_factory_function),
        ("Config File", check_config_file),
        ("Code Flow", check_code_flow),
        ("Comparison with Qwen", compare_with_qwen),
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
        print("✓ All static validation checks passed!")
        print("\n下一步:")
        print("1. 确保安装了所需的依赖: pip install -r requirements.txt")
        print("2. 运行实际测试: python test_llava.py")
        print("3. 或者在config.py中修改MODEL_ID并运行main.py")
    else:
        print("✗ Some validation checks failed. Please review the errors above.")
    print("=" * 80)

    return 0 if all_passed else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
