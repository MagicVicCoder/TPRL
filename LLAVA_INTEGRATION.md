# LLaVA模型集成测试说明

## ✓ 代码验证结果

所有静态代码检查已通过！LLaVA-1.5-7B和13B模型的实现格式正确。

## 实现概览

### 剪枝位置
```
图像输入 → ViT (vision_tower) → Projector (multi_modal_projector)
    → [剪枝位置] → 与文本嵌入拼接 → LLM → 生成答案
```

### 关键组件

1. **视觉特征提取** ([llava_mllm.py:38-43](model/llava_mllm.py))
   - 通过 `vision_tower` 提取ViT特征
   - 通过 `multi_modal_projector` 投影到LLM空间

2. **剪枝操作** (在evaluator中)
   - 对投影后的视觉特征进行token剪枝
   - 保留重要的视觉token，减少计算量

3. **答案生成** ([llava_mllm.py:76-84](model/llava_mllm.py))
   - 使用 `language_model.generate()` 生成回答

## 使用方法

### 方法1: 修改配置文件运行

编辑 [config.py](config.py):

```python
# 测试 LLaVA-1.5-7B
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# 或测试 LLaVA-1.5-13B
MODEL_ID = "llava-hf/llava-1.5-13b-hf"
```

然后运行:
```bash
python main.py
```

### 方法2: 运行专门的测试脚本

```bash
# 测试LLaVA-7B和13B的完整流程
python test_llava.py
```

这个脚本会：
1. 加载模型
2. 创建测试图像和问题
3. 提取视觉和文本特征
4. 执行剪枝操作
5. 生成答案
6. 验证整个流程

## 验证清单

### ✓ 已验证的内容

- [x] 文件结构完整
- [x] LLaVA类正确继承BaseMLLM
- [x] 所有必需方法已实现
- [x] 视觉特征提取路径正确
- [x] Projector调用正确
- [x] 文本嵌入提取正确
- [x] 图像token定位正确
- [x] 返回值格式与Qwen一致
- [x] 工厂函数正确识别LLaVA模型
- [x] 配置文件包含模型选项

### 待实际运行验证

- [ ] 模型能否成功加载
- [ ] 特征提取是否正常
- [ ] 剪枝是否按预期工作
- [ ] 生成的答案是否合理
- [ ] 7B和13B模型是否都能正常工作

## 预期输出格式

### get_components_for_env 返回:
```python
{
    "original_visual_features": torch.Tensor,  # [1, num_patches, hidden_dim]
    "text_embeds_part1": torch.Tensor,         # [1, len1, hidden_dim]
    "text_embeds_part2": torch.Tensor,         # [1, len2, hidden_dim]
    "query_embeddings": torch.Tensor,          # [1, 1, hidden_dim]
    "current_num_patches": int                 # 原始patch数量
}
```

### 剪枝后:
```python
pruned_visual_features: torch.Tensor  # [1, num_patches * target_ratio, hidden_dim]
```

### generate_answer 返回:
```python
str  # 生成的文本答案
```

## 注意事项

1. **内存需求**
   - LLaVA-7B: 约14GB GPU内存 (fp16)
   - LLaVA-13B: 约26GB GPU内存 (fp16)

2. **模型下载**
   - 首次运行会从HuggingFace下载模型
   - 7B模型约13GB，13B模型约25GB

3. **依赖要求**
   ```bash
   pip install transformers torch pillow
   ```

## 故障排查

如果遇到问题，请检查：

1. **AttributeError: 'LlavaForConditionalGeneration' has no attribute 'xxx'**
   - 检查transformers版本: `pip install transformers>=4.37.0`

2. **CUDA out of memory**
   - 减少batch size
   - 使用更小的模型 (7B而不是13B)
   - 增加剪枝比例 (减少保留的token数量)

3. **模型加载失败**
   - 检查网络连接
   - 确认HF_HOME路径可写
   - 尝试手动下载模型

## 下一步

运行测试验证实际效果:
```bash
python test_llava.py
```

或直接在你的任务上测试:
```bash
# 修改config.py中的MODEL_ID
python main.py
```
