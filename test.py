import os
import sys
import config
from data.base_loader import get_data_loader

def test_dataset_loading():
    """测试数据集加载，只处理前30条数据"""
    print("=" * 60)
    print("Dataset Loading Test (First 30 samples)")
    print("=" * 60)
    
    # 临时修改配置，限制处理数量
    original_split = config.DATASET_SPLIT
    # 如果split是"train"，改为只取前30条
    if "train" in config.DATASET_SPLIT.lower():
        config.DATASET_SPLIT = "train[:30]"
    elif ":" in config.DATASET_SPLIT:
        # 如果已经有切片，修改为只取30条
        base_split = config.DATASET_SPLIT.split(":")[0]
        config.DATASET_SPLIT = f"{base_split}[:30]"
    else:
        config.DATASET_SPLIT = "train[:30]"
    
    try:
        # 先检查数据集的splits和features（不加载数据）
        print(f"\n1. Inspecting dataset structure: {config.DATASET_NAME}")
        print("-" * 60)
        try:
            from datasets import load_dataset
            print("   Loading full dataset to check available splits...")
            full_dataset = load_dataset(config.DATASET_NAME, download_mode="reuse_cache_if_exists")
            print(f"   Available splits: {list(full_dataset.keys())}")
            
            # 检查每个split的features
            for split_name in full_dataset.keys():
                split_data = full_dataset[split_name]
                print(f"\n   Split '{split_name}':")
                if hasattr(split_data, 'features'):
                    if hasattr(split_data.features, 'keys'):
                        print(f"     Features: {list(split_data.features.keys())}")
                    else:
                        print(f"     Features: {split_data.features}")
                if len(split_data) > 0:
                    first_sample = split_data[0]
                    print(f"     First sample keys: {list(first_sample.keys())}")
                    # 显示第一个样本的字段详情
                    print(f"     Sample count: {len(split_data)}")
        except Exception as e:
            print(f"   Warning: Could not inspect dataset structure: {e}")
        
        print("\n" + "-" * 60)
        print(f"\n2. Loading dataset with limited samples")
        print(f"   Original split: {original_split}")
        print(f"   Test split (limited): {config.DATASET_SPLIT}")
        print(f"   Train/Test ratio: {config.TRAIN_TEST_SPLIT_RATIO}")
        
        data_loader = get_data_loader(config)
        
        # 获取测试样本
        test_samples = data_loader.get_test_samples()
        train_samples = data_loader.get_train_samples()
        
        print(f"\n3. Dataset loaded successfully!")
        print(f"   Total train samples: {len(train_samples)}")
        print(f"   Total test samples: {len(test_samples)}")
        
        # 只测试前30条测试样本（如果已经限制到30条，这里就是全部）
        test_subset = test_samples[:30]
        actual_count = len(test_subset)
        print(f"\n4. Testing first {actual_count} test samples:")
        print("=" * 60)
        
        success_count = 0
        error_count = 0
        sample_details = []
        
        for idx, sample in enumerate(test_subset):
            try:
                # 检查样本结构
                has_image = 'image' in sample and sample['image'] is not None
                has_question = 'question' in sample and sample['question'] is not None
                has_bbox = 'bbox' in sample and sample['bbox'] is not None
                has_answer = 'answer' in sample and sample.get('answer') is not None
                
                # 获取详细信息
                image_type = type(sample.get('image')).__name__ if has_image else "None"
                question_preview = sample.get('question', '')[:50] if has_question else "None"
                bbox_value = sample.get('bbox') if has_bbox else None
                
                is_valid = has_image and has_question and has_bbox
                
                if is_valid:
                    success_count += 1
                else:
                    error_count += 1
                
                sample_details.append({
                    'idx': idx,
                    'valid': is_valid,
                    'has_image': has_image,
                    'has_question': has_question,
                    'has_bbox': has_bbox,
                    'has_answer': has_answer,
                    'image_type': image_type,
                    'question': question_preview,
                    'bbox': bbox_value
                })
                
                # 每10条输出一次进度
                if (idx + 1) % 10 == 0:
                    print(f"   Processed {idx + 1}/{actual_count} samples...")
                    
            except Exception as e:
                error_count += 1
                print(f"   Error processing sample {idx}: {e}")
                sample_details.append({
                    'idx': idx,
                    'valid': False,
                    'error': str(e)
                })
        
        # 输出详细结果
        print("\n" + "=" * 60)
        print("5. Detailed Results:")
        print("=" * 60)
        print(f"   Valid samples: {success_count}/{actual_count}")
        print(f"   Invalid samples: {error_count}/{actual_count}")
        if actual_count > 0:
            print(f"   Success rate: {success_count/actual_count*100:.1f}%")
        
        # 显示前5个样本的详细信息
        print("\n6. First 5 samples details:")
        print("-" * 60)
        for detail in sample_details[:5]:
            print(f"\n   Sample {detail['idx']}:")
            print(f"     Valid: {detail['valid']}")
            print(f"     Has image: {detail['has_image']} ({detail.get('image_type', 'N/A')})")
            print(f"     Has question: {detail['has_question']}")
            if detail['has_question']:
                print(f"       Question: {detail['question']}...")
            print(f"     Has bbox: {detail['has_bbox']}")
            if detail['has_bbox']:
                print(f"       Bbox: {detail['bbox']}")
            print(f"     Has answer: {detail.get('has_answer', False)}")
            if 'error' in detail:
                print(f"     Error: {detail['error']}")
        
        # 如果有无效样本，显示问题统计
        invalid_samples = [d for d in sample_details if not d.get('valid', False)]
        if invalid_samples:
            print("\n7. Invalid samples analysis:")
            print("-" * 60)
            missing_image = sum(1 for d in invalid_samples if not d.get('has_image', True))
            missing_question = sum(1 for d in invalid_samples if not d.get('has_question', True))
            missing_bbox = sum(1 for d in invalid_samples if not d.get('has_bbox', True))
            print(f"   Missing image: {missing_image}")
            print(f"   Missing question: {missing_question}")
            print(f"   Missing bbox: {missing_bbox}")
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        
        return success_count > 0
        
    except Exception as e:
        print(f"\n❌ Error during dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    sys.exit(0 if success else 1)

