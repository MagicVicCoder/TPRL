import random
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from datasets import load_dataset
from PIL import Image

from .base_loader import BaseDataLoader


class ScreenProDataLoader(BaseDataLoader):
    """
    Data loader for the ScreenSpot-Pro dataset (Voxel51/ScreenSpot-Pro).

    It normalizes each sample into the common:
        {'image': PIL.Image, 'question': str, 'bbox': (x, y, w, h), 'answer': Optional[str]}
    format that the rest of the pipeline expects.
    """

    _QUESTION_KEYS = ("instruction", "question", "prompt", "query", "caption")
    _ANSWER_KEYS = ("answer", "response", "label", "target_description", "description")
    _IMAGE_PATH_KEYS = ("image_path", "screenshot_path")
    _BBOX_KEYS = (
        "bbox", "target_bbox", "bbox_xywh", "bounding_box", "boundingbox",
        "box", "target_box", "coordinates", "coords", "rect", "rectangle",
        "region", "area", "location", "target_location"
    )

    def _load_and_split_data(self):
        print(f"Loading dataset: {self.name}")
        
        # 先检查缓存状态
        cache_info = self._check_cache_status()
        print(f"Cache directory: {cache_info['cache_path']}")
        if cache_info['exists']:
            print(f"Cache exists: Yes (but datasets library will verify integrity)")
        else:
            print(f"Cache exists: No - will download from scratch")
        
        max_retries = 2
        dataset = None
        for attempt in range(max_retries):
            try:
                # 第一次尝试：使用默认模式（重用缓存）
                download_mode = "reuse_cache_if_exists" if attempt == 0 else "force_redownload"
                if attempt == 0:
                    print(f"Attempt {attempt + 1}: Using download_mode='{download_mode}'")
                else:
                    print(f"Attempt {attempt + 1}: Using download_mode='{download_mode}' (force redownload)")
                
                # 先加载整个数据集（不指定split）来查看可用的splits
                if attempt == 0:
                    print("Checking available splits and features...")
                    try:
                        full_dataset = load_dataset(self.name, download_mode=download_mode)
                        print(f"Available splits: {list(full_dataset.keys())}")
                        # 检查每个split的features
                        for split_name in full_dataset.keys():
                            split_data = full_dataset[split_name]
                            print(f"\nSplit '{split_name}' features: {split_data.features if hasattr(split_data, 'features') else 'N/A'}")
                            if hasattr(split_data, 'features') and hasattr(split_data.features, 'keys'):
                                print(f"  Feature names: {list(split_data.features.keys())}")
                            # 检查第一个样本
                            if len(split_data) > 0:
                                first_sample = split_data[0]
                                print(f"  First sample keys: {list(first_sample.keys())}")
                    except Exception as e:
                        print(f"Warning: Could not inspect full dataset: {e}")
                
                # 加载指定的split
                dataset = load_dataset(self.name, split=self.split, download_mode=download_mode)
                print("Dataset loaded successfully!")
                break  # 成功加载，退出循环
            except (FileNotFoundError, OSError) as exc:
                error_msg = str(exc)
                # 检查是否是缓存文件损坏的错误
                if "arrow" in error_msg.lower() or "cache" in error_msg.lower() or "no such file" in error_msg.lower():
                    print(f"Attempt {attempt + 1}/{max_retries}: Detected corrupted cache. Cleaning up...")
                    self._cleanup_cache()
                    if attempt < max_retries - 1:
                        print("Retrying with force_redownload...")
                        continue
                    else:
                        print("Failed after retries. Trying force_redownload one more time...")
                        try:
                            dataset = load_dataset(self.name, split=self.split, download_mode="force_redownload")
                            break
                        except Exception as final_exc:
                            print(f"Final attempt failed: {final_exc}")
                            raise RuntimeError(
                                f"Failed to load dataset {self.name} after {max_retries} attempts. "
                                f"Please manually delete the cache directory and try again. "
                                f"Last error: {final_exc}"
                            ) from final_exc
                else:
                    # 其他类型的错误，直接抛出
                    raise
            except Exception as exc:
                print(f"Failed to load dataset {self.name}. Error: {exc}")
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed. Retrying...")
                    self._cleanup_cache()
                    continue
                raise

        # 转换为列表，避免迭代器只能遍历一次的问题
        # 先检查数据集的 features/schema
        print("\n" + "=" * 80)
        print("=== DATASET FEATURES/SCHEMA ===")
        print("=" * 80)
        if hasattr(dataset, 'features'):
            print(f"Dataset features: {dataset.features}")
            print(f"Feature names: {list(dataset.features.keys()) if hasattr(dataset.features, 'keys') else 'N/A'}")
        if hasattr(dataset, 'info'):
            print(f"Dataset info: {dataset.info}")
        print("=" * 80)
        
        print("Converting dataset to list...")
        all_samples = list(dataset)
        total_count = len(all_samples)
        print(f"Total samples in dataset: {total_count}")
        
        # 先详细检查前3个样本的完整结构
        print("\n" + "=" * 80)
        print("=== RAW SAMPLE STRUCTURE INSPECTION ===")
        print("=" * 80)
        for i in range(min(3, total_count)):
            raw_sample = all_samples[i]
            print(f"\n{'='*80}")
            print(f"SAMPLE {i} - Complete Structure:")
            print(f"{'='*80}")
            print(f"All keys: {list(raw_sample.keys())}")
            print(f"\nDetailed field inspection:")
            for key in sorted(raw_sample.keys()):
                value = raw_sample[key]
                value_type = type(value).__name__
                
                # 打印字段信息
                if isinstance(value, (list, tuple)):
                    print(f"  {key}: {value_type}[{len(value)}]")
                    if len(value) > 0:
                        print(f"    First element: {value[0]}")
                        if len(value) > 1:
                            print(f"    Second element: {value[1]}")
                        if len(value) > 2:
                            print(f"    ... (total {len(value)} elements)")
                elif isinstance(value, dict):
                    print(f"  {key}: {value_type} with keys: {list(value.keys())}")
                    # 打印字典的前几个键值对
                    for k, v in list(value.items())[:5]:
                        print(f"    {k}: {type(v).__name__} = {v}")
                    if len(value) > 5:
                        print(f"    ... (total {len(value)} keys)")
                elif isinstance(value, str):
                    print(f"  {key}: {value_type} (length {len(value)})")
                    print(f"    Content: {value[:200]}{'...' if len(value) > 200 else ''}")
                elif hasattr(value, 'shape'):  # numpy array or tensor
                    print(f"  {key}: {value_type} with shape {value.shape}")
                else:
                    print(f"  {key}: {value_type} = {value}")
        
        print("\n" + "=" * 80)
        print("=== END OF RAW SAMPLE INSPECTION ===")
        print("=" * 80)
        print("\nNow processing samples for normalization...")
        
        normalized_samples = []
        skipped_count = 0
        skip_reasons = {"no_image": 0, "no_bbox": 0, "no_question": 0}
        
        print("\n=== Processing all samples ===")
        for idx, raw_sample in enumerate(all_samples):
            normalized = self._normalize_sample(raw_sample, idx, skip_reasons)
            if normalized is not None:
                normalized_samples.append(normalized)
            else:
                skipped_count += 1
                
            # 每处理100个样本输出一次进度
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total_count} samples: {len(normalized_samples)} valid, {skipped_count} skipped")

        print(f"\n=== Normalization Summary ===")
        print(f"Total samples processed: {total_count}")
        print(f"Valid samples: {len(normalized_samples)}")
        print(f"Skipped samples: {skipped_count}")
        print(f"Skip reasons: {skip_reasons}")

        if not normalized_samples:
            raise RuntimeError(
                f"ScreenSpot-Pro dataset did not yield any usable samples after normalization. "
                f"Skip reasons: {skip_reasons}. "
                f"Please check the dataset structure - expected fields: image, instruction/question, bbox"
            )

        random.shuffle(normalized_samples)
        split_idx = int(self.split_ratio * len(normalized_samples))
        self.train_samples = normalized_samples[:split_idx]
        self.test_samples = normalized_samples[split_idx:]

        print(
            f"Dataset loaded and split: {len(self.train_samples)} for training, "
            f"{len(self.test_samples)} for testing."
        )

    def _normalize_sample(self, sample: Dict[str, Any], idx: int = -1, skip_reasons: dict = None) -> Optional[Dict[str, Any]]:
        if skip_reasons is None:
            skip_reasons = {}
            
        image = self._extract_image(sample)
        if image is None:
            skip_reasons["no_image"] = skip_reasons.get("no_image", 0) + 1
            if idx < 3:  # 只对前几个样本输出详细调试信息
                print(f"  Sample {idx}: Missing image. Available keys: {list(sample.keys())}")
            return None
            
        bbox = self._extract_bbox(sample)
        if bbox is None:
            skip_reasons["no_bbox"] = skip_reasons.get("no_bbox", 0) + 1
            if idx < 3:
                print(f"  Sample {idx}: Missing bbox. Available keys: {list(sample.keys())}")
                # 尝试查找可能的bbox字段
                for key in sample.keys():
                    if "bbox" in key.lower() or "box" in key.lower() or "coord" in key.lower():
                        print(f"    Found potential bbox key: {key} = {sample[key]}")
            return None

        question = self._extract_text(sample, self._QUESTION_KEYS)
        if question is None:
            skip_reasons["no_question"] = skip_reasons.get("no_question", 0) + 1
            if idx < 3:
                print(f"  Sample {idx}: Missing question/instruction. Available keys: {list(sample.keys())}")
                # 尝试查找可能的question字段
                for key in sample.keys():
                    if any(term in key.lower() for term in ["text", "prompt", "query", "instruction", "desc"]):
                        print(f"    Found potential question key: {key} = {str(sample[key])[:100]}")
            return None

        answer = self._extract_text(sample, self._ANSWER_KEYS)

        normalized: Dict[str, Any] = {
            "image": image,
            "question": question,
            "bbox": bbox,
        }
        if answer:
            normalized["answer"] = answer

        return normalized

    def _extract_text(
        self, sample: Dict[str, Any], candidate_keys: Sequence[str]
    ) -> Optional[str]:
        for key in candidate_keys:
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _extract_image(self, sample: Dict[str, Any]) -> Optional[Image.Image]:
        image = sample.get("image") or sample.get("screenshot")
        if isinstance(image, Image.Image):
            return image

        # Some dataset variants only store relative paths.
        for key in self._IMAGE_PATH_KEYS:
            path_value = sample.get(key)
            if not path_value:
                continue
            try:
                path = Path(path_value)
                if path.exists():
                    return Image.open(path).convert("RGB")
            except Exception:
                continue

        return None

    def _extract_bbox(self, sample: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        # 策略1: 直接查找已知的bbox字段名
        for key in self._BBOX_KEYS:
            bbox_value = sample.get(key)
            parsed = self._parse_bbox_value(bbox_value)
            if parsed is not None:
                return parsed

        # 策略2: 模糊匹配字段名（包含bbox/box/coord等关键词）
        for key in sample.keys():
            key_lower = key.lower()
            if any(term in key_lower for term in ["bbox", "box", "coord", "bound", "rect", "region"]):
                bbox_value = sample.get(key)
                parsed = self._parse_bbox_value(bbox_value)
                if parsed is not None:
                    return parsed

        # 策略3: 单独的标量字段
        candidates = (
            ("x", "y", "w", "h"),
            ("left", "top", "width", "height"),
            ("x1", "y1", "x2", "y2"),  # 支持 [x1, y1, x2, y2] 格式
        )
        for keys in candidates:
            try:
                if all(k in sample for k in keys):
                    vals = [float(sample[k]) for k in keys]
                    if len(vals) == 4:
                        # 如果是 [x1, y1, x2, y2] 格式，转换为 [x, y, w, h]
                        if keys == ("x1", "y1", "x2", "y2"):
                            x1, y1, x2, y2 = vals
                            return (x1, y1, x2 - x1, y2 - y1)
                        return tuple(vals)  # type: ignore[return-value]
            except (KeyError, TypeError, ValueError):
                continue

        return None

    def _parse_bbox_value(
        self, value: Optional[Any]
    ) -> Optional[Tuple[float, float, float, float]]:
        if value is None:
            return None

        # 处理字典格式
        if isinstance(value, dict):
            # 尝试多种可能的键名组合
            x = value.get("x") or value.get("x1") or value.get("left") or value.get("x_min")
            y = value.get("y") or value.get("y1") or value.get("top") or value.get("y_min")
            w = value.get("w") or value.get("width")
            h = value.get("h") or value.get("height")
            
            # 如果有 x2, y2，计算 width 和 height
            if w is None and "x2" in value:
                x2 = value.get("x2") or value.get("right") or value.get("x_max")
                if x is not None and x2 is not None:
                    w = float(x2) - float(x)
            if h is None and "y2" in value:
                y2 = value.get("y2") or value.get("bottom") or value.get("y_max")
                if y is not None and y2 is not None:
                    h = float(y2) - float(y)
            
            try:
                if x is not None and y is not None and w is not None and h is not None:
                    return float(x), float(y), float(w), float(h)
            except (TypeError, ValueError):
                pass
            return None

        # 处理列表/元组格式
        if isinstance(value, (list, tuple)):
            seq: Sequence[Any] = value
            if len(seq) == 4:
                try:
                    coords = tuple(float(v) for v in seq)  # type: ignore[return-value]
                    # 假设是 [x, y, w, h] 格式，直接返回
                    return coords
                except (TypeError, ValueError):
                    return None
            elif len(seq) >= 4:
                # 可能是嵌套结构，尝试提取前4个数字
                try:
                    nums = [float(v) for v in seq[:4]]
                    return tuple(nums)  # type: ignore[return-value]
                except (TypeError, ValueError):
                    return None

        return None

    def _get_cache_path(self):
        """获取数据集缓存路径"""
        # 获取 datasets 库的缓存目录
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = os.path.join(os.path.expanduser(hf_home), "datasets")
        
        # 构建数据集特定的缓存路径
        dataset_name_safe = self.name.replace("/", "___")
        dataset_cache_path = os.path.join(cache_dir, dataset_name_safe)
        return cache_dir, dataset_cache_path

    def _check_cache_status(self):
        """检查缓存状态"""
        cache_dir, dataset_cache_path = self._get_cache_path()
        exists = os.path.exists(dataset_cache_path)
        
        # 检查是否有子目录（版本目录）
        version_dirs = []
        if exists:
            try:
                for item in os.listdir(dataset_cache_path):
                    item_path = os.path.join(dataset_cache_path, item)
                    if os.path.isdir(item_path):
                        version_dirs.append(item)
            except Exception:
                pass
        
        return {
            "cache_dir": cache_dir,
            "cache_path": dataset_cache_path,
            "exists": exists,
            "version_dirs": version_dirs
        }

    def _cleanup_cache(self):
        """清理可能损坏的数据集缓存"""
        try:
            cache_dir, dataset_cache_path = self._get_cache_path()
            
            if os.path.exists(dataset_cache_path):
                print(f"Removing cache directory: {dataset_cache_path}")
                try:
                    shutil.rmtree(dataset_cache_path)
                    print("Cache directory removed successfully.")
                except Exception as e:
                    print(f"Warning: Could not fully remove cache directory: {e}")
                    # 尝试删除特定版本目录
                    try:
                        for item in os.listdir(dataset_cache_path):
                            item_path = os.path.join(dataset_cache_path, item)
                            try:
                                if os.path.isdir(item_path):
                                    shutil.rmtree(item_path)
                            except Exception:
                                pass
                    except Exception:
                        pass
            else:
                print(f"Cache directory does not exist: {dataset_cache_path}")
        except Exception as e:
            print(f"Warning: Error during cache cleanup: {e}")
            # 不阻止继续执行


