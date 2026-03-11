import random
from datasets import load_dataset
from .base_loader import BaseDataLoader

class MMEDataLoader(BaseDataLoader):
    """
    Data loader for VQA datasets like MME.
    """
    def _load_and_split_data(self):
        print(f"Loading dataset: {self.name}")
        try:
            dataset = load_dataset(self.name, split=self.split)
        except Exception as e:
            print(f"Failed to load dataset {self.name}. Error: {e}")
            raise

        all_samples = [item for item in dataset]
        random.shuffle(all_samples)

        split_idx = int(self.split_ratio * len(all_samples))
        self.train_samples = all_samples[:split_idx]
        self.test_samples = all_samples[split_idx:]

        print(f"Dataset loaded and split: {len(self.train_samples)} for training, {len(self.test_samples)} for testing.")