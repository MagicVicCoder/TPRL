from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    All data loaders should inherit from this class and implement its methods.
    """
    def __init__(self, name, split, split_ratio):
        self.name = name
        self.split = split
        self.split_ratio = split_ratio
        self.train_samples = []
        self.test_samples = []
        self._load_and_split_data()

    @abstractmethod
    def _load_and_split_data(self):
        """
        Load the dataset and split it into training and testing sets.
        This method should populate self.train_samples and self.test_samples.
        """
        pass

    def get_train_samples(self):
        return self.train_samples

    def get_test_samples(self):
        return self.test_samples


def get_data_loader(config):
    """
    Factory function to get the specified data loader.
    """
    if config.DATASET_NAME == "lmms-lab/MME":
        from .mme_loader import MMEDataLoader
        return MMEDataLoader(
            name=config.DATASET_NAME,
            split=config.DATASET_SPLIT,
            split_ratio=config.TRAIN_TEST_SPLIT_RATIO
        )
    elif config.DATASET_NAME in {"Screen-Pro", "ScreenSpot-Pro", "Voxel51/ScreenSpot-Pro"}:
        from .screen_pro_loader import ScreenProDataLoader
        resolved_name = (
            "Voxel51/ScreenSpot-Pro"
            if config.DATASET_NAME in {"Screen-Pro", "ScreenSpot-Pro"}
            else config.DATASET_NAME
        )
        return ScreenProDataLoader(
            name=resolved_name,
            split=config.DATASET_SPLIT,
            split_ratio=config.TRAIN_TEST_SPLIT_RATIO
        )
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET_NAME}")