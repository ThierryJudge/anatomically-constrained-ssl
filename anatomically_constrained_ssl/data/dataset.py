"""Inspired by https://github.com/ElementAI/baal/blob/master/src/baal/active/dataset.py."""
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset, random_split


class SemiSupervisedLearningDataset(Dataset):
    """Wrapper for dataset used for semi-supervised learning."""

    def __init__(self, dataset: Dataset, label_split: Optional[float] = None, pool: Optional[Dataset] = None):
        """Initialize dataset and pool.

        Args:
            dataset: Original dataset
            pool: Optional dataset used as unsupervised pool
            label_split: fraction of the original dataset that will be labeled
        """
        full_len = len(dataset)
        label_len = int(label_split * full_len)
        self._dataset, self._pool = random_split(dataset, [label_len, full_len - label_len])

        # If the pool is not None, use it, otherwise use split from original dataset.
        self._pool = pool or self._pool

    def __getitem__(self, index: int):
        """Return stuff from the original dataset."""
        return self._dataset[index]

    def __len__(self) -> int:
        """Return how many actual data / label pairs we have."""
        return len(self._dataset)

    @property
    def pool(self) -> Dataset:
        """Returns a new Dataset made from unlabelled samples.

        Raises:
            ValueError if a pool specific attribute cannot be set.
        """
        return self._pool
