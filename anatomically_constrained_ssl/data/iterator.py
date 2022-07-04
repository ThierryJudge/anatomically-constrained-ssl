"""Copied from https://github.com/ElementAI/baal/blob/master/src/baal/utils/ssl_iterator.py."""
from itertools import cycle
from typing import Dict, Optional, Sequence, Union

import numpy as np
from vital.data.config import Tags

from anatomically_constrained_ssl.data.dataset import SemiSupervisedLearningDataset
from torch.utils.data import DataLoader


class AlternateIterator:
    """Create an iterator that will alternate between two dataloaders."""

    LABELED = 0
    UN_LABELED = 1

    def __init__(
            self,
            dl_1: DataLoader,
            dl_2: DataLoader,
            num_steps: Optional[int] = None,
            p: Optional[float] = None
    ):
        """
        Args:
            dl_1: first DataLoader
            dl_2: second DataLoader
            num_steps: Number of steps, if None will be the sum of both length.
            p : Probability of choosing dl_1 over dl_2. If None, will be alternate between the two.
        """
        self.dl_1 = dl_1
        self.dl_1_iter = cycle(dl_1)
        # self.dl_1_iter = iter(dl_1)
        self.len_dl1 = len(dl_1)

        if dl_2 is not None:  # Avoid crash if p=1
            self.dl_2 = dl_2
            self.dl_2_iter = cycle(dl_2)
            # self.dl_2_iter = iter(dl_2)
            self.len_dl2 = len(dl_2)
        else:
            p = 1

        self.num_steps = num_steps or (self.len_dl1 + self.len_dl2)
        self.p = None if p is None else [p, 1 - p]
        self._pool = None
        self._iter_idx = None

    def _make_index(self):
        if self.p is None:
            # If p is None, we just alternate.
            arr = np.array([i % 2 for i in range(self.num_steps)])
        else:
            arr = np.random.choice([self.LABELED, self.UN_LABELED], self.num_steps, p=self.p)
        return list(arr)

    def __len__(self):  # noqa D105
        return self.num_steps

    def __iter__(self):  # noqa D105
        # hack to prevent multiple __iter__ calls in _with_is_last(...) pytorch_lightning/trainer/training_loop.py
        if self._iter_idx is None or len(self._iter_idx) <= 0:
            self._iter_idx = self._make_index()
        return self

    def __next__(self):  # noqa D105
        if len(self._iter_idx) <= 0:
            raise StopIteration
        idx = self._iter_idx.pop(0)
        if idx == self.LABELED:
            return self.handle_format(next(self.dl_1_iter), idx)
        else:
            return self.handle_format(next(self.dl_2_iter), idx)

    def handle_format(self, item, idx):  # noqa D102
        return item, idx


class SemiSupervisedIterator(AlternateIterator):
    """Class to be used as dataloader to alternate between labeled and un-labeled data."""

    IS_LABELED_TAG = "is_labelled"

    def __init__(
        self,
        ssl_dataset: SemiSupervisedLearningDataset,
        batch_size: int,
        num_steps: Optional[int] = None,
        p: Optional[None] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False

    ):
        """
        Args:
            ssl_dataset: dataset that contains both labeled dataset and unlabeled pool
            batch_size: size of the batch
            num_steps: Number of steps, if None will be the sum of both length.
            p : Probability of choosing dl_1 over dl_2. If None, will be alternate between the two.
            shuffle: Whether to shuffle the dataloaders
            num_workers: Number of workers for dataloaders
            drop_last: Drop last passed to dataloader.
        """
        self.al_dataset = ssl_dataset
        active_dl = DataLoader(
            ssl_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
            pin_memory=pin_memory
        )

        if len(ssl_dataset.pool) > 0:
            pool_dl = DataLoader(
                ssl_dataset.pool, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
                pin_memory=pin_memory
            )

            if num_steps is None:
                if p is None:
                    # By default num_steps if 2 times the length of active set or less if pool is too small.
                    # This allows all the labeled data to be seen during one epoch.
                    num_steps = len(active_dl) + min(len(active_dl), len(pool_dl))
                else:
                    # Show all labeled data + unlabeled data.
                    num_steps = int(len(active_dl) + len(active_dl) * (1.0 - p) / p)
        else:
            pool_dl = None
            p = 1

        if p == 1:
            # Allows running only supervised training.
            num_steps = len(active_dl)

        super().__init__(dl_1=active_dl, dl_2=pool_dl, num_steps=num_steps, p=p)

    def handle_format(self, item, idx):  # noqa D102
        if isinstance(item, dict):
            item.update({self.IS_LABELED_TAG: idx == 0})
            if idx != 0 and Tags.gt in item.keys():
                item.pop(Tags.gt)
            return item
        else:
            return item, idx

    @staticmethod
    def is_labeled(batch: Union[Dict, Sequence]) -> bool:
        """Check if batch returned from SemiSupervisedIterator is labeled.

        Args:
            batch (Union[Dict, Sequence]): batch to check

        Returns:
            bool, if batch is labeled.
        """
        if isinstance(batch, dict):
            return batch[SemiSupervisedIterator.IS_LABELED_TAG]
        elif isinstance(batch, tuple):
            item, idx = batch
            return idx == SemiSupervisedIterator.LABELED

    @staticmethod
    def get_batch(batch: Union[Dict, Sequence]):
        """Get batch without is_labeled.

        Args:
            batch (Union[Dict, Sequence]): batch

        Returns:
            batch without is_labeled
        """
        if isinstance(batch, dict):
            return batch
        elif isinstance(batch, tuple):
            item, idx = batch
            return item
