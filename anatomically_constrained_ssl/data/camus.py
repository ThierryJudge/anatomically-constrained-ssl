import random
from pathlib import Path
from typing import Literal, Union, Optional
import torch.utils.data

from vital.data.camus.config import CamusTags
from vital.data.config import Subset
from vital.metrics.camus.anatomical.utils import check_segmentation_validity
from vital.data.camus.data_module import CamusDataModule

from anatomically_constrained_ssl.data.dataset import SemiSupervisedLearningDataset
from anatomically_constrained_ssl.data.iterator import SemiSupervisedIterator


class CamusSSLDataModule(CamusDataModule):
    """Implementation of the ``VitalDataModule`` for the CAMUS dataset."""

    def __init__(
            self,
            dataset_path: Union[str, Path],
            label_split: float = 1,
            num_steps: Optional[int] = None,
            p: Optional[None] = None,
            drop_last: bool = False,
            limit_val_set: bool = False,
            **kwargs,
    ):
        """Initializes class instance.

        Args:
            dataset_path: Path to the HDF5 dataset.
            labels: Labels of the segmentation classes to take into account (including background). If None, target all
                labels included in the data.
            fold: ID of the cross-validation fold to use.
            use_sequence: Enable use of full temporal sequences.
            num_neighbors: Number of neighboring frames on each side of an item's frame to include as part of an item's
                data.
            neighbor_padding: Mode used to determine how to pad neighboring instants at the beginning/end of a sequence.
                The options mirror those of the ``mode`` parameter of ``numpy.pad``.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(dataset_path, **kwargs)
        self.label_split = label_split
        self.num_steps = num_steps
        self.p = p
        self.drop_last = drop_last
        self.limit_val_set = limit_val_set

        self.voxel_tag = CamusTags.voxelspacing
        self.check_segmentation_validity = check_segmentation_validity

    def setup(self, stage: Literal["fit", "test"]) -> None:  # noqa: D102
        super().setup(stage)
        if stage == "fit":
            self._dataset[Subset.TRAIN] = SemiSupervisedLearningDataset(self._dataset[Subset.TRAIN],
                                                                        label_split=self.label_split)
            if self.limit_val_set:
                full_set_len = len(self._dataset[Subset.VAL])
                indices = random.sample(range(full_set_len), int(self.label_split * full_set_len))
                self._dataset[Subset.VAL] = torch.utils.data.Subset(self._dataset[Subset.VAL], indices)

    def train_dataloader(self) -> SemiSupervisedIterator:  # noqa: D102
        return SemiSupervisedIterator(
            self.dataset(subset=Subset.TRAIN),
            self.batch_size,
            num_steps=self.num_steps,
            p=self.p,
            num_workers=self.num_workers,
            shuffle=True,
        )
