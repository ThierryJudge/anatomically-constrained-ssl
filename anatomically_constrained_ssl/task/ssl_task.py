from typing import Dict
import torch.nn.functional as F
from torch import Tensor
from anatomically_constrained_ssl.data.iterator import SemiSupervisedIterator
from vital.data.config import Tags

from vital.tasks.segmentation import SegmentationTask


class Baseline(SegmentationTask):
    pass


class SSLTask(SegmentationTask):
    """Pytorch Lightning module for semi-supervised learning.

    Override methods for semi-supervised behavior.
    """

    def training_step(self, batch, batch_idx):  # noqa: D102
        if SemiSupervisedIterator.is_labeled(batch):
            return self.supervised_training_step(SemiSupervisedIterator.get_batch(batch), batch_idx)
        else:
            return self.unsupervised_training_step(SemiSupervisedIterator.get_batch(batch), batch_idx)

    def supervised_training_step(self, batch, *args, **kwargs) -> Dict:  # noqa: D102
        raise NotImplementedError

    def unsupervised_training_step(self, batch, *args, **kwargs) -> Dict:  # noqa: D102
        raise NotImplementedError

    def compute_segmentation_metrics(self, y_hat: Tensor, y: Tensor):  # noqa D102
        ce = F.cross_entropy(y_hat, y)
        dice_values = self._dice(y_hat, y)
        dices = {f"dice_{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}

        return {"ce": ce, "dice": dice_values.mean(), **dices}