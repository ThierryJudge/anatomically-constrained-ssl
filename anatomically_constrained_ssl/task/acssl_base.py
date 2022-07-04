# flake8: noqa
# TODO
import functools
import random
from collections import deque
from typing import Dict, Optional, List
import re

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from sklearn.metrics import confusion_matrix
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR
from vital.data.config import Tags
from vital.utils.format.native import prefix
from matplotlib import pyplot as plt
from anatomically_constrained_ssl.task.ssl_task import SSLTask


def get_discriminator(input_channels, num_classes) -> nn.Module:
    discriminator = models.resnet50(num_classes=num_classes)
    discriminator.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # discriminator = patch_module(discriminator)
    return discriminator


class AnatomicallyConstrainedLearningBase(SSLTask):
    """Class for Anatomically Constrained Learning.

    This method uses a discriminator network to learn a non-differentiable metric function.
    The segmentation network is trained adversarially to maximise the metric function.
    """

    def __init__(
            self,
            lambda_sup=1,
            lambda_adv=0.1,
            lambda_semi=0.5,
            mem_len: int = 0,
            pretraining_steps: int = 1000,
            training_schedule: str = '1:1',
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.output_size = 1
        self.discriminator = get_discriminator(len(self.hparams.data_params.labels), self.output_size)

        self.adversarial_loss = nn.BCEWithLogitsLoss()

        pattern = re.compile(r"[0-9]+:[0-9]+$")
        assert bool(pattern.match(training_schedule))
        x = training_schedule.split(":")
        print(x)
        self.training_schedule = np.concatenate([np.ones(int(x[0])), np.zeros(int(x[1]))])
        print(self.training_schedule)

        self.lambda_sup = lambda_sup
        self.lambda_adv = lambda_adv
        self.lambda_semi = lambda_semi

        self.sample_memory = deque(maxlen=mem_len)

        # Used for logging confusion matrix
        self._train_discriminator_target, self._train_discriminator_pred = [], []
        self._val_discriminator_target, self._val_discriminator_pred = [], []

    def on_fit_start(self) -> None:
        print("Steps per epoch", len(self.trainer.datamodule.train_dataloader()))
        print(self.pretraining_epochs)
        for callback in self.trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                callback.patience = callback.patience + self.pretraining_epochs

    @functools.cached_property
    def pretraining_epochs(self):
        return int(self.hparams.pretraining_steps / len(self.trainer.datamodule.train_dataloader()))

    def get_discriminator_target(self, pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get discriminator targets.

        Args:
            pred (Tensor): Model prediction
            kwargs (dict): dictionary of extra parameters

        Returns:
            Tensor, discriminator_targets
        """
        voxel = kwargs[self.trainer.datamodule.voxel_tag]
        with torch.no_grad():
            validities = []
            for y, vox in zip(pred.argmax(dim=1).cpu().detach().numpy(), voxel.cpu().detach().numpy()):
                try:
                    validities.append(self.trainer.datamodule.check_segmentation_validity(y, vox))
                except:
                    validities.append(np.zeros(self.output_size))
            target = torch.from_numpy(np.array(validities, dtype="float32")).view((-1, self.output_size)).to(
                self.device)
        return target

    def save_sample(self, predictions: Tensor, metric_values: Tensor) -> None:
        """Save samples to avoid class imbalance while training the discriminator.

        Args:
            predictions: predictions to be saved and used to train discriminator.
            metric_values: binary values of the metric for each prediction
        """
        if self.hparams.mem_len > 0:
            predictions = predictions.detach().cpu()
            metric_values = metric_values.detach().cpu()
            positive_idx = torch.where(metric_values.all(dim=-1) == 1)[0]
            negative_idx = torch.where(metric_values.all(dim=-1) == 0)[0]

            if len(self.sample_memory) > 0:
                if len(positive_idx) >= len(negative_idx):
                    positive_idx = np.random.choice(positive_idx.cpu(), size=len(negative_idx))
                else:
                    negative_idx = np.random.choice(negative_idx.cpu(), size=len(positive_idx))

                self.sample_memory.extend(list(zip(predictions[positive_idx], metric_values[positive_idx])))
                self.sample_memory.extend(list(zip(predictions[negative_idx], metric_values[negative_idx])))

            else:
                self.sample_memory.extend(list(zip(predictions, metric_values)))

    def get_samples(self, example_tensor: Tensor):
        """Get samples from sample_memory
        Args:
           example_tensor:
        Returns:
           generator_prediction:
           discriminator_target:
        """

        batch = random.sample(
            self.sample_memory,
            len(example_tensor) if len(self.sample_memory) > len(example_tensor) else len(self.sample_memory),
        )
        generator_prediction, discriminator_target = zip(*batch)

        shape = (-1, *example_tensor.shape[1:4])
        generator_prediction = torch.cat(generator_prediction).view(shape).type_as(example_tensor)
        discriminator_target = torch.cat(discriminator_target).view(-1, self.output_size).type_as(example_tensor)

        return generator_prediction, discriminator_target

    def validation_step(self, batch: Dict, batch_nb: int) -> Dict:
        """Validation step for on batch.

        Args:
            batch (Dict): input batch
            batch_nb (int): batch number

        Returns:
            dict, loss, metrics and logs.
        """
        x, y = batch[Tags.img], batch[Tags.gt]

        y_hat = self.forward(x)

        # Segmentation accuracy metrics
        metrics = self.compute_segmentation_metrics(y_hat, y)

        supervised_loss = (self.hparams.ce_weight * metrics["ce"]) + (self.hparams.dice_weight * (1 - metrics["dice"]))

        discriminator_target = self.get_discriminator_target(y_hat, **batch)

        discriminator_prediction = self.discriminator(F.softmax(y_hat.detach(), dim=1))

        discriminator_loss = self.adversarial_loss(discriminator_prediction, discriminator_target)

        generator_target = torch.ones(x.shape[0], self.output_size).type_as(x)

        adversarial_loss = self.adversarial_loss(self.discriminator(F.softmax(y_hat.detach(), dim=1)), generator_target)

        discriminator_prediction = torch.sigmoid(discriminator_prediction).round()
        discriminator_accuracy = discriminator_prediction.eq(discriminator_target).float().mean()

        if batch_nb == 0:
            y_hat = y_hat.argmax(1) if y_hat.shape[1] > 1 else torch.sigmoid(y_hat).round()
            self.log_images(
                title="Sample",
                num_images=5,
                axes_content={
                    "Image": x.cpu().squeeze().numpy(),
                    "Gt": y.squeeze().cpu().numpy(),
                    "Pred": y_hat.detach().cpu().squeeze().numpy(),
                },
                info=[f"(Valid: {discriminator_target[i].item()} - "
                      f"Pred_valid: {discriminator_prediction[i].item()})" for i in range(5)],

            )

        # Record values to log confusion matrix
        self._val_discriminator_target.append(discriminator_target.detach())
        self._val_discriminator_pred.append(discriminator_prediction.detach())

        logs = {
            "early_stop_on": supervised_loss,
            "val_loss": supervised_loss,
            "val_AnatomicalValidity": discriminator_target.float().mean(),
            "val_predicted_anatomical_validity": discriminator_prediction.float().mean(),
            "val_discriminator_loss": discriminator_loss,
            "val_discriminator_accuracy": discriminator_accuracy,
            "val_generator_adversarial_loss": adversarial_loss,
        }

        logs.update(prefix(metrics, "val_"))
        logs.update(metrics)
        self.log_dict(logs, **self.hparams.val_log_kwargs)
        return logs

    def training_epoch_end(self, outputs) -> None:
        if len(self._train_discriminator_target) > 0 and len(self._train_discriminator_pred) > 0:
            with torch.no_grad():
                self.log_confusion_matrix(
                    torch.cat(self._train_discriminator_target, dim=0),
                    torch.cat(self._train_discriminator_pred, dim=0),
                    "train",
                )
            self._train_discriminator_target, self._train_discriminator_pred = [], []

    def validation_epoch_end(self, outputs) -> None:
        self.log_confusion_matrix(
            torch.cat(self._val_discriminator_target, dim=0), torch.cat(self._val_discriminator_pred, dim=0), "val"
        )
        self._val_discriminator_target, self._val_discriminator_pred = [], []

    def log_confusion_matrix(self, target: Tensor, pred: Tensor, set):
        if isinstance(self.trainer.logger, CometLogger):
            cm = confusion_matrix(target.flatten().tolist(), pred.flatten().tolist())
            file_name = f"confusion-matrix-{set}-{self.current_epoch:04d}'.json"
            self.trainer.logger.experiment.log_confusion_matrix(matrix=cm, step=self.current_epoch, file_name=file_name)

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()
        sch.step()

    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999

        opt_g = hydra.utils.instantiate(self.hparams.optim, params=self.model.parameters(), betas=(b1, b2))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(b1, b2), weight_decay=0.001
        )

        lambda_d = lambda epoch: 1 if epoch < self.pretraining_epochs else 0.5
        scheduler_d = LambdaLR(opt_d, lr_lambda=lambda_d)
        return [opt_g, opt_d], [scheduler_d]

    def log_images(
        self, title: str, num_images: int, axes_content: Dict[str, np.ndarray], info: Optional[List[str]] = None
    ):
        """Log images to Logger if it is a TensorBoardLogger or CometLogger.
        Args:
            title: Name of the figure.
            num_images: Number of images to log.
            axes_content: Mapping of axis name and image.
            info: Additional info to be appended to title for each image.
        """
        for i in range(num_images):
            fig, axes = plt.subplots(1, len(axes_content.keys()), squeeze=False)
            if info is not None:
                name = f"{title}_{info[i]}_{i}"
            else:
                name = f"{title}_{i}"
            plt.suptitle(name)
            axes = axes.ravel()
            for j, (ax_title, img) in enumerate(axes_content.items()):
                axes[j].imshow(img[i].squeeze())
                axes[j].set_title(ax_title)

            if isinstance(self.trainer.logger, TensorBoardLogger):
                self.trainer.logger.experiment.add_figure("{}_{}".format(title, i), fig, self.current_epoch)
            if isinstance(self.trainer.logger, CometLogger):
                self.trainer.logger.experiment.log_figure("{}_{}".format(title, i), fig, step=self.current_epoch)

            plt.close()
