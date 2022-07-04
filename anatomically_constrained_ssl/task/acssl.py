import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from vital.data.config import Tags

from anatomically_constrained_ssl.task.acssl_base import AnatomicallyConstrainedLearningBase


class ACSSL(AnatomicallyConstrainedLearningBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def supervised_training_step(self, batch, batch_idx) -> Dict:
        opt_g, opt_d = self.optimizers()
        x, y = batch[Tags.img], batch[Tags.gt]

        generator_prediction = self.forward(x)
        discriminator_target = self.get_discriminator_target(generator_prediction, **batch)
        self.save_sample(generator_prediction, discriminator_target)

        # if self.global_step >= self.hparams.pretraining_steps:
        if self.current_epoch >= self.pretraining_epochs and self.training_schedule[self.global_step % len(self.training_schedule)]:
            metrics = self.compute_segmentation_metrics(generator_prediction, y)
            supervised_loss = (self.hparams.ce_weight  * metrics["ce"]) + (self.hparams.dice_weight * (1 - metrics["dice"]))

            generator_target = torch.ones(x.shape[0], self.output_size).type_as(x)
            adversarial_loss = self.adversarial_loss(
                self.discriminator(F.softmax(generator_prediction, dim=1)), generator_target
            )

            loss = self.lambda_sup * supervised_loss + self.lambda_adv * adversarial_loss

            logs = {
                "generator_loss": loss.detach(),
                "supervised_loss": supervised_loss.detach(),
                "supervised_adversarial_loss": adversarial_loss.detach(),
                "supervised_anatomical_validity": discriminator_target.all(dim=-1).float().mean().detach(),
            }

            logs.update({key: val.detach() for key, val in metrics.items()})

            self.manual_backward(loss)
            opt_g.step()
            opt_g.zero_grad()

            # if (batch_idx + 1) % self.hparams.discriminator_step_update == 0:
            #     discriminator_logs = self.discriminator_training_step(
            #         generator_prediction.detach(), discriminator_target, opt_d, "supervised"
            #     )
            #     logs.update(discriminator_logs)
        else:
            logs = self.discriminator_training_step(
                generator_prediction.detach(), discriminator_target, opt_d, "supervised"
            )

        logs['pretraining'] = float(int(self.current_epoch < self.pretraining_epochs))
        logs['memory_length'] = float(len(self.sample_memory))
        logs['schedule'] = float(int(self.training_schedule[self.global_step % len(self.training_schedule)]))
        self.log_dict(logs, **self.hparams.train_log_kwargs)

        return logs

    def unsupervised_training_step(self, batch, batch_idx) -> Dict:
        opt_g, opt_d = self.optimizers()

        x = batch[Tags.img]

        generator_prediction = self.forward(x)
        discriminator_target = self.get_discriminator_target(generator_prediction, **batch)
        self.save_sample(generator_prediction, discriminator_target)

        # if self.global_step >= self.hparams.pretraining_steps:
        # if self.current_epoch >= self.pretraining_epochs and :
        if self.current_epoch >= self.pretraining_epochs and self.training_schedule[self.global_step % len(self.training_schedule)]:
            generator_target = torch.ones(x.shape[0], self.output_size).type_as(generator_prediction)

            adversarial_loss = self.adversarial_loss(
                self.discriminator(F.softmax(generator_prediction, dim=1)), generator_target
            )

            loss = self.lambda_semi * adversarial_loss

            logs = {
                "unsupervised_adversarial_loss": adversarial_loss.detach(),
                "unsupervised_anatomical_validity": discriminator_target.all(dim=-1).float().mean().detach(),
            }

            self.manual_backward(loss)
            opt_g.step()
            opt_g.zero_grad()

            # if (batch_idx + 1) % self.hparams.discriminator_step_update == 0:
            #     discriminator_logs = self.discriminator_training_step(
            #         generator_prediction.detach(), discriminator_target, opt_d, "unsupervised"
            #     )
            #     logs.update(discriminator_logs)
        else:
            logs = self.discriminator_training_step(
                generator_prediction.detach(), discriminator_target, opt_d, "unsupervised"
            )

        logs['schedule'] = float(self.training_schedule[self.global_step % len(self.training_schedule)])
        self.log_dict(logs, **self.hparams.train_log_kwargs)

        return logs

    def discriminator_training_step(
            self,
            generator_prediction: Tensor,
            discriminator_target: Tensor,
            optimizer: Optimizer,
            log_prefix: str
    ) -> Dict:
        """Train the discriminator to learn the metric function.

        Args:
            generator_prediction:
            discriminator_target:
            optimizer:
            batch: input batch
            log_prefix (str): prefix for the log keys. ('supervised', 'unsupervised')

        Returns:
            dict, loss and log metrics.
        """
        # if not self.hparams.freeze_discriminator or self.global_step < self.hparams.pretraining_steps:
        # if not self.hparams.freeze_discriminator or self.current_epoch >= self.pretraining_epochs:
        if self.hparams.mem_len > 0:
            generator_prediction, discriminator_target = self.get_samples(generator_prediction)

        discriminator_prediction = self.discriminator(F.softmax(generator_prediction.detach(), dim=1))

        discriminator_loss = self.adversarial_loss(discriminator_prediction, discriminator_target)

        discriminator_prediction = torch.sigmoid(discriminator_prediction).round()
        discriminator_accuracy = discriminator_prediction.eq(discriminator_target).float().mean()

        # Record values to log confusion matrix
        self._train_discriminator_target.append(discriminator_target.detach())
        self._train_discriminator_pred.append(discriminator_prediction.detach())

        logs = {
            f"{log_prefix}_discriminator_loss": discriminator_loss.detach(),
            f"{log_prefix}_discriminator_accuracy": discriminator_accuracy.detach(),
            f"{log_prefix}_discriminator_target_mean": discriminator_target.mean().detach(),
        }

        self.manual_backward(discriminator_loss)
        optimizer.step()
        optimizer.zero_grad()

        return logs
        # else:
        #     return {}

    def on_validation_end(self) -> None:
        if self.current_epoch == self.pretraining_epochs:
            torch.save(self.state_dict(), "discriminator.ckpt")


