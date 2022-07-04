import comet_ml # noqa
from typing import Sequence, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from torch import nn
from torchmetrics.functional import accuracy


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 26 * 26, 128)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn1(x)
        x = self.fc2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc3 = nn.Linear(20, 128)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        x = self.fc3(x)
        x = self.bn1(x)
        x = self.lrelu1(x)
        x = self.fc4(x)
        return x


class MnistNonDiffOptim(pl.LightningModule):
    """
        System for training classification without a differentiable loss.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.adversarial_loss = nn.BCEWithLogitsLoss()

        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    @staticmethod
    def get_discriminator_target(pred, gt):
        """Get discriminator targets.
            Discriminator targets are the the result of the non-differentiable function being optimized.
            In this case the function is np.equals.

        Args:
            pred (Tensor): Model prediction
            gt (Tensor): Data gt

        Returns:
            Tensor, discriminator_targets
        """
        target = torch.from_numpy(np.equal(np.argmax(gt.cpu().detach().numpy(), axis=-1),
                                           np.argmax(pred.detach().cpu().numpy(), axis=-1)) == 1).float()
        target = target.view((-1, 1))

        return target

    def training_step(self, batch, batch_nb):
        opt_g, opt_d = self.optimizers()
        img, gt = batch
        gt = torch.nn.functional.one_hot(gt, num_classes=10)
        gt = gt.float()

        generator_prediction = self.forward(img)
        generator_target = torch.ones(img.shape[0], 1).type_as(generator_prediction)
        generator_loss = self.adversarial_loss(self.discriminator(generator_prediction, gt), generator_target)

        acc = accuracy(generator_prediction, torch.argmax(gt, -1))

        logs = {'loss': generator_loss.detach(),
                'accuracy': acc.detach(),
                'generator_loss': generator_loss.detach()}

        self.manual_backward(generator_loss)
        opt_g.step()
        opt_g.zero_grad()

        discriminator_target = self.get_discriminator_target(generator_prediction, gt).type_as(generator_prediction)

        discriminator_prediction = self.discriminator(generator_prediction.detach(), gt)

        discriminator_loss = self.adversarial_loss(discriminator_prediction, discriminator_target)

        discriminator_accuracy = (discriminator_prediction.round() == discriminator_target).float().mean()

        logs.update({'loss': discriminator_loss.detach(),
                     'discriminator_loss': discriminator_loss.detach(),
                     'discriminator_accuracy': discriminator_accuracy.detach()})

        self.manual_backward(discriminator_loss)
        opt_d.step()
        opt_d.zero_grad()

        self.log_dict(logs)
        return logs

    def testval_step(self, batch, batch_nb, prefix):
        img, gt = batch
        gt = torch.nn.functional.one_hot(gt, num_classes=10)
        gt = gt.float()

        # GENERATOR STEP
        generator_prediction = self.forward(img)
        generator_target = torch.ones(img.shape[0], 1).type_as(generator_prediction)
        generator_loss = self.adversarial_loss(self.discriminator(generator_prediction, gt), generator_target)

        # DISCRIMINATOR STEP
        discriminator_target = self.get_discriminator_target(generator_prediction, gt).type_as(generator_prediction)
        discriminator_prediction = self.discriminator(generator_prediction.detach(), gt)
        discriminator_loss = self.adversarial_loss(discriminator_prediction, discriminator_target)
        discriminator_accuracy = (discriminator_prediction.round() == discriminator_target).float().mean()

        acc = accuracy(generator_prediction, torch.argmax(gt, -1))

        logs = {f'{prefix}_loss': generator_loss + discriminator_loss,
                f'{prefix}_generator_loss': generator_loss,
                f'{prefix}_discriminator_loss': discriminator_loss,
                f'{prefix}_discriminator_accuracy': discriminator_accuracy,
                f'{prefix}_accuracy': acc
                }
        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_step(self, batch, batch_nb):
        return self.testval_step(batch, batch_nb, 'val')

    def test_step(self, batch, batch_nb):
        return self.testval_step(batch, batch_nb, 'test')

    def configure_optimizers(self):
        b1, b2 = 0.5, 0.999
        lr = 0.0002
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []


if __name__ == '__main__':
    import dotenv

    dotenv.load_dotenv(override=True)

    model = MnistNonDiffOptim()
    dm = MNISTDataModule(batch_size=32)

    logger = CometLogger()
    trainer = Trainer(max_epochs=10, gpus=int(torch.cuda.is_available()), logger=logger)

    trainer.fit(model, dm)

    trainer.test(datamodule=dm)
