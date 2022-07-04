from copy import copy
from typing import Any, Dict, List, Tuple, Sequence, Union

import h5py
import numpy as np
import pandas as pd
from medpy import metric
from pytorch_lightning import Callback
from vital.data.camus.config import CamusTags, seg_save_options

import pytorch_lightning as pl

from vital.utils.format.native import prefix
from vital.utils.format.numpy import to_categorical
from vital.utils.image.transform import resize_image
from matplotlib import pyplot as plt

from vital.data.camus.config import Label

from vital.metrics.camus.anatomical.utils import check_segmentation_validity


class CamusResults(Callback):
    """Implementation of the mixin handling the evaluation phase for the ACDC dataset."""

    scores = {"dice": metric.dc}
    distances = {"hausdorff": metric.hd, "assd": metric.assd}
    reductions = {"anatomical_errors": np.sum}
    num_figures = 0

    def __init__(self, labels: Sequence[Union[str, Label]]):
        self.labels = labels

    def reduce(self, df: pd.DataFrame, index: str = None) -> dict:
        """Reduce a dataframe according to reductions."""
        d = {}
        for col in df.columns:
            if col is not index:
                if col in self.reductions.keys():
                    d[col] = self.reductions[col](df[col])
                else:
                    d[col] = np.mean((df[col]))

        return d

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: List[Any]) -> None:
        """Performs test-time inference for a patient and saves the results to an HDF5 file.

        Args:
            batch: Structured batch of data for which to perform test-time i    #nference and save the results.
            batch_idx: Index of the patient in the test subset list on which the predictions were made.
        """
        patient_logs = []
        instant_logs = []

        with h5py.File(self.log_dir / "test.h5", "a") as dataset:
            patient_group = dataset.create_group(batch.id)

            for view, data in batch.views.items():
                pred = to_categorical(self(data.img_proc.to(self.device)).cpu().detach().numpy(), channel_axis=1)

                height, width = data.gt.shape[1:]  # Extract images' original dimensions
                for instant, i in data.instants.items():
                    y = resize_image(pred[i].squeeze(), (width, height))
                    instant_log = self.compute_instant_metrics(y, data.gt[i], data.voxelspacing)

                    patient_logs.append(copy(instant_log))
                    instant_log.update({"InstantID": f"{batch.id}_{view}_{instant}"})
                    instant_logs.append(instant_log)

                    if self.num_figures < 25:
                        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
                        ax1.imshow(data.img_proc[i].cpu().squeeze())
                        ax2.imshow(data.gt_proc[i].cpu().squeeze())
                        ax3.imshow(pred[i].squeeze())
                        plt.savefig(f'sample_{self.num_figures}.png')
                        plt.close()
                        self.num_figures += 1

                view_group = patient_group.create_group(view)
                view_group.create_dataset(CamusTags.gt, data=data.gt_proc.cpu(), **seg_save_options)
                view_group.create_dataset(CamusTags.pred, data=pred, **seg_save_options)

        patient_logs = self.reduce(pd.DataFrame(patient_logs))
        patient_logs.update({"PatientID": batch.id})

        return patient_logs, instant_logs

    def compute_instant_metrics(self, pred, gt, voxelspacing) -> Dict:
        """Computes binary segmentation metrics on one label.

        Args:
            pred: ndarray of prediction
            gt: ndarray of prediction
            voxelspacing: ndarray of voxelspacing

        Returns:
            Dictionary of results.
        """
        metrics = {}
        for label in self.hparams.data_params.labels:
            pred_mask, gt_mask = np.isin(pred, label.value), np.isin(gt, label.value)
            # Compute the reconstruction accuracy metrics
            metrics.update(
                {f"{label}_{score}": score_fn(pred_mask, gt_mask) for score, score_fn in self.scores.items()}
            )

            # Compute the distance metrics (that require the images' voxelspacing)
            # only if the requested label is present in both result and reference
            if np.any(pred_mask) and np.any(gt_mask):
                metrics.update(
                    {
                        f"{label}_{dist}": dist_fn(pred_mask, gt_mask, voxelspacing=voxelspacing[1:])
                        for dist, dist_fn in self.distances.items()
                    }
                )
            # Otherwise mark distances as NaN for this item
            else:
                metrics.update({f"{label}_{distance}": np.NaN for distance in self.distances})

        for score in list(self.scores.keys()) + list(self.distances.keys()):
            metrics[score] = np.mean([res for key, res in metrics.items() if score in key and "bg" not in key])

        try:
            validity = check_segmentation_validity(pred, voxelspacing[1:])
        except:
            validity = False
        metrics.update({"anatomical_errors": float(not validity)})
        metrics.update({"anatomical_validity": int(validity)})

        return metrics

    def test_epoch_end(self, outputs: List[Any]) -> None:
        """Aggregate logs from all test_step calls.

        Args:
            outputs: List of logs from each call from test_step
        """
        patient_logs, instant_logs = zip(*outputs)

        patient_logs = pd.DataFrame(patient_logs).set_index("PatientID")
        patient_reductions = self.reduce(patient_logs, index="PatientID")
        patient_logs.loc["reduction"] = patient_reductions
        patient_logs.to_csv(self.log_dir / "patient_results.csv")

        instant_logs = [item for sublist in instant_logs for item in sublist]
        instant_logs = pd.DataFrame(instant_logs).set_index("InstantID")
        instant_logs.to_csv(self.log_dir / "instant_results.csv")

        self.log_dict(prefix(patient_reductions, "test_"), **self.val_log_kwargs)
