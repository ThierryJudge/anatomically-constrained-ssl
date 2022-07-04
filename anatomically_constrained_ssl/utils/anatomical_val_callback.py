import pytorch_lightning as pl
import torch
from typing import Dict, Optional, Tuple

import logging

log = logging.getLogger(__name__)


class EarlyStoppingWithDivergenceWaitCount(pl.callbacks.early_stopping.EarlyStopping):
    """
    Override Pytorch Lightning early stopping callback to support wait count on divergence threshold
    """
    def __init__(
            self,
            monitor: Optional[str] = None,
            min_delta: float = 0.0,
            patience: int = 3,
            verbose: bool = False,
            mode: str = "min",
            strict: bool = True,
            check_finite: bool = True,
            stopping_threshold: Optional[float] = None,
            divergence_threshold: Optional[float] = None,
            check_on_train_epoch_end: Optional[bool] = None,
            div_patience: int = 10,
    ):
        self.wait_count_div = 0
        self.div_patience = div_patience
        super().__init__(monitor, min_delta, patience, verbose, mode, strict, check_finite,
                         stopping_threshold, divergence_threshold, check_on_train_epoch_end)

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, str]:
        """
        Override _evaluate_stopping_criteria for divergence treshold with patience
        Args:
           current: metric to be evaluated
        Returns:
           should_stop (bool): does training stop at this step
           reason (str): reason for the stop
        """

        should_stop = False
        reason = None

        if self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            self.wait_count_div += 1
            print(f"Metric fell below threshold {self.divergence_threshold}, at {current.item()}")
            print(f"Wait count: {self.wait_count_div}/{self.div_patience}")
            if self.wait_count_div >= self.div_patience:
                should_stop = True
                reason = (
                    "Divergence threshold reached with patience:"
                    f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                    " Signaling Trainer to stop."
                )
        elif self.wait_count_div is not 0 and not self.monitor_op(-current, -self.divergence_threshold):
            # reset wait_cout for divergence to zero if metric went back above threshold
            self.wait_count_div = 0

        return should_stop, reason

    def _validate_condition_metric(self, logs: Dict[str, float]) -> bool:
        """
        Override _validate_condition_metric to avoid raising execption on metric that doesn't exist yet
        Args:
            logs: Logged metrics
        Returns:
            (bool): Does monitored metric exist
        """

        monitor_val = logs.get(self.monitor)

        error_msg = (
            f"Early stopping conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `EarlyStopping` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            return False

        return True
