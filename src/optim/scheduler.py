import logging
import torch
import numpy as np
from torch.optim import lr_scheduler as lrs


class EarlyStopping:
    def __init__(self, patience, mode="max", delta=0.0001, min_score=None):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        if min_score is None:
            if self.mode == "min":
                self.val_score = np.inf
            else:
                self.val_score = -np.inf
        else:
            self.val_score = min_score
        logging.info("Initializing Early Stopping. Previous best score = {}".format(self.val_score))

    def __call__(self, epoch, epoch_score, model, optimizer, schedulers, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score == None:
            self.best_score = score
            self.save_checkpoint(
                epoch=epoch,
                epoch_score=epoch_score,
                model=model,
                optimizer=optimizer,
                schedulers=schedulers,
                model_path=model_path,
            )

        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter > self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(
                epoch=epoch,
                epoch_score=epoch_score,
                model=model,
                optimizer=optimizer,
                schedulers=schedulers,
                model_path=model_path,
            )
            self.counter = 0

    def save_checkpoint(self, epoch, epoch_score, model, schedulers, optimizer, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            logging.info(f"Validation score improved - {self.val_score} --> {epoch_score}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": [sch.state_dict() for sch in schedulers],
                    "epoch_score": epoch_score,
                },
                f=f"{model_path}/best.pkl",
            )
        self.val_score = epoch_score


class ReduceLR(object):
    def __init__(
        self,
        optimizer,
        patience,
        mode="min",
        factor=0.1,
        min_lr=0,
        eps=1e-8,
        threshold=1e-4,
        threshold_mode="rel",
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.counter = 0
        self.mode = mode
        self.best = None
        self.last_epoch = 0
        self.mode_worse = None
        self.last_epoch = 0
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        if self.factor >= 1.0:
            raise ValueError("Factor should be < 1.0")

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(self.optimizer.param_groups):
                raise ValueError(
                    "Expected {} min_lrs, got {}".format(len(self.optimizer.param_groups), len(min_lr))
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(self.optimizer.param_groups)

        logging.info(
            "Initializing ReduceLR. Original LR - {}".format(
                [group["lr"] for group in self.optimizer.param_groups]
            )
        )

        self._init_is_better(self.mode, self.threshold, self.threshold_mode)
        self._reset()

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Check score
        if self.is_better(current, self.best):
            self.best = current
            self.counter = 0
        else:
            self.counter -= -1
            logging.info("ReduceLR counter: {}/{}".format(self.counter, self.patience))

        # Update lr
        if self.counter > self.patience:
            self._reduce_lr(epoch)
            self.counter = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("Mode" + mode + " is unknown")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("Threshold mode" + threshold_mode + " is unknown")

        if mode == "min":
            self.mode_worse = np.inf
        else:
            self.mode_worse = -np.inf

        self.mode = mode
        self.threshold_mode = threshold_mode

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 + self.threshold
            return a > best * rel_epsilon

        else:
            return a > best * rel_epsilon

    def _reset(self):
        self.best = self.mode_worse
        self.counter = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            param_group["lr"] = new_lr
            logging.info("Epoch {}: Reducing learning rate of group {} to {:.4e}.".format(epoch, i, new_lr))

    def __str__():
        return "ReduceLR"


def reduce_lr_builtin(optimizer, patience, factor=0.1):
    return lrs.ReduceLROnPlateau(optimizer=optimizer, factor=factor, patience=patience, verbose=True)

