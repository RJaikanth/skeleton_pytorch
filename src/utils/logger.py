import os
import logging
import torch.nn as nn


trainable_layers = (nn.Conv2d, nn.BatchNorm2d, nn.Linear)
non_trainable_layers = (
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveMaxPool2d,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.Dropout,
    nn.ReLU,
)


def set_logging(log_file):
    logging.basicConfig(
        filename=log_file, filemode="w", format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO
    )


def log_grads(model, batch=None):
    for i, layer in enumerate(model.modules()):

        if isinstance(layer, non_trainable_layers):
            continue

        elif isinstance(layer, trainable_layers):
            # For weights
            if layer.weight.grad is not None:
                min_weight = abs(layer.weight.grad.min().item())
                max_weight = layer.weight.grad.max().item()

                # Log if diminishing or exploding
                if min_weight < 1e-15:
                    logging.info(
                        "Weight Gradient diminishing during Batch - {} at layer ({} - {})."
                        "Minimum weight at layer - {}".format(batch, i, layer._get_name(), min_weight)
                    )
                if max_weight > 5:
                    logging.info(
                        "Weight Gradient exploding during Batch - {} at layer ({} - {})."
                        "Minimum weight at layer - {}".format(batch, i, layer._get_name(), max_weight)
                    )

            # For Bias
            if layer.bias.grad is not None:
                min_weight = abs(layer.bias.grad.min().item())
                max_weight = layer.bias.grad.max().item()
                # Log if diminishing or exploding
                if min_weight < 1e-15:
                    logging.info(
                        "Bias Gradient diminishing during Batch - {} at layer ({} - {})."
                        " Minimum weight at layer - {}".format(batch, i, layer._get_name(), min_weight)
                    )
                if max_weight > 5:
                    logging.info(
                        "Bias Gradient exploding during Batch - {} at layer ({} - {})."
                        "Minimum weight at layer - {}".format(batch, i, layer._get_name(), max_weight)
                    )
