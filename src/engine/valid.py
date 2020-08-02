import logging
from tqdm import tqdm
from torch.cuda import amp

import src.utils.meters as agg
import src.utils.logger as log
import src.utils.metric as metrics


def eval_one_epoch(data_loader, model, loss_fn, device="cuda"):
    # Load model to device and set to train mode
    model = model.to(device)
    model.eval()

    # Set average meters
    losses = agg.AverageMeter()
    accuracy = agg.AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader), unit="batches")
    for batch, (images, labels) in enumerate(tk0):
        # Load input on device
        images = images.to(device)
        labels = labels.to(device)

        # Predict
        out = model(images)

        # Calculate metrics
        loss = loss_fn(true=labels, pred=out)
        acc = metrics.accuracy_fn(true=labels, pred=out)

        # Update aggregators
        losses.update(loss.item(), data_loader.batch_size)
        accuracy.update(acc, data_loader.batch_size)

        tk0.set_postfix(valid_loss=losses.avg, valid_accuracy=accuracy.avg)

    dict_ = {"loss": losses.avg, "accuracy": accuracy.avg}

    return dict_
