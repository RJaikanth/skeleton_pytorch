import logging
from tqdm import tqdm
from torch.cuda import amp

import src.utils.meters as agg
import src.utils.logger as log
import src.utils.metric as metrics


def train_one_epoch(data_loader, model, optimizer, loss_fn, device="cuda", log_grad=True, scaler=None):
    model.train()

    # for k in optimizer.state_dict().keys():
    print(type(optimizer.state_dict()["state"]))
    # print(type(optimizer.state_dict()["param_groups"]))
    # break

    # Set average meters
    losses = agg.AverageMeter()
    accuracy = agg.AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader), unit="batches")
    for batch, (images, labels) in enumerate(tk0):
        # Load input on device
        images = images.to(device)
        print(images.device)
        labels = labels.to(device)

        # Train Step With AMP
        if scaler is not None:
            if device != "cuda":
                raise Exception("Use GPU for AMP")
            with amp.autocast():
                # Forward Pass
                out = model(images)
                loss = loss_fn(true=labels, pred=out)
                # Backward Pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Train Step Without AMP
        else:
            # Forward Pass
            out = model(images)
            loss = loss_fn(true=labels, pred=out)
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log gradients
        if log_grad:
            log.log_grads(model, batch)

        # Calculate metrics
        acc = metrics.accuracy_fn(true=labels, pred=out)

        # Update aggregators
        losses.update(loss.item(), data_loader.batch_size)
        accuracy.update(acc, data_loader.batch_size)

        tk0.set_postfix(train_loss=losses.avg, train_accuracy=accuracy.avg)

    dict_ = {"loss": losses.avg, "accuracy": accuracy.avg}

    return dict_
