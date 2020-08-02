import os
import pprint
import logging
import argparse
import pandas as pd

import torch
from torch.cuda import amp
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

import src.engine as engine
import src.utils.augment as aug
import src.utils.logger as log
import src.utils.params as params

from src.models import CLS_DICT
from src.datasets import IntelImages
from src.optim import OPTIMIZER_DICT, SCHEDULER_DICT, LOSS_DICT


def create_args():
    parser = argparse.ArgumentParser(description="Pytorch Boiler Plate for CV")
    parser.add_argument("run_conf", metavar="--hp_conf", help="Path to hyper parameter file")

    return parser


def main(run_params):
    # Load params
    run_params = params.load_yaml(str(run_params))

    # Set hp dicts
    experiment = run_params["experiment"]
    model_dict = run_params["model"]
    paths = run_params["paths"]
    optim_dict = run_params["optimizer"]
    sch_dict = run_params["schedulers"]

    # Seed
    SEED = experiment["seed"]
    torch.manual_seed(SEED)

    # Set paths
    log_path = os.path.join(paths["log_path"], model_dict["type"], f"exp_{experiment['exp']}")
    tb_log_path = os.path.join(paths["tb_log_path"], model_dict["type"], f"exp_{experiment['exp']}",)
    model_path = os.path.join(
        paths["model_path"], model_dict["type"], f"exp_{experiment['exp']}", f"run_{experiment['run']}",
    )

    # Create paths in file structure
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(tb_log_path, exist_ok=True)
    tb_log = os.path.join(tb_log_path, f"run_{experiment['run']}")
    run_log = os.path.join(log_path, f"run_{experiment['run']}.log")

    # File/Dir exists check
    if experiment["check_log"]:
        if os.path.isdir(tb_log) or os.path.isfile(run_log):
            raise ValueError(
                "Logs for Run - {} has been completed. Please set check_log to false to rerun"
                " or increment run".format(experiment["run"])
            )

    # Set running logger and tb_logger
    log.set_logging(log_file=run_log)
    writer = SummaryWriter(log_dir=tb_log)

    # Log params
    logging.info("Parameters used for this run - \n{}".format(pprint.pformat(run_params, sort_dicts=False)))

    # Define Data Loaders
    logging.info("Initializing Dataloaders ")
    train_dl = data.DataLoader(
        IntelImages(pd.read_csv(paths["train_csv"]), transforms=aug.train_aug),
        shuffle=True,
        batch_size=experiment["train_batch_size"],
    )
    valid_dl = data.DataLoader(
        IntelImages(pd.read_csv(paths["valid_csv"]), transforms=aug.train_aug),
        shuffle=True,
        batch_size=experiment["valid_batch_size"],
    )
    logging.info("Initialized Dataloaders ")

    # Check checkpoint
    if model_dict["checkpoint"] is not None:
        load_cpt = True
        cpt_dict = params.load_checkpoint(model_dict["checkpoint"])
        epoch_score = cpt_dict["epoch_score"]
        start_epoch = cpt_dict["epoch"]
        num_epochs = experiment["num_epochs"] + start_epoch

    else:
        load_cpt = False
        epoch_score = None
        start_epoch = 0
        num_epochs = experiment["num_epochs"]

    # Load model
    model = CLS_DICT[model_dict["type"]](**model_dict["params"])
    # Checkpoint
    if load_cpt:
        model.load_state_dict(cpt_dict["model_state_dict"])
    model.to(device=experiment["device"])

    # Write model to tensorboard
    if not os.path.isdir(f"logs/tensorboard/models/{model_dict['type']}"):
        model_writer = SummaryWriter(f"logs/tensorboard/models/{model_dict['type']}")
        model_writer.add_graph(model, input_to_model=torch.zeros([3, 3, 64, 64]))

    # Load optimizer
    optimizer = OPTIMIZER_DICT[optim_dict["type"]](model.parameters(), **optim_dict["params"])
    # Checkpoint
    if load_cpt:
        optimizer.load_state_dict(cpt_dict["optim_state_dict"])
        if experiment["device"] == "cuda":
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    # Load loss function
    loss_fn = LOSS_DICT[run_params["loss"]]

    # Load Scheduler(s)
    schedulers = list()
    for i, sch in enumerate(sch_dict.keys()):
        schedulers.append(SCHEDULER_DICT[sch](optimizer, **sch_dict[sch]))
        logging.info("Initialized scheduler {}.".format(SCHEDULER_DICT[sch].__str__()))
    # Checkpoint
    if load_cpt:
        for i, sch in enumerate(schedulers):
            sch.load_state_dict(cpt_dict["scheduler_state_dict"][i])

    # Load Early Stopping
    es = SCHEDULER_DICT["early_stopping"](**run_params["early_stopping"], min_score=epoch_score)
    logging.info("Initialized Early Stopping")

    # Set scaler
    scaler = amp.GradScaler()

    # Train and Validate
    logging.info("Start Training")
    for epoch in range(start_epoch + 1, num_epochs + 1):
        logging.info("Epoch - {}/{}".format(epoch, num_epochs))
        print("Epoch - {}/{}".format(epoch, num_epochs))

        # Train
        train_metrics = engine.train_one_epoch(
            data_loader=train_dl, model=model, optimizer=optimizer, loss_fn=loss_fn, scaler=scaler
        )
        logging.info(f"Training Metrics - {train_metrics}")

        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)

        # Eval
        valid_metrics = engine.eval_one_epoch(data_loader=valid_dl, model=model, loss_fn=loss_fn)
        logging.info(f"Validation Metrics - {valid_metrics}")

        writer.add_scalar("valid/loss", valid_metrics["loss"], epoch)
        writer.add_scalar("valid/accuracy", valid_metrics["accuracy"], epoch)

        for sch in schedulers:
            if isinstance(sch, SCHEDULER_DICT["reduce_lr"]):
                sch.step(valid_metrics["loss"])
            else:
                sch.step()
        es(epoch, valid_metrics["loss"], model, optimizer, schedulers, model_path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": [sch.state_dict() for sch in schedulers],
                "epoch_score": epoch_score,
            },
            f=f"{model_path}/last.pkl",
        )


if __name__ == "__main__":

    parser = create_args()
    args = parser.parse_args()

    main(args.run_conf)
