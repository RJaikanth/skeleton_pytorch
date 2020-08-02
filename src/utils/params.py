import os
import json
import yaml
import torch


def load_json(json_path):
    if not os.path.isfile(json_path):
        raise FileNotFoundError("{} does not exist".format(json_path))

    with open(json_path) as json_file:
        param_dict = json.load(json_file)

    return param_dict


def load_yaml(yaml_path):
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError("{} does not exist".format(yaml_path))

    with open(yaml_path) as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    return yaml_dict


def load_checkpoint(cpt_path):
    a = torch.load(f=cpt_path)
    # print(a["scheduler_state_dict"])
    return a


if __name__ == "__main__":
    load_checkpoint("weights/trained/alexnet/exp_0/last.pkl")
