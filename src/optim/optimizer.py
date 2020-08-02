import torch.optim as optim


def Adam(params, lr, weight_decay=0):
    return optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
