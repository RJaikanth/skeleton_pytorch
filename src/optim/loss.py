from torch import nn


def cross_entropy(true, pred):
    return nn.CrossEntropyLoss()(pred, true)


def l1_loss(true, pred):
    return nn.L1Loss()(pred, true)


def mse_loss(true, pred):
    return nn.MSELoss()(pred, true)


def ctc_loss(true, pred):
    return nn.CTCLoss()(pred, true)


def nll_loss(true, pred):
    return nn.NLLLoss()(pred, true)


def poisson_nll_loss(true, pred):
    return nn.PoissonNLLLoss()(pred, true)


def kld_loss(true, pred):
    return nn.KLDivLoss()(pred, true)


def bce_loss(true, pred):
    return nn.BCELoss()(pred, true)


def margin_rank_loss(true, pred):
    return nn.MarginRankingLoss()(pred, true)


def hinge_embedding_loss(true, pred):
    return nn.HingeEmbeddingLoss()(pred, true)


def multilabel_margin_loss(true, pred):
    return nn.MultiLabelMarginLoss()(pred, true)


def smooth_l1_loss(true, pred):
    return nn.SmoothL1Loss()(pred, true)
    

def soft_margin_loss(true, pred):
    return nn.SoftMarginLoss()(pred, true)
    

def multilabel_soft_margin_loss(true, pred):
    return nn.MultiLabelSoftMarginLoss()(pred, true)
    

def cosine_embedding_loss(true, pred):
    return nn.CosineEmbeddingLoss()(pred, true)


def multi_margin_loss(true, pred):
    return nn.MultiMarginLoss()(pred, true)
    

def triple_margin_loss(true, pred):
    return nn.TripletMarginLoss()(pred, true)
    