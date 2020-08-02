from .loss import *
from .optimizer import *
from .scheduler import *

LOSS_DICT = {"cross_entropy": cross_entropy}
OPTIMIZER_DICT = {"adam": Adam}
SCHEDULER_DICT = {"early_stopping": EarlyStopping, "reduce_lr": ReduceLR, "reduce_lr1": reduce_lr_builtin}
