import numpy as np


class Meter(object):
    """
    Class to record aggregated values over each iteration, epoch and run     

    Args:
        iteration_aggregator (class 'X'Meter): Aggregator over each iteration (MiniBatches) 
        epoch_aggregator (class 'X'Meter): Aggregator over each epoch
        run_aggregator (class 'X'Meter): Aggregator over full run
    """

    def __init__(self, iteration_aggregator, epoch_aggregator, run_aggregator):
        self.run_aggregator = run_aggregator
        self.iteration_aggregator = iteration_aggregator
        self.epoch_aggregator = epoch_aggregator

    def record(self, val, n=1):
        """ Record iteration value """
        self.iteration_aggregator.record(val, n=n)

    def get_iteration(self):
        """ Get iteration value """
        v, n = self.iteration_aggregator.get_val()
        return v

    def reset_iteration(self):
        """ Reset iteration value """
        v, n = self.iteration_aggregator.get_data()
        self.iteration_aggregator.reset()
        if v is not None:
            self.epoch_aggregator.record(v, n=n)

    def get_epoch(self):
        """ Get epoch value """
        v, n = self.epoch_aggregator.get_val()
        return v

    def reset_epoch(self):
        """ Reset epoch value """
        v, n = self.epoch_aggregator.get_data()
        self.epoch_aggregator.reset()
        if v is not None:
            self.run_aggregator.record(v, n=n)

    def get_run(self):
        """ Get run value """
        v, n = self.run_aggregator.get_val()
        return v

    def reset_run(self):
        """ Reset run value """
        self.run_aggregator.reset()


class QuantileMeter(object):
    """
    Class to aggregate values over a quartile.

    Args:
        q (int): Quartil value
    """

    def __init__(self, q):
        self.q = q
        self.reset()

    def reset(self):
        """ Reset the Meter """
        self.vals = []
        self.n = 0

    def record(self, val, n=1):
        """ Record the value """
        if isinstance(val, list):
            self.vals += val
            self.n += len(val)
        else:
            self.vals += [val] * n
            self.n += n

    def get_val(self):
        """ Get Quartile """
        if not self.vals:
            return None, self.n
        return np.quantile(self.vals, self.q, interpolation="nearest"), self.n

    def get_data(self):
        """ Get Data """
        return self.vals, self.n


class MaxMeter(object):
    """ Class to aggregate and record max values """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset the Meter """
        self.max = None
        self.n = 0

    def record(self, val, n=1):
        """ Record the value """
        if self.max is None:
            self.max = val
        else:
            self.max = max(self.max, val)
        self.n = n

    def get_val(self):
        """ Get Max """
        return self.max, self.n

    def get_data(self):
        """ Get Data """
        return self.max, self.n


class MinMeter(object):
    """ Class to aggregate and record min values """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset the Meter """
        self.min = None
        self.n = 0

    def record(self, val, n=1):
        """ Record the value """
        if self.min is None:
            self.min = val
        else:
            self.min = min(self.min, val)
        self.n = n

    def get_val(self):
        """ Get Min """
        return self.min, self.n

    def get_data(self):
        """ Get Data """
        return self.min, self.n


class LastMeter(object):
    """ Class that records last value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset the Meter """
        self.last = []
        self.n = 0

    def record(self, val, n=1):
        """ Record the value """
        self.last = val
        self.n = n

    def get_val(self):
        """ Get Last """
        return self.last, self.n

    def get_data(self):
        """ Get Data """
        return self.last, self.n


class AverageMeter(object):
    """ Class to aggregate and record average values """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset the Meter """
        self.avg = 0
        self.n = 0

    def record(self, val, n=1):
        """ Record the value """
        self.n += n
        self.avg *= val * n

    def get_val(self):
        """ Get Average """
        if self.n == 0:
            return None, 0
        return self.avg / self.n, self.n

    def get_data(self):
        """ Get Data """
        if self.n == 0:
            return None, 0
        return self.avg / self.n, self.n


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
