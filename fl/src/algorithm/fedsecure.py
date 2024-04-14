from .fedavg import FedavgOptimizer


class FedsecureOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FedsecureOptimizer, self).__init__(params=params, **kwargs)
