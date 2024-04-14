import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from .fedavgclient import FedavgClient
from src import MetricManager

import logging
logger = logging.getLogger(__name__)


class FedsecureClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedsecureClient, self).__init__(**kwargs)
        self.eps = {}

        # Ajoutez un indicateur pour suivre si l'initialisation a été effectuée
        self.initialized = False

    def initsecure(self):
        self.privacy_engine = PrivacyEngine()
        # Vérifiez d'abord si l'initialisation a déjà été effectuée
        if not self.initialized:
            self._model = self.model
            self._model.train()
            self._train_loader = self.train_loader
            self._optimizer = self.optim(
                self._model.parameters(), **self._refine_optim_args(self.args))

            self.delta = 1 / (1.1*len(self.training_set))

            self.client_model, self.client_optimizer, self.client_train_loader = self.privacy_engine.make_private(
                module=self._model,
                optimizer=self._optimizer,
                data_loader=self._train_loader,
                noise_multiplier=1.1,
                max_grad_norm=1.0)

            self.client_model.train(mode=True)

            # Marquez l'initialisation comme complétée
            self.initialized = True

    def update(self, global_round):
        # Appelez initsecure pour vous assurer que tout est initialisé
        self.initsecure()

        mm = MetricManager(self.args.eval_metrics)

        with BatchMemoryManager(data_loader=self.client_train_loader,
                                max_physical_batch_size=self.args.max_physical_batch_size,
                                optimizer=self.client_optimizer) as memory_safe_data_loader:

            for e in range(self.args.E):
                for inputs, targets in memory_safe_data_loader:

                    self.client_optimizer.zero_grad()

                    inputs, targets = inputs.to(
                        self.args.device), targets.to(self.args.device)

                    outputs = self.client_model(inputs)
                    loss = self.criterion()(outputs, targets)

                    for param in self.client_model.parameters():
                        param.grad = None
                    loss.backward()

                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.client_model.parameters(), self.args.max_grad_norm)

                    self.client_optimizer.step()

                    mm.track(loss.item(), outputs, targets)
                else:
                    mm.aggregate(len(self.training_set), e + 1)
            else:
                self.client_model.to('cpu')
        return mm.results
