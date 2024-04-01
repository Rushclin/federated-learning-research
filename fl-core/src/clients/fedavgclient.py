import copy
import inspect
import torch
import itertools
from torch.utils.data import DataLoader

from src import MetricManager

from .clientbase import BaseClient


class FedavgClient(BaseClient):
    def __init__(self, args, training_set, test_set):
        super(FedavgClient, self).__init__()

        self.args = args  # La liste des arguments du FL
        self.training_set = training_set  # Le dataset d'entrainement du Client
        self.test_set = test_set  # Le dataset de test ou de validation du Client

        # On vérifie bien que l'optimisateur passé en est un existant `SGD` par exemple
        self.optim = torch.optim.__dict__[self.args.optimizer]
        # La fonction de perte aussi, on vérifie que ca existe bien.
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_loader = self._create_dataloader(
            self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(
            self.test_set, shuffle=False)

    def _collect_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # Collecte les arguments necessaires
        all_args = {}
        for argument in required_args:
            if hasattr(args, argument):
                all_args[argument] = getattr(args, argument)
        return all_args

    def _create_dataloader(self, dataset, shuffle):
        if self.args.B == 0:
            self.args.B = len(self.training_set)
        return DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)

    def update(self):
        metrics = MetricManager(self.args.eval_metrics)
       
        self.model.train()
        self.model.to(self.args.device)

        optimizer = self.optim(self.model.parameters(),
                               **self._collect_args(self.args))
        
        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(
                    self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                for param in self.model.parameters():
                    param.grad = None
                loss.backward()

                optimizer.step()

                metrics.track(loss.item(), outputs, targets)
            else:
                metrics.aggregate(len(self.training_set), e + 1)
        else:
            self.model.to('cpu')
        return metrics.results

    @torch.inference_mode()  # Activation du mode de test
    def evaluate(self):

        metrics = MetricManager(self.args.eval_metrics)

        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(
                self.args.device), targets.to(self.args.device)

            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            metrics.track(loss.item(), outputs, targets)
        else:
            self.model.to('cpu')
            metrics.aggregate(len(self.test_set))
        return metrics.results

    def download(self, model):
        self.model = copy.deepcopy(model)

    def upload(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print("Taille des poids du modèle : {:.2f} Mo".format(total_params * 4 / (1024 ** 2)))  # Conversion des octets en Mo
        return itertools.chain.from_iterable([self.model.named_parameters(), self.model.named_buffers()])

    def __len__(self):
        return len(self.training_set)

    def __repr__(self): # Pour formater lorsqu'on doit vouloir imprimer la classe FedavgClient
        return f'CLIENT < {self.id} >'
