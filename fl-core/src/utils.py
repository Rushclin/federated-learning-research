import os
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from importlib import import_module
from collections import defaultdict
from torch.utils.data import Subset

logger = logging.getLogger(__name__)

########
# Seed #
########
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f'[SEED] ...seed modifié à : {seed}!')


    
###############
#     TQDM    #
###############
class TqdmToLogger(tqdm):
    def __init__(
            self, 
            *args, 
            logger=None, 
            mininterval=0.1, 
            bar_format='{desc:<}{percentage:3.0f}% |{bar:20}| [{n_fmt:6s}/{total_fmt}]', 
            desc=None, 
            **kwargs
        ):
        self._logger = logger
        super().__init__(*args, mininterval=mininterval, bar_format=bar_format, ascii=True, desc=desc, **kwargs)

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        return logger

    def display(self, msg=None, pos=None):
        if not self.n:
            return
        if not msg:
            msg = self.__str__()
        self.logger.info(msg)


##################
# Metric manager #
##################
class MetricManager:
    """
    """
    def __init__(self, eval_metrics):

        self.metric_funcs = {
            name: import_module(f'.metrics', package=__package__).__dict__[name.title()]()
            for name in eval_metrics
        }
        self.figures = defaultdict(int) 
        self._results = dict()

    def track(self, loss, pred, true):
        
        # Mise à jour 
        self.figures['loss'] += loss * len(pred)

        for module in self.metric_funcs.values():
            module.collect(pred, true)

    def aggregate(self, total_len, curr_step=None):
        running_figures = {name: module.summarize() for name, module in self.metric_funcs.items()}
        running_figures['loss'] = self.figures['loss'] / total_len
        if curr_step is not None:
            self._results[curr_step] = {
                'loss': running_figures['loss'], 
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
                }
        else:
            self._results = {
                'loss': running_figures['loss'], 
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
                }
        self.figures = defaultdict(int)

    @property
    def results(self):
        return self._results
    


##############################
# Vérification des arguments #
##############################
    
def check_args(args):
    # Vérification du périphérique d'exécution de l'algorithme
    if 'cuda' in args.device:
        if not torch.cuda.is_available():
            err = "Veuillez s'il vous plaît vérifiér que votre Terminal dispose d'un GPU NVIDIA" 
            logger.exception(err)
            raise AssertionError(err)

    # Vérification de l'optimisateur
    if args.optimizer not in torch.optim.__dict__.keys():
        err = f'`{args.optimizer}` Optimisateur non reconnu'
        logger.exception(err)
        raise AssertionError(err)
    
    # Vérification de la fonction de perte
    if args.criterion not in torch.nn.__dict__.keys():
        err = f'`{args.criterion}` Fonction de perte non reconnue'
        logger.exception(err)
        raise AssertionError(err)

    # Vérification de l'algorithme à exécuter
    if args.algorithm != ('fedavg' or 'fedprox'):
        err = "Le type d'agrégation passé n'est pas pris en charge !"
        logger.exception(err)
        raise AssertionError(err)

    return args

# Prend en entrée un ensemble de données brut (raw_dataset) ainsi qu'une taille de test (test_size). 
# Divise ensuite cet ensemble de données en un ensemble d'apprentissage et un ensemble de test tout 
# en préservant la distribution des étiquettes (ou des classes) dans les deux ensembles

def stratified_split(raw_dataset, test_size):
    indices_par_label = defaultdict(list)

    for index, label in enumerate(np.array(raw_dataset.dataset.targets)[raw_dataset.indices]):
        indices_par_label[label.item()].append(index)
    
    train_indices, test_indices = [], []

    for label, indices in indices_par_label.items():
        n_samples_for_label = round(len(indices) * test_size)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        test_indices.extend(random_indices_sample)
        train_indices.extend(set(indices) - set(random_indices_sample))

    return Subset(raw_dataset, train_indices), Subset(raw_dataset, test_indices)
