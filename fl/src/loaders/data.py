import os
import gc
import logging
import concurrent.futures
from torch.utils.data import Subset, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src import TqdmToLogger, stratified_split
from .split import split
from src.datasets import *

logger = logging.getLogger(__name__)


class SubsetWrapper(Dataset):
    """Recreation du subset du Dataset
    """

    def __init__(self, subset, suffix):
        self.subset = subset
        self.suffix = suffix

    def __getitem__(self, index):
        inputs, targets = self.subset[index]
        return inputs, targets

    def __len__(self):
        return len(self.subset)

    def __repr__(self):
        return f'{repr(self.subset.dataset.dataset)} {self.suffix}'


def load_dataset(args):
    """
    Charger et diviser le dataset.

    Args:
        args: arguments

    Returns: 
        Le dataset de test 
        L'ensemble du dataset d'entrainement des clients
    """

    # Transformation des images
    def _get_transform(args):
        transform = Compose(
            [
                Resize((args.resize, args.resize)),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )
        return transform

    # Construction du Dataset
    def _construct_dataset(train_dataset, idx, sample_indices):
        subset = Subset(train_dataset, sample_indices)

        training_set, test_set = stratified_split(subset, args.test_size)

        traininig_set = SubsetWrapper(
            training_set, f'< {str(idx).zfill(8)} > (train)')
        if len(subset) * args.test_size > 0:
            test_set = SubsetWrapper(
                test_set, f'< {str(idx).zfill(8)} > (test)')
        else:
            test_set = None
        return (traininig_set, test_set)

    train_dataset, test_dataset = None, None

    split_map, client_datasets = None, None

    transforms = [None, None]

    transforms = [_get_transform(args), _get_transform(args)]
    train_dataset, test_dataset = fetch_dataset(args=args, transforms=transforms)

    if args.eval_type == 'local':
        if args.test_size == -1:
            assert test_dataset is not None
        test_dataset = None

    if split_map is None:
        logger.info(
            f'[SIMULATION] Distribution du dataset en utilisant le stratégie : `{args.split_type.upper()}`)!')
        split_map = split(args, train_dataset)
        logger.info(
            f'[SIMULATION] ...Fin de la distribution avec la strategie : `{args.split_type.upper()}`)!')

    if client_datasets is None:
        logger.info(f'[SIMULATION] Création du dataset pour les clients !')
        
        client_datasets = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.K, os.cpu_count() - 1)) as workhorse:
            for idx, sample_indices in TqdmToLogger(
                enumerate(split_map.values()),
                logger=logger,
                desc=f'[SIMULATION] ...Création du dataset Client... ',
                total=len(split_map)
            ):
                    client_datasets.append(workhorse.submit( _construct_dataset, train_dataset, idx, sample_indices).result())
        logger.info(f'[SIMULATION] ...Création du dataset client terminé')

    gc.collect() # On vide la mémoire en faisant passer le Garbage Collector
    return test_dataset, client_datasets
