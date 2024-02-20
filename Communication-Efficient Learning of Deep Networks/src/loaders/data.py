import os
import gc
import torch
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
    def _construct_dataset(raw_train, idx, sample_indices):
        subset = Subset(raw_train, sample_indices)

        training_set, test_set = stratified_split(subset, args.test_size)

        traininig_set = SubsetWrapper(
            training_set, f'< {str(idx).zfill(8)} > (train)')
        if len(subset) * args.test_size > 0:
            test_set = SubsetWrapper(
                test_set, f'< {str(idx).zfill(8)} > (test)')
        else:
            test_set = None
        return (traininig_set, test_set)

    raw_train, raw_test = None, None

    split_map, client_datasets = None, None

    transforms = [None, None]

    transforms = [_get_transform(args), _get_transform(args)]
    raw_train, raw_test = fetch_dataset(args=args, transforms=transforms)

    if args.eval_type == 'local':
        if args.test_size == -1:
            assert raw_test is not None
        raw_test = None
    # else:
    #     if raw_test is None:
    #         err = f'[LOAD] Dataset `{args.dataset.upper()}` does not support pre-defined validation/test set, which can be used for `global` evluation... please check! (current `eval_type`=`{args.eval_type}`)'
    #         logger.exception(err)
    #         raise AssertionError(err)

    if split_map is None:
        logger.info(
            f'[SIMULATION] Distribution du dataset en utilisant le stratégie : `{args.split_type.upper()}`)!')
        split_map = split(args, raw_train)
        logger.info(
            f'[SIMULATION] ...Fin de la distribution avec la strategie : `{args.split_type.upper()}`)!')

    if client_datasets is None:
        logger.info(f'[SIMULATION] Creation du dataset pour les clients !')
        
        client_datasets = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.K, os.cpu_count() - 1)) as workhorse:
            for idx, sample_indices in TqdmToLogger(
                enumerate(split_map.values()),
                logger=logger,
                desc=f'[SIMULATION] ...creating client datasets... ',
                total=len(split_map)
            ):
                # for idx, sample_indices in enumerate(split_map.items()):
                #     logger.info("[SIMULATION] ...Creation du dataset client...")
                    client_datasets.append(workhorse.submit( _construct_dataset, raw_train, idx, sample_indices).result())
        logger.info(f'[SIMULATION] ...Creation du dataset client termine')

        # when if assigning pre-defined test split as a local holdout set (just divided by the total number of clients)
        # if (args.eval_type == 'local') and (args.test_size == -1):
        #     holdout_sets = torch.utils.data.random_split(
        #         _raw_test, [int(len(_raw_test) / args.K) for _ in range(args.K)])
        #     holdout_sets = [SubsetWrapper(
        #         holdout_set, f'< {str(idx).zfill(8)} > (test)') for idx, holdout_set in enumerate(holdout_sets)]
        #     augmented_datasets = []
        #     for idx, client_dataset in enumerate(client_datasets):
        #         augmented_datasets.append(
        #             (client_dataset[0], holdout_sets[idx]))
        #     client_datasets = augmented_datasets
    gc.collect() # On vide la memoire en faisant passer le Garbage Collector
    return raw_test, client_datasets
