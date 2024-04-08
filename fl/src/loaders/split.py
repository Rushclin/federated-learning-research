import logging
import numpy as np

logger = logging.getLogger(__name__)


def split(args, dataset):
    """
    Découpage du dataset

    """

    if args.split_type == "iid":

        shuffled_indices = np.random.permutation(len(dataset))

        split_indices = np.array_split(shuffled_indices, args.K)

        # On construit une hash map
        split_map = {k: split_indices[k] for k in range(args.K)}
        return split_map
    if args.split_type == 'non-iid':

        shuffled_indices = np.random.permutation(len(dataset))

        split_indices = np.array_split(shuffled_indices, args.K)

        keep_ratio = np.random.uniform(
            low=0.95, high=0.99, size=len(split_indices))

        split_indices = [indices[:int(len(indices) * ratio)]
                         for indices, ratio in zip(split_indices, keep_ratio)]

        split_map = {k: split_indices[k] for k in range(args.K)}
        return split_map
