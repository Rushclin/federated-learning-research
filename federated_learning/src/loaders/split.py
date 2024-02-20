import numpy as np


def split(args, dataset):
    """

    """

    # IID (statistical homogeneity)
    if args.split_type == "iid":

        shuffled_indices = np.random.permutation(len(dataset))

        split_indices = np.array_split(shuffled_indices, args.K)

        # On construit une hash map
        split_map = {k: split_indices[k] for k in range(args.K)}
        return split_map
