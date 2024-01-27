import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import random

import warnings
import os

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# Create DateFrame from dataset
def define_paths(data_dir: str):
    """
    Fonction qui doit prendre un chemin contenant les images et renvoyer les chemins d'accès y compris les différents labels
    """
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)

    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)

        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)

    return filepaths, labels


def define_df(files, classes):
    """
    Construction du datasets 
    """
    Fseries = pd.Series(files, name='filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis=1)


def split_data(data_dir: str):
    """
    Division de notre Dataset en trois (03), entrainement, validation et test
    """

    # Construction du Dataset d'entrainement
    # 80% pour l'entrainement et 20% pour le reste
    files, classes = define_paths(data_dir)
    df = define_df(files, classes)
    strat = df['labels']
    train_df, rest = train_test_split(
        df,  train_size=0.8, shuffle=True, random_state=123, stratify=strat)

    # 50% donc 10% pour la validation et le reste pour le test
    strat = rest['labels']
    valid_df, test_df = train_test_split(  # Utilise la strtification base sur les labels pour le decoupage
        rest,  train_size=0.5, shuffle=True, random_state=123, stratify=strat)  # random_state = 123 ici nous permet de produire le meme resultat quelque soit les parametres

    return train_df, valid_df, test_df


if __name__ == "__main__":
    train_df, valid_df, test_df = split_data(r"./../data")

    # Test d'affichage des images 
    filespaths, labels = define_paths(r"./../data")

    random_indices = random.sample(range(len(filespaths)), 6)

    for i, idx in enumerate(random_indices):
        image = Image.open(filespaths[idx])
        plt.subplot(2, 3, i+1)
        plt.imshow(image)
        plt.title(f"{labels[idx]}")
    
    plt.show()