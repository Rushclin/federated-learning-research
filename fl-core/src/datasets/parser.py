import os
import shutil
import random
import logging

from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)


def split_dataset(input_folder: str, output_folder: str, train_ratio=0.8) -> None:
    """
    Organiser le dataset en deux groupes, Entraînnement et Test
    """

    logger.info("[LOAD] Organisation des dossiers du dataset")

    if os.path.exists(output_folder):
        logger.info(
            "[LOAD] Le dossier de sortie existe déjà. Aucune opération n'est effectuée.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)

        if not os.path.isdir(class_path):
            continue

        all_files = os.listdir(class_path)

        num_train = int(len(all_files) * train_ratio)
        train_files = random.sample(all_files, num_train)
        validation_files = [
            file for file in all_files if file not in train_files]

        train_path = os.path.join(output_folder, "train", class_folder)
        validation_path = os.path.join(
            output_folder, "validation", class_folder)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(validation_path, exist_ok=True)

        for file in train_files:
            shutil.copy(os.path.join(class_path, file),
                        os.path.join(train_path, file))

        for file in validation_files:
            shutil.copy(os.path.join(class_path, file),
                        os.path.join(validation_path, file))

    logger.info("[LOAD] Fin organisation des dossiers du dataset")


def fetch_dataset(args, transforms):
    logger.info(f'[LOAD] Chargement du dataset!')

    split_dataset(input_folder=args.input_folder,
                  output_folder=args.output_folder, train_ratio=args.train_ratio)

    data_transform = {
        'train': transforms[0],
        'validation': transforms[1]
    }

    image_datasets = {
        'train':
        ImageFolder(f'{args.output_folder}/train',
                    data_transform['train']),
        'validation':
        ImageFolder(f'{args.output_folder}/validation',
                    data_transform['validation'])
    }

    logger.info(f'[LOAD] Fin chargement du dataset!')

    return image_datasets['train'], image_datasets['validation']
