import os
import shutil
import random

def split_dataset(input_folder: str, output_folder: str, train_ratio=0.8):
    # Si le dossier de sortie existe déjà, aucune opération n'est effectuée
    if os.path.exists(output_folder):
        print("Le dossier de sortie existe déjà. Aucune opération n'est effectuée.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)

        if not os.path.isdir(class_path):
            continue

        all_files = os.listdir(class_path)

        num_train = int(len(all_files) * train_ratio)
        train_files = random.sample(all_files, num_train)
        validation_files = [file for file in all_files if file not in train_files]

        train_path = os.path.join(output_folder, "train", class_folder)
        validation_path = os.path.join(output_folder, "validation", class_folder)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(validation_path, exist_ok=True)

        for file in train_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(train_path, file))

        for file in validation_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(validation_path, file))

