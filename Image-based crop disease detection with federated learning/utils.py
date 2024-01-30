import os
import shutil
import random

from models import DenseNet121, InceptionV3, VGG16, ViT_B16, ViT_B32, ResNet50

def get_model(model_str: str, num_classes: int, device: str = "cpu"):
    if model_str == "ResNet50":
        return ResNet50(num_classes=num_classes, pretrained=True, device=device)
    elif model_str == "DenseNet121":
        return DenseNet121(num_classes)
    elif model_str == "InceptionV3":
        return InceptionV3(num_classes)
    elif model_str == "VGG16":
        return VGG16(num_classes)
    elif model_str == "ViT_B16":
        return ViT_B16(num_classes)
    else:
        return ViT_B32(num_classes)


def split_dataset(input_folder, output_folder, train_ratio=0.8):
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

input_dataset_folder = r"./dataset/"
output_data_folder = r"./data/"

# split_dataset(input_dataset_folder, output_data_folder)
