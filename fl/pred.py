# import torch
# import argparse
# import numpy as np

# from PIL import Image
# import matplotlib.pylab as plt
# from torch.nn import functional as F
# from torchvision.transforms import Compose, ToTensor, Resize, Normalize


# from src import Range, load_model


# def main(args):

#     model, args = load_model(args)

#     if torch.cuda.is_available() and args.device == "cuda":
#         model.load_state_dict(torch.load(
#             'result/FedAvg_PLANT_VILLAGE_TwoNN_IID_240511_182614/FedAvg_PLANT_VILLAGE_TwoNN_IID.pt'))
#         model.to(args.device)
#     else:
#         model.load_state_dict(torch.load(
#             'result/FedAvg_PLANT_VILLAGE_TwoNN_IID_240511_182614/FedAvg_PLANT_VILLAGE_TwoNN_IID.pt', map_location=torch.device('cpu')))

#     # Liste des chemins des images de validation
#     validation_img_paths = [
#         "./dataset/validation/Apple___Apple_scab/image (5).JPG",
#         "./dataset/validation/Apple___Black_rot/image (2).JPG",
#         "./dataset/validation/Apple___Cedar_apple_rust/image (3).JPG",
#         "./dataset/validation/Apple___healthy/image (13).JPG",
#         "./dataset/validation/Corn___healthy/image (2).JPG",
#         "./dataset/validation/Corn___Common_rust/image (11).JPG",
#         "./dataset/validation/Corn___Cercospora_leaf_spot Gray_leaf_spot/image (17).JPG",
#         "./dataset/validation/Corn___Northern_Leaf_Blight/image (4).JPG",
#         "./dataset/validation/Potato___Early_blight/image (5).JPG",
#         "./dataset/validation/Potato___healthy/image (1).JPG",
#         "./dataset/validation/Potato___Late_blight/image (5).JPG",
#         "./dataset/validation/Tomato___Early_blight/image (2).JPG",
#         "./dataset/validation/Tomato___healthy/image (1).JPG",
#         "./dataset/validation/Tomato___Late_blight/image (4).JPG",
#         "./dataset/validation/Tomato___Septoria_leaf_spot/image (10).JPG",
#     ]

#     # Liste des noms des classes correspondants
#     class_names = [
#         "Apple - Apple scab",
#         "Apple - Black rot",
#         "Apple - Cedar apple rust",
#         "Apple - healthy",
#         "Corn - healthy",
#         "Corn - Common rust",
#         "Corn - Cercospora leaf spot Gray leaf spot",
#         "Corn - Northern Leaf Blight",
#         "Potato - Early blight",
#         "Potato - healthy",
#         "Potato - Late blight",
#         "Tomato - Early blight",
#         "Tomato - healthy",
#         "Tomato - Late blight",
#         "Tomato - Septoria leaf spot",
#     ]

#     img_list = [Image.open(img_path) for img_path in validation_img_paths]

#     transform = _get_transform(args)

#     with torch.no_grad():
#         validation_batch = torch.stack([transform(Image.open(img_path)).to(
#             args.device) for img_path in validation_img_paths])

#     pred_logits_tensor = model(validation_batch)
#     pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
#     print("Probabilite ==>", pred_probs)
#     predicted_classes = np.argmax(pred_probs, axis=1)
#     print(predicted_classes)

#     # Affichage
#     fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
#     for i, img in enumerate(img_list):
#         ax = axs[i]
#         ax.axis('off')
#         ax.set_title("Classe {:.0f}".format(predicted_classes[i]))
#         ax.imshow(img)
#     plt.show()


# def _get_transform(args):
#     transform = Compose(
#         [
#             Resize((args.resize, args.resize)),
#             ToTensor(),
#             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ]
#     )
#     return transform


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.RawTextHelpFormatter)

#     parser.add_argument('--resize', type=int, default=None)
#     parser.add_argument('--crop', type=int, default=None)
#     parser.add_argument('--imnorm', action='store_true')
#     parser.add_argument('--randrot', type=int, default=None)
#     parser.add_argument('--randhf', type=float,
#                         choices=[Range(0., 1.)], default=None)
#     parser.add_argument('--randvf', type=float,
#                         choices=[Range(0., 1.)], default=None)
#     parser.add_argument('--randjit', type=float,
#                         choices=[Range(0., 1.)], default=None)
#     parser.add_argument('--hidden_size', type=int, default=64)

#     parser.add_argument('--model_name', type=str,
#                         choices=[
#                             'TwoNN', 'TwoCNN',
#                             'VGG9', 'VGG9BN', 'VGG11', 'VGG11BN', 'VGG13', 'VGG13BN',
#                             'ResNet10', 'ResNet18', 'ResNet34',
#                         ],
#                         required=True
#                         )
#     parser.add_argument('--num_classes', type=int, default=4)
#     parser.add_argument('--in_channels', type=int, default=3)

#     parser.add_argument('--device', type=str, default='cpu')

#     args = parser.parse_args()

#     main(args)




import torch
import argparse
import numpy as np

from PIL import Image
import matplotlib.pylab as plt
from torch.nn import functional as F
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

from src import Range, load_model, tensorboard_runner

def main(args):
    # Charger le modèle
    model, args = load_model(args)
    tensorboard_runner(args)

    # Charger les poids du modèle
    load_model_weights(model, args)

    # Liste des chemins des images de validation
    validation_img_paths = [
        "./dataset/validation/Apple___Black_rot/image (2).JPG",
        "./dataset/validation/Apple___Black_rot/image (2).JPG",
        "./dataset/validation/Apple___Apple_scab/image (5).JPG",
        "./dataset/validation/Apple___Cedar_apple_rust/image (3).JPG",
        "./dataset/validation/Apple___healthy/image (13).JPG",
        "./dataset/validation/Corn___healthy/image (2).JPG",
        "./dataset/validation/Corn___Common_rust/image (11).JPG",
        "./dataset/validation/Corn___Cercospora_leaf_spot Gray_leaf_spot/image (17).JPG",
        "./dataset/validation/Corn___Northern_Leaf_Blight/image (4).JPG",
        "./dataset/validation/Potato___Early_blight/image (5).JPG",
        "./dataset/validation/Potato___healthy/image (1).JPG",
        "./dataset/validation/Potato___Late_blight/image (5).JPG",
        "./dataset/validation/Tomato___Early_blight/image (2).JPG",
        "./dataset/validation/Tomato___healthy/image (1).JPG",
        "./dataset/validation/Tomato___Late_blight/image (4).JPG",
        "./dataset/validation/Tomato___Septoria_leaf_spot/image (10).JPG",
    ]

    # Liste des noms des classes correspondants
    class_names = [
        "Apple - Apple scab",
        "Apple - Black rot",
        "Apple - Cedar apple rust",
        "Apple - healthy",
        "Corn - Cercospora leaf spot Gray leaf spot",
        "Corn - Common rust",
        "Corn - healthy",
        "Corn - Northern Leaf Blight",
        "Potato - Early blight",
        "Potato - healthy",
        "Potato - Late blight",
        "Tomato - Early blight",
        "Tomato - healthy",
        "Tomato - Late blight",
        "Tomato - Septoria leaf spot",
    ]

    # Transformer les images
    transform = get_transform(args)

    # Charger et transformer les images
    img_list = [Image.open(img_path) for img_path in validation_img_paths]
    validation_batch = load_and_transform_images(validation_img_paths, transform, args.device)

    # Faire des prédictions
    predicted_classes = make_predictions(model, validation_batch)

    # Afficher les résultats
    display_results(img_list, predicted_classes, class_names)

def load_model_weights(model, args):
    """Charge les poids du modèle en fonction du dispositif disponible."""
    if torch.cuda.is_available() and args.device == "cuda":
        model.load_state_dict(torch.load(
            'result/FedAvg_PLANT_VILLAGE_TwoNN_IID_240511_182614/FedAvg_PLANT_VILLAGE_TwoNN_IID.pt'))
        model.to(args.device)
    else:
        model.load_state_dict(torch.load(
            'result/FedAvg_PLANT_VILLAGE_TwoNN_IID_240511_182614/FedAvg_PLANT_VILLAGE_TwoNN_IID.pt', map_location=torch.device('cpu')))

def get_transform(args):
    """Retourne les transformations d'images."""
    return Compose([
        Resize((args.resize, args.resize)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def load_and_transform_images(img_paths, transform, device):
    """Charge et transforme une liste d'images."""
    with torch.no_grad():
        return torch.stack([transform(Image.open(img_path)).to(device) for img_path in img_paths])

def make_predictions(model, validation_batch):
    """Fait des prédictions sur un lot d'images."""
    model.eval()  # Met le modèle en mode évaluation
    pred_logits_tensor = model(validation_batch)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    return np.argmax(pred_probs, axis=1)

def display_results(img_list, predicted_classes, class_names):
    """Affiche les images avec leurs classes prédites, 4 par ligne."""
    num_images = len(img_list)
    num_cols = 5
    # num_rows = (num_images + num_cols - 1) // num_cols  # Calculer le nombre de lignes nécessaires
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculer le nombre de lignes nécessaires

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    axs = axs.flatten()  # Aplatir la grille pour itérer facilement

    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        ax.set_title(f"{class_names[predicted_classes[i]]}")
        ax.imshow(img)

    # Désactiver les axes pour les sous-graphiques vides
    for i in range(num_images, num_rows * num_cols):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Arguments de l'interface en ligne de commande
    parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--crop', type=int, default=None)
    parser.add_argument('--imnorm', action='store_true')
    parser.add_argument('--randrot', type=int, default=None)
    parser.add_argument('--randhf', type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randvf', type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randjit', type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--hidden_size', type=int, default=64)

    parser.add_argument('--model_name', type=str, choices=[
        'TwoNN', 'TwoCNN', 'VGG9', 'VGG9BN', 'VGG11', 'VGG11BN', 'VGG13', 'VGG13BN',
        'ResNet10', 'ResNet18', 'ResNet34',
    ], required=True)

    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_path', help='Chemin des logs',
                        type=str, default='./log')
    parser.add_argument('--tb_port', help='TensorBoard',
                        type=int, default=6006)

    args = parser.parse_args()
    main(args)
