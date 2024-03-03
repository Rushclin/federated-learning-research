import torch
from torch.nn import functional as F
from torchvision.transforms import Compose, ToTensor, Resize

import hydra
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import DictConfig

from src import load_model


@hydra.main(config_name="base", config_path="./src/config", version_base=None)
def main(args: DictConfig):

    # Load the model
    model = load_model(args)

    # Check if device is available before moving to GPU
    if torch.cuda.is_available() and args.device == "cuda":
        model.load_state_dict(torch.load('resultats/FedAVG_240303_042032/FedAVG.pt'))
        model.to(args.device)  # Move model to GPU only if available
    else:
        model.load_state_dict(torch.load('resultats/FedAVG_240303_042032/FedAVG.pt', map_location=torch.device('cpu')))  # Load on CPU

    validation_img_paths = [
        "./dataset/validation/0/img_110.jpg",
        "./dataset/validation/9/img_592.jpg",
        "./dataset/validation/4/img_700.jpg",
        # "./dataset/validation/5/img_33155.jpg",
        # "./dataset/validation/5/img_33155.jpg",
        "./dataset/validation/8/img_848.jpg",
        "./dataset/validation/9/img_5393.jpg",
        # "./dataset/validation/5/img_33155.jpg",
        "./dataset/validation/8/img_440.jpg",
    ]

    img_list = [Image.open(img_path) for img_path in validation_img_paths]

    # Transform images for validation
    transform = _get_transform(args)

    with torch.no_grad():  
        validation_batch = torch.stack([transform(_grayscale_to_rgb_duplicate(Image.open(img_path))).to(args.device) for img_path in validation_img_paths])


    pred_logits_tensor = model(validation_batch)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    predicted_classes = np.argmax(pred_probs, axis=1)

    # Affichage 
    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):        
        ax = axs[i]
        ax.axis('off')
        ax.set_title("Classe {:.0f}".format(predicted_classes[i]))
        ax.imshow(img)
    plt.show()


def _get_transform(args):
    transform = Compose(
        [
            Resize((args.resize, args.resize)),
            ToTensor(),
        ]
    )
    return transform


def _grayscale_to_rgb_duplicate(image):
  """
  Convertit une image en niveaux de gris en RGB en dupliquant le canal de niveaux de gris trois fois.

  Args:
    image: Une image en niveaux de gris (PIL Image).

  Retourne:
    Une image RGB (PIL Image).
  """
  # Convertir l'image en NumPy array
  image_array = np.array(image)

  # Dupliquer le canal de niveaux de gris trois fois
  image_array = np.repeat(image_array[:, :, np.newaxis], 3, axis=2)

  # Convertir le NumPy array en image PIL
  image = Image.fromarray(image_array, 'RGB')

  return image


if __name__ == "__main__":
    main()
