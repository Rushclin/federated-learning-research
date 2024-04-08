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
        model.load_state_dict(torch.load(
            'resultats/FedProx_MNIST_2NN_IID_C0.0_B0.pt'))
        model.to(args.device)  # Move model to GPU only if available
    else:
        model.load_state_dict(torch.load(
            'resultats/FedProx_MNIST_2NN_IID_C0.0_B0.pt', map_location=torch.device('cpu')))  # Load on CPU

    validation_img_paths = ["./dataset/validation/Corn___Cercospora_leaf_spot Gray_leaf_spot/image (10).JPG",
                            "./dataset/validation/Potato___healthy/image (2).JPG",
                            "./dataset/validation/Potato___Late_blight/image (12).JPG",
                            "./dataset/validation/Potato___Late_blight/image (22).JPG",
                            "./dataset/validation/Corn___healthy/image (2).jpg",
                            "./dataset/validation/Corn___Cercospora_leaf_spot Gray_leaf_spot/image (17).JPG",
                            "./dataset/validation/Potato___healthy/image (99).JPG",]

    img_list = [Image.open(img_path) for img_path in validation_img_paths]

    transform = _get_transform(args)

    with torch.no_grad():
        validation_batch = torch.stack([transform(Image.open(img_path)).to(
            args.device) for img_path in validation_img_paths])

    pred_logits_tensor = model(validation_batch)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    predicted_classes = np.argmax(pred_probs, axis=1)
    print(predicted_classes)

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


if __name__ == "__main__":
    main()
