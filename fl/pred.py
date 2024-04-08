import torch
import argparse
import numpy as np

from PIL import Image
import matplotlib.pylab as plt
from torch.nn import functional as F
from torchvision.transforms import Compose, ToTensor, Resize


from src import Range, load_model


def main(args):

    model, args = load_model(args)

    if torch.cuda.is_available() and args.device == "cuda":
        model.load_state_dict(torch.load(
            'result/FedProx_PLANT_VILLATE_2CNN_IID_240405_072812/FedProx_PLANT_VILLATE_2CNN_IID.pt'))
        model.to(args.device)
    else:
        model.load_state_dict(torch.load(
            'result/FedProx_PLANT_VILLATE_2CNN_IID_240405_072812/FedProx_PLANT_VILLATE_2CNN_IID.pt', map_location=torch.device('cpu')))

    validation_img_paths = [
        "./dataset/validation/Corn___Cercospora_leaf_spot Gray_leaf_spot/image (10).JPG",
        "./dataset/validation/Potato___healthy/image (2).JPG",
        "./dataset/validation/Potato___Late_blight/image (77).JPG",
        "./dataset/validation/Potato___Late_blight/image (95).JPG",
        "./dataset/validation/Corn___healthy/image (2).jpg",
        "./dataset/validation/Corn___Cercospora_leaf_spot Gray_leaf_spot/image (17).JPG",
        "./dataset/validation/Potato___healthy/image (99).JPG",
        "./dataset/validation/img2.jpg",
        "./dataset/validation/213.jpg",
    ]

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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--crop', type=int, default=None)
    parser.add_argument('--imnorm', action='store_true')
    parser.add_argument('--randrot', type=int, default=None)
    parser.add_argument('--randhf', type=float,
                        choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randvf', type=float,
                        choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randjit', type=float,
                        choices=[Range(0., 1.)], default=None)
    parser.add_argument('--hidden_size', type=int, default=64)

    parser.add_argument('--model_name', type=str,
                        choices=[
                            'TwoNN', 'TwoCNN',
                            'VGG9', 'VGG9BN', 'VGG11', 'VGG11BN', 'VGG13', 'VGG13BN',
                            'ResNet10', 'ResNet18', 'ResNet34',
                        ],
                        required=True
                        )
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=3)

    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    main(args)
