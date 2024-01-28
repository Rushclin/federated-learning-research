from models import DenseNet121, InceptionV3, VGG16, ViT_B16, ViT_B32, ResNet50

def get_model(model_str: str, num_classes: int):
    if model_str == "ResNet50":
        return ResNet50(num_classes)
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
