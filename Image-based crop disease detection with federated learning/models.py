import torch.nn as nn
import torchvision.models as models
import timm


# ResNet50

# class ResNet50(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNet50, self).__init__()
#         self.resnet50 = models.resnet50(pretrained=True)
#         self.resnet50.fc = nn.Sequential(
#                nn.Linear(2048, 128),
#                nn.ReLU(inplace=True),
#                nn.Linear(128, num_classes))

#     def forward(self, x):
#         return self.resnet50(x)

def ResNet50(num_classes: int): 
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False   
        
    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes))
    
    return model


# DenseNet121
class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)

        self.densenet121.classifier = nn.Linear(
            in_features=self.densenet121.classifier.in_features, out_features=num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) # On ajoute une dimension, car le modele fonction en 3D ou en 4D
        return self.densenet121(x)


# InceptionV3
class InceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3, self).__init__()
        self.inception_v3 = models.inception_v3(pretrained=True)

        self.inception_v3.fc = nn.Linear(
            in_features=self.inception_v3.fc.in_features, num_classes=num_classes)

    def forward(self, x):
        return self.inception_v3(x)

# VGG16


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)

        in_features = self.vgg16.classifier[-1].in_features
        self.vgg16.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)


# ViTB16

class ViT_B16(nn.Module):
    def __init__(self, num_classes):
        super(ViT_B16, self).__init__()
        self.vit_b16 = timm.create_model(
            'vit_base_patch16_224', pretrained=True)

        self.vit_b16.head = nn.Linear(
            in_features=self.vit_b16.head.in_features, num_classes=num_classes)

    def forward(self, x):
        return self.vit_b16(x)


# ViTB32

class ViT_B32(nn.Module):
    def __init__(self, num_classes):
        super(ViT_B32, self).__init__()
        self.vit_b32 = timm.create_model(
            'vit_base_patch32_224', pretrained=True)

        self.vit_b32.head = nn.Linear(
            in_features=self.vit_b32.head.in_features, num_classes=num_classes)

    def forward(self, x):
        return self.vit_b32(x)


if __name__ == "__main__":
    """
    Juste pour tester
    """
    num_classes = 100
    rest = ResNet50(num_classes)

    print(rest)
