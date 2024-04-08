import torch

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(planes),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(planes)
        )

        self.shortcut = torch.nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        x = self.features(x) + self.shortcut(x) 
        x = torch.nn.functional.relu(x)
        return x
