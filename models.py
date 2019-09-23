from torch import nn
from torchsummary import summary
from torchvision import models

from config import device, num_classes


class FaceExpressionModel(nn.Module):
    def __init__(self):
        super(FaceExpressionModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 1, 1]
        x = x.view(-1, 2048)  # [N, 2048]
        x = self.fc(x)
        out = self.softmax(x)
        return out


if __name__ == "__main__":
    model = FaceExpressionModel().to(device)
    summary(model, input_size=(3, 224, 224))
