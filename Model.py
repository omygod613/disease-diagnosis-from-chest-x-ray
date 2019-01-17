import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model_ft = models.alexnet(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class VGG11BN(nn.Module):
    def __init__(self):
        super(VGG11BN, self).__init__()
        self.model_ft = models.vgg11_bn(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class VGG19BN(nn.Module):
    def __init__(self):
        super(VGG19BN, self).__init__()
        self.model_ft = models.vgg19_bn(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class ResNet50_mod(nn.Module):
    def __init__(self):
        super(ResNet50_mod, self).__init__()
        self.model_ft = models.resnet50(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False

        self.prediction = nn.Sequential(
            nn.Linear(2048, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x)

        x = self.model_ft.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.prediction(x)#8
        
        return x


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model_ft = models.resnet34(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)#3        

        return x


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model_ft = models.resnet50(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)#3        

        return x


class DenseNet169(nn.Module):
    def __init__(self):
        super(DenseNet169, self).__init__()
        self.model_ft = models.densenet169(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class DenseNet201(nn.Module):
    def __init__(self):
        super(DenseNet201, self).__init__()
        self.model_ft = models.densenet201(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class SqueezeNet1_0(nn.Module):
    def __init__(self):
        super(SqueezeNet1_0, self).__init__()
        self.model_ft = models.squeezenet1_0(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class SqueezeNet1_1(nn.Module):
    def __init__(self):
        super(SqueezeNet1_1, self).__init__()
        self.model_ft = models.squeezenet1_1(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(64 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Improved_MyModel(nn.Module):
    def __init__(self):
        super(Improved_MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 14 * 14, 512)
        self.batch_norm = nn.BatchNorm2d(512)
        self.fc_mid = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.batch_norm(x)
        x = F.relu(self.fc_mid(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MyModel_Gray(nn.Module):
    def __init__(self):
        super(MyModel_Gray, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(64 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Improved_MyModel_Gray(nn.Module):
    def __init__(self):
        super(Improved_MyModel_Gray, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 14 * 14, 512)
        self.batch_norm = nn.BatchNorm2d(512)
        self.fc_mid = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.batch_norm(x)
        x = F.relu(self.fc_mid(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
