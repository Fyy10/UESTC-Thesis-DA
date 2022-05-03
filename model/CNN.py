import torch.nn as nn
import torchvision.models
import torch


# LeNet-5
class LeNet(nn.Module):
    """
    `LeNet-5` Model

    Input size: 32x32

    Feature dimension: 84
    """
    def __init__(self):
        super(LeNet, self).__init__()
        # input: [N, 3, 32, 32]
        # output: [N, 84]
        self.conv1 = nn.Conv2d(3, 6, (5, 5), padding=2)
        self.avgpool = nn.AvgPool2d((2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5), padding=0)
        # for [28, 28]
        # self.fc1 = nn.Linear(5 * 5 * 16, 120)
        # for [32, 32]
        self.fc1 = nn.Linear(6 * 6 * 16, 120)
        self.fc2 = nn.Linear(120, 84)

        # there is no need to init param, PyTorch will do it for you
        # self.init_param()

    def init_param(self):
        # conv1
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.normal_(self.conv1.bias, mean=0, std=0.01)
        # conv2
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.normal_(self.conv2.bias, mean=0, std=0.01)
        # fc1
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, mean=0, std=0.01)
        # fc2
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, mean=0, std=0.01)

    def forward(self, x):
        # x: [N, 3, 28, 28] for [28, 28]
        # x: [N, 3, 32, 32] for [32, 32]
        # input: [N, C, H, W]
        x = torch.sigmoid(self.conv1(x))
        # [N, 6, 28, 28] for [28, 28]
        # [N, 6, 32, 32] for [32, 32]
        x = self.avgpool(x)
        # [N, 6, 14, 14] for [28, 28]
        # [N, 6, 16, 16] for [32, 32]
        x = torch.sigmoid(self.conv2(x))
        # [N, 16, 10, 10] for [28, 28]
        # [N, 16, 12, 12] for [32, 32]
        x = self.avgpool(x)
        # [N, 16, 5, 5] for [28, 28]
        # [N, 16, 6, 6] for [32, 32]
        x = torch.flatten(x, 1)
        # [N, 16 * 5 * 5] for [28, 28]
        # [N, 16 * 6 * 6] for [32, 32]
        x = torch.sigmoid(self.fc1(x))
        # [N, 120]
        x = torch.sigmoid(self.fc2(x))
        # [N, 84]
        return x


# 3conv2fc
class Net_3conv2fc(nn.Module):
    def __init__(self):
        """
        `3conv2fc` Model

        Input size: 32x32

        Feature dimension: 2048
        """
        super(Net_3conv2fc, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)

        # there is no need to init param, PyTorch will do it for you
        # self.init_param()

    def init_param(self):
        # conv1
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.normal_(self.conv1.bias, mean=0, std=0.01)
        # bn1
        nn.init.normal_(self.bn1.weight, mean=0, std=0.01)
        nn.init.normal_(self.bn1.bias, mean=0, std=0.01)
        # conv2
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.normal_(self.conv2.bias, mean=0, std=0.01)
        # bn2
        nn.init.normal_(self.bn2.weight, mean=0, std=0.01)
        nn.init.normal_(self.bn2.bias, mean=0, std=0.01)
        # conv3
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.normal_(self.conv3.bias, mean=0, std=0.01)
        # bn3
        nn.init.normal_(self.bn3.weight, mean=0, std=0.01)
        nn.init.normal_(self.bn3.bias, mean=0, std=0.01)
        # fc1
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, mean=0, std=0.01)
        # bn1_fc
        nn.init.normal_(self.bn1_fc.weight, mean=0, std=0.01)
        nn.init.normal_(self.bn1_fc.bias, mean=0, std=0.01)
        # fc2
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, mean=0, std=0.01)
        # bn2_fc
        nn.init.normal_(self.bn2_fc.weight, mean=0, std=0.01)
        nn.init.normal_(self.bn2_fc.bias, mean=0, std=0.01)

    def forward(self, x):
        x = torch.max_pool2d(torch.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = torch.max_pool2d(torch.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = torch.relu(self.bn1_fc(self.fc1(x)))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = torch.relu(self.bn2_fc(self.fc2(x)))
        # x: [batch_size, num_features=2048]
        return x


# AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # input: [N, 3, 224, 224]
        # output: [N, 4096]
        self.model = torchvision.models.alexnet()
        self.feature = self.model.features
        self.avgpool = self.model.avgpool
        self.classifier = self.model.classifier[:-1]

    def forward(self, x):
        # x: [N, 3, 224, 224]
        x = self.feature(x)
        # x: [N, 256, 6, 6]
        x = self.avgpool(x)
        # x: [N, 256, 6, 6]
        x = torch.flatten(x, 1)
        # x: [N, 9216]
        x = self.classifier(x)
        return x


# AlexNet implementation of original paper
# refer to https://github.com/jiecaoyu/pytorch_imagenet
# Local Response Normalization (LRN)
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(
                kernel_size=(local_size, 1, 1),
                stride=1,
                padding=(int((local_size-1.0)/2), 0, 0)
            )
        else:
            self.average=nn.AvgPool2d(
                kernel_size=local_size,
                stride=1,
                padding=int((local_size-1.0)/2)
            )
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


# original AlexNet (without the last classification layer)
class AlexNetOrg(nn.Module):
    def __init__(self):
        super(AlexNetOrg, self).__init__()
        # input: [N, 3, 224, 224]
        # output: [N, 4096]
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            # for [256, 256]
            # nn.Linear(256 * 6 * 6, 4096),
            # for [224, 224]
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

    def forward(self, x):
        # x: [N, 3, 224, 224]
        x = self.features(x)
        # [N, 256, 6, 6] (for [256, 256])
        # [N, 256, 5, 5] (for [224, 224])
        x = torch.flatten(x, 1)
        # [N, 256 * 6 * 6 = 9216] (for [256, 256])
        # [N, 256 * 5 * 5 = 6400] (for [224, 224])
        x = self.classifier(x)
        # [N, 4096]
        return x


# TODO: ResNet152
class ResNet152(nn.Module):
    def __init__(self):
        super(ResNet152, self).__init__()
        self.model = torchvision.models.resnet152()


# classifier
class Classifier(nn.Module):
    def __init__(self, in_dim=2048, out_dim=10):
        super(Classifier, self).__init__()
        # [N, in_dim] -> [N, out_dim]
        self.fc1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc1(x)


def test_lenet():
    feature = LeNet()
    classifier = Classifier(84, 10)
    print(feature)
    print(classifier)
    data = torch.randn((10, 3, 32, 32))
    feat = feature(data)
    print('feature:', feat.size())
    out = classifier(feat)
    print('out:', out.size())


def test_3conv2fc():
    feature = Net_3conv2fc()
    classifier = Classifier(2048, 10)
    print(feature)
    print(classifier)
    data = torch.randn((10, 3, 32, 32))
    feat = feature(data)
    print('feature:', feat.size())
    out = classifier(feat)
    print('out:', out.size())


def test_alexnet():
    feature = AlexNet().to(0)
    classifier = Classifier(4096, 10).to(0)
    print(feature)
    print(classifier)
    data = torch.randn((10, 3, 224, 224)).to(0)
    feat = feature(data)
    print('feature:', feat.size())
    out = classifier(feat)
    print('out:', out.size())


def test_alexnet_org():
    feature = AlexNetOrg().to(0)
    classifier = Classifier(4096, 10).to(0)
    print(feature)
    print(classifier)
    data = torch.randn((10, 3, 224, 224)).to(0)
    feat = feature(data)
    print('feature:', feat.size())
    out = classifier(feat)
    print('out:', out.size())


def test_resnet152():
    feature = ResNet152().to(0)
    classifier = Classifier(2048, 10).to(0)
    print(feature)
    print(classifier)
    data = torch.randn((10, 3, 224, 224)).to(0)
    feat = feature(data)
    print('feature:', feat.size())
    out = classifier(feat)
    print('out:', out.size())


# test model
if __name__ == '__main__':
    # test_lenet()
    test_3conv2fc()
    # test_alexnet()
    # test_alexnet_org()
