
import torch
import torch.nn as nn

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Inception(nn.Module):

    def __init__(self, in_channels, conv1x1_out, conv3x3_reduce, conv3x3_out, 
                 conv5x5_reduce, conv5x5_out, pool_proj):
        super().__init__()
        self.branch1 = BasicConv(in_channels, conv1x1_out, 1)

        self.branch2 = nn.Sequential(
            BasicConv(in_channels, conv3x3_reduce, 1),
            BasicConv(conv3x3_reduce, conv3x3_out, 3, padding=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv(in_channels, conv5x5_reduce, 1),
            BasicConv(conv5x5_reduce, conv5x5_out, 5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            BasicConv(in_channels, pool_proj, 1),
        )

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        return torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], 1)

class AuxClassifier(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AvgPool2d(5, stride=3)
        self.conv = BasicConv(in_channels, 128, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GoogLeNet(nn.Module):

    def __init__(self, num_classes=21, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv(3, 64, 7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)
        self.lrn1 = nn.LocalResponseNorm(2)

        self.conv2 = nn.Sequential(
            BasicConv(64, 64, 1),
            BasicConv(64, 192, 3, padding=1),
        )
        self.lrn2 = nn.LocalResponseNorm(2)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)
            self.aux2 = AuxClassifier(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.lrn1(x)

        x = self.conv2(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        aux1_out = None
        if self.aux_logits and self.training:
            aux1_out = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        aux2_out = None
        if self.aux_logits and self.training:
            aux2_out = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return x, aux1_out, aux2_out
        return x