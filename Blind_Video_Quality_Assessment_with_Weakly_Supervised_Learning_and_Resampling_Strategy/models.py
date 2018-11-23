import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcNet(nn.Module):
    def __init__(self, is_training=True):
        super(ArcNet, self).__init__()
        self.conv1 = CNNBlock(in_channels=1, out_channels=50, pooling_kernel=2, is_training=is_training)
        self.conv2 = CNNBlock(in_channels=50, out_channels=50, padding=1, pooling_kernel=3, is_training=is_training)
        self.conv3 = CNNBlock(in_channels=50, out_channels=50, pooling_kernel=3, is_training=is_training)
        self.dense1 = nn.Linear(1800, 500)
        self.dense2 = nn.Linear(500, 1)

    def forward(self, x):
        if len(x.size()) < 4:
            x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view([-1, 1800])

        x = self.dense1(x)
        x = self.dense2(x)

        return torch.squeeze(x)


class CNNBlock(nn.Module):
    """
    This class constructs a cnn block, including a convolution layer,
    a batchnorm layer, a relu layer, and a max pooling layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 batchnorm=False,
                 relu=True,
                 max_pooling=True,
                 pooling_kernel=None,
                 is_training=True):
        super(CNNBlock, self).__init__()
        # config
        self.batchnorm = batchnorm
        self.relu = relu
        self.max_pooling = max_pooling
        self.pooling_kernel = pooling_kernel

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if self.batchnorm:
            self.bn = nn.BatchNorm2d(out_channels, track_running_stats=is_training)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x)
        if self.max_pooling:
            x = F.max_pool2d(x, kernel_size=self.pooling_kernel, stride=2)

        return x


class LinearRegression(nn.Module):
    def __init__(self, in_channels):
        super(LinearRegression, self).__init__()
        self.dense = nn.Linear(in_features=in_channels, out_features=1)

    def forward(self, x):
        x = self.dense(x)
        return torch.squeeze(x)


if __name__ == '__main__':
    x = torch.randn((2, 64, 64))
    model = ArcNet()
    y = model(x)
    print(y)