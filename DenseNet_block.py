from torch import nn
import torch

class bottleneck(nn.Module):
    def __init__(self, nchannels, growthrate):
        super(bottleneck, self).__init__()
        interchannels = 4 * growthrate
        self.bn1 = nn.BatchNorm2d(nchannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nchannels, interchannels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(interchannels)
        self.conv2 = nn.Conv2d(interchannels, growthrate, 3, padding=1, bias=False)

    def forward(self,x):
        x_finish = self.bn1(x)
        x_finish = self.relu(x_finish)
        x_finish = self.conv1(x_finish)
        x_finish = self.bn2(x_finish)
        x_finish = self.relu(x_finish)
        x_finish = self.conv2(x_finish)
        out = torch.cat((x , x_finish) , dim=1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nchannels, growthrate, nDenseBlocks):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(bottleneck(nchannels, growthrate))
            nchannels +=growthrate
        self.denseblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.denseblock(x)


denseblock = DenseBlock(64, 32, 6)
input = torch.randn(1, 64, 256, 256)
output = denseblock(input)
print(output.shape)