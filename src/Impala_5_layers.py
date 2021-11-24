import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module
    
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLu()(x)
        out = self.conv1(out)
        return out + x
        
class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class FiveBlocksImpala(nn.Module):
    def __init__(self,
                 in_channels, out_channels = 256,
                 **kwargs):
        super(FiveBlocksImpala, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=8)
        self.block2 = ImpalaBlock(in_channels=8, out_channels=16)
        self.block3 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block4 = ImpalaBlock(in_channels=32, out_channels=64)
        self.block5 = ImpalaBlock(in_channels=64, out_channels=128)
        self.fc = nn.Linear(in_features=128 * 8 * 8, out_features=out_channels)

        self.output_dim = out_channels
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = nn.ReLu()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLu()(x)
        return x
