import torch
import torch.nn as nn

def swish(x):
    return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.sample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        x = self.sample(x)
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels):
        super(Downsample, self).__init__()
        self.sample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        x = self.sample(x)
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super(ResidualBlock, self).__init__()
        if in_channels != out_channels:
            self.convx = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.convx = nn.Identity()
        self.convt = nn.Conv2d(time_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.group_norm2 = nn.GroupNorm(32, out_channels)
    
    def forward(self, x, t):
        xi = self.convx(x)
        x = self.group_norm1(x)
        x = swish(x)
        x = self.conv1(x)
        t = self.convt(t)
        x = x + t
        x = self.group_norm2(x)
        x = swish(x)
        x = self.conv2(x)
        x = x + xi
        return x


class Diffusion(nn.Module):
    """
        Simple version of U-Net, refer to kexue.fm
    """
    def __init__(self, T, embedding_size, channels, blocks):
        super(Diffusion, self).__init__()
        self.block = blocks
        self.embedding = nn.Embedding(T, embedding_size)
        self.convx = nn.Conv2d(3, embedding_size, kernel_size=3, padding=1, bias=False)
        time_channels = embedding_size * 4
        self.convt = nn.Conv2d(embedding_size, time_channels, kernel_size=3, padding=1, bias=False)
        
        self.down_blocks = nn.ModuleList()
        in_ch = embedding_size
        chans = [in_ch]
        for i, ch in enumerate(channels):
            for _ in range(blocks):
                self.down_blocks.append(ResidualBlock(in_ch, ch * embedding_size, time_channels))
                in_ch = ch * embedding_size
                chans.append(in_ch)
            if i != len(channels) - 1:
                self.down_blocks.append(Downsample(in_ch))
                chans.append(in_ch)

        self.residual_block = ResidualBlock(in_ch, in_ch, time_channels)      
        
        self.up_blocks = nn.ModuleList()
        for i, ch in enumerate(channels[::-1]):
            for _ in range(blocks + 1):
                self.up_blocks.append(ResidualBlock(in_ch + chans.pop(), ch * embedding_size, time_channels))
                in_ch = ch * embedding_size
            if i != len(channels) - 1:
                self.up_blocks.append(Upsample(in_ch))

        self.conv2 = nn.Conv2d(in_ch, 3, kernel_size=3, padding=1, bias=False)
        self.group_norm = nn.GroupNorm(32, in_ch)

    def forward(self, x, t):
        t = self.embedding(t)[:, :, None, None]
        x = self.convx(x)
        t = swish(self.convt(t))
        
        inputs = [x]
        for module in self.down_blocks:
            x = module(x, t)
            inputs.append(x)

        x = self.residual_block(x, t)
        
        for i, module in enumerate(self.up_blocks):
            if (i + 1) % (self.block + 2) != 0: 
                x = torch.cat([x, inputs.pop()], dim=1)
            x = module(x, t)
            
        x = self.group_norm(x)
        x = swish(x)
        x = self.conv2(x)
        return x