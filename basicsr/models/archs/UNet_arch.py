import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, batch_norm=True):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        if batch_norm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, batch_norm=batch_norm),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, batch_norm=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, batch_norm
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, batch_norm=batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetDynamic(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth: int = 64,
        layers: int = 4,
        bilinear=True,
        batch_norm=True,
    ):
        super(UNetDynamic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.batch_norm = batch_norm

        if depth not in [8, 16, 32, 64]:
            raise ValueError("Incorrect input kernel size")
        self.depth = depth

        if layers not in [1, 2, 3, 4]:
            raise ValueError("Incorrect Layer size")
        self.layers = layers

        factor = 2 if bilinear else 1

        inc = DoubleConv(in_channels, depth, batch_norm=batch_norm)
        self.encoder = nn.ModuleList()
        self.encoder.append(inc)
        for i in range(layers):
            self.encoder.append(
                Down(
                    depth * pow(2, i),
                    depth * pow(2, i + 1) // (factor if i == layers - 1 else 1),
                    batch_norm=batch_norm,
                )
            )

        self.decoder = nn.ModuleList()
        for i in range(layers, 0, -1):
            self.decoder.append(
                Up(
                    depth * pow(2, i),
                    depth * pow(2, i - 1) // (factor if i != 1 else 1),
                    bilinear,
                    batch_norm,
                )
            )
        
        self.outc = OutConv(depth, out_channels)

    def forward(self, x):
        in_x = x
        res = []

        for i, down in enumerate(self.encoder):
            x = down(x)
            if i != len(self.encoder) - 1:
                res.append(x)
        
        for up in self.decoder:
            x = up(x, res.pop())
        
        out = F.relu(self.outc(x))

        return out