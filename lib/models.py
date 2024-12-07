import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ProgressiveGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, resolution, ngf=64):
        super(ProgressiveGenerator, self).__init__()
        # resolutions list
        res_list = [16, 32, 64, 128, 256, 512]
        stage = res_list.index(resolution) + 1

        # init block
        self.initial = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # downsampling layers
        down_layers = []
        in_channels = ngf
        down = stage if stage < 6 else stage - 1
        for i in range(down):
            out_channels = in_channels * 2
            down_layers += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_channels = out_channels
        self.down = nn.Sequential(*down_layers)

        # residual blocks
        res_layers = []
        for _ in range(stage):
            res_layers.append(ResidualBlock(in_channels))
        self.residual = nn.Sequential(*res_layers)

        # upsampling layers
        up_layers = []
        up = stage if stage < 6 else stage - 1
        for i in range(up):
            out_channels = in_channels // 2
            up_layers += [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_channels = out_channels
        self.up = nn.Sequential(*up_layers)

        # final layer
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, output_nc, kernel_size=7, padding=3), nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down(x)
        x = self.residual(x)
        x = self.up(x)
        x = self.final(x)
        return x


class ProgressiveDiscriminator(nn.Module):
    def __init__(self, input_nc, resolution, ndf=64):
        super(ProgressiveDiscriminator, self).__init__()
        if resolution < 32:
            disc_layers = 2
        elif resolution < 128:
            disc_layers = 3
        else:
            disc_layers = 4

        layers = []
        # first layer
        layers.append(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # second layer
        if disc_layers > 1:
            layers.append(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(ndf * 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # third layer
        if disc_layers > 2:
            layers.append(
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.InstanceNorm2d(ndf * 4))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # fourth layer
        if disc_layers > 3:
            layers.append(nn.Conv2d(ndf * 4, 1, kernel_size=4, padding=1))
        else:
            final_in = ndf * (2 ** (disc_layers - 1))
            layers.append(nn.Conv2d(final_in, 1, kernel_size=4, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
