# lib/models.py

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, norm=True, activation='relu'
    ):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)
        )
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, activation=None)
        )
    
    def forward(self, x):
        return x + self.block(x)

class ProgressiveGenerator(nn.Module):
    def __init__(self, start_scale=64, max_scale=256, num_residuals=6):
        super(ProgressiveGenerator, self).__init__()
        self.start_scale = start_scale
        self.max_scale = max_scale
        self.current_scale = start_scale
        self.num_residuals = num_residuals
        self.models = nn.ModuleDict()
        self._build_models()
    
    def _build_models(self):
        scale = self.start_scale
        while scale <= self.max_scale:
            self.models[str(scale)] = self._build_model()
            scale *= 2
    
    def _build_model(self):
        model = []
        # Initial Convolution Block
        model.append(ConvBlock(3, 64, kernel_size=7, stride=1, padding=3))
        # Downsampling
        in_channels = 64
        out_channels = in_channels * 2
        for _ in range(2):
            model.append(ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            in_channels = out_channels
            out_channels *= 2
        # Residual Blocks
        for _ in range(self.num_residuals):
            model.append(ResidualBlock(in_channels))
        # Upsampling
        out_channels = in_channels // 2
        for _ in range(2):
            model.append(nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ))
            model.append(nn.InstanceNorm2d(out_channels))
            model.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            out_channels = in_channels // 2
        # Output Layer
        model.append(nn.Conv2d(in_channels, 3, kernel_size=7, stride=1, padding=3))
        model.append(nn.Tanh())
        return nn.Sequential(*model)
    
    def forward(self, x):
        x = nn.functional.interpolate(x, size=(self.current_scale, self.current_scale))
        model = self.models[str(self.current_scale)]
        return model(x)
    
    def increase_scale(self):
        if self.current_scale < self.max_scale:
            self.current_scale *= 2
        else:
            print("Already at maximum scale.")

class ProgressiveDiscriminator(nn.Module):
    def __init__(self, start_scale=64, max_scale=256):
        super(ProgressiveDiscriminator, self).__init__()
        self.start_scale = start_scale
        self.max_scale = max_scale
        self.current_scale = start_scale
        self.models = nn.ModuleDict()
        self._build_models()
    
    def _build_models(self):
        scale = self.start_scale
        while scale <= self.max_scale:
            self.models[str(scale)] = self._build_model()
            scale *= 2
    
    def _build_model(self):
        model = []
        model.append(ConvBlock(3, 64, kernel_size=4, stride=2, padding=1, norm=False, activation='leaky_relu'))
        in_channels = 64
        out_channels = in_channels * 2
        for _ in range(3):
            model.append(ConvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1, activation='leaky_relu'))
            in_channels = out_channels
            out_channels *= 2
        model.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1))
        return nn.Sequential(*model)
    
    def forward(self, x):
        x = nn.functional.interpolate(x, size=(self.current_scale, self.current_scale))
        model = self.models[str(self.current_scale)]
        return model(x)
    
    def increase_scale(self):
        if self.current_scale < self.max_scale:
            self.current_scale *= 2
        else:
            print("Already at maximum scale.")
