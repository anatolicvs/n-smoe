import datetime
import glob
import os
from dataclasses import dataclass

import click
import piq
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (CenterCrop, Compose, Resize, ToPILImage,
                                    ToTensor)
from torchvision.utils import make_grid, save_image

from dnnlib import EasyDict
import matplotlib.cm as cm 
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
import matplotlib.pyplot as plt

def network_parameters(nets):
    num_params = sum(param.numel() for param in nets.parameters())
    return num_params


class OpenImageDataset(Dataset):
    def __init__(
        self,
        dir: str,
        train: bool = False,
        test: bool = False,
        val: bool = False,
        bm3d: bool = False,
        transform: object = None,
        convert: str = "L",
        phw: int = 8,
        stride: int = 8,
        length: int = 1000,
        noise_type: str = "gauss",
        noise_level: float = 25,
        device="cpu",
    ) -> object:
        super(OpenImageDataset, self).__init__()

        self.phw = phw
        self.stride = stride
        self.transform = transform
        self.convert = convert
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.device = device

        root_dir = os.path.abspath(dir)
        if train:
            self.path = os.path.join(root_dir, "train_0")
        elif test:
            self.path = os.path.join(root_dir, "test")
        elif val:
            self.path = os.path.join(root_dir, "validation")
        elif bm3d:
            self.path = root_dir
        else:
            raise ValueError(
                "No valid dataset type specified. Choose train, test, or val."
            )

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Directory {self.path} does not exist")

        self.images = sorted(glob.glob(self.path + "/*.*")[:length])
        if not self.images:
            raise FileNotFoundError(f"No images found in directory {self.path}")

    @staticmethod
    def extract_blocks(img_tensor, block_size, overlap):
        blocks = []
        step = block_size - overlap
        for i in range(0, img_tensor.shape[1] - block_size + 1, step):
            for j in range(0, img_tensor.shape[2] - block_size + 1, step):
                block = img_tensor[:, i:i+block_size, j:j+block_size]
                blocks.append(block)
        return torch.stack(blocks)


    def __getitem__(self, index: int):
        rng = torch.Generator().manual_seed(torch.initial_seed() + index)
        try:
            image_path = self.images[index]
            with Image.open(image_path) as img:
                gt_y = img.convert(self.convert)

            if self.transform:
                gt_y = self.transform(gt_y)

            n_y = self.add_noise(gt_y, rng)

            # n_y_p = n_y.unfold(1, self.phw, self.stride).unfold(2, self.phw, self.stride)
            # n_y_p = F.max_pool3d(n_y_p, kernel_size=1, stride=1)
            # n_y_p = n_y_p.view(
            #     n_y_p.shape[1] * n_y_p.shape[2], n_y_p.shape[0], n_y_p.shape[3], n_y_p.shape[4]
            # )
            
            n_y_p = self.extract_blocks(n_y, self.phw, self.stride)

            return gt_y.to(self.device), n_y_p.to(self.device), n_y.to(self.device)
        except Exception as e:
            print(f"Failed to process image at {image_path}: {e}", exc_info=True)
            raise

    def add_noise(self, x, rng):
        if self.noise_type == "gauss":
            noise = torch.normal(0, self.noise_level / 255, x.shape, generator=rng).to(
                x.device
            )
            noisy = x + noise
            noisy = torch.clamp(noisy, 0, 1)
        elif self.noise_type == "poiss":
            noisy = (
                torch.poisson(self.noise_level * x, generator=rng) / self.noise_level
            )
            noisy = torch.clamp(noisy, 0, 1)
        elif self.noise_type == "speckle":
            noise = torch.randn(x.shape).to(x.device)
            noisy = x + x * noise
            noisy = torch.clamp(noisy, 0, 1)
        elif self.noise_type == "pepper":
            noisy = x.clone()
            mask = torch.rand(noisy.shape).to(x.device) < self.noise_level / 255
            noisy[mask] = 0
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")
        return noisy

    def __len__(self):
        return len(self.images)

@dataclass
class InceptionConfig:
    out_1x1: int
    red_3x3: int
    out_3x3: int
    red_5x5: int
    out_5x5: int
    out_1x1pool: int


class FFv3(nn.Module):
    def __init__(self, in_features, out_features, layers=4, dropout_rate=0.0):
        super(FFv3, self).__init__()

        intermediate_features = [in_features * (2**i) for i in range(1, layers - 1)]
        layer_sizes = [in_features] + intermediate_features + [out_features]

        sequence = []
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            sequence.append(layer)
            sequence.append(nn.LayerNorm(layer_sizes[i + 1]))
            sequence.append(nn.ReLU())
            sequence.append(nn.Dropout(dropout_rate))

        self.layers = nn.Sequential(*sequence)

    def forward(self, x):
        return self.layers(x)

class DenseResidualBlock(nn.Module):
    def __init__(self, num_channels, growth_rate, num_layers, group_size=32):
        super(DenseResidualBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_channels = num_channels + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.GroupNorm(1, layer_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        layer_channels,
                        growth_rate,
                        kernel_size=3,
                        padding=1,
                        groups=group_size,
                        bias=False,
                    ),
                )
            )
            for layer in self.layers:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )

    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(outputs, 1))
            outputs.append(out)
        return torch.cat(outputs, 1)


class DenseAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(DenseAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(1, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.Sigmoid(),
        )
        for m in self.attention:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        attention = self.attention(x)
        return torch.cat([x, attention], dim=1)


class CBAM(nn.Module):
    def __init__(self, C, r=16, k=7, p1=1, p2=2):
        super(CBAM, self).__init__()
        self.avgP = nn.AdaptiveAvgPool2d(p1)
        self.maxP = nn.AdaptiveMaxPool2d(p1)
        self.fc = nn.Sequential(
            nn.Conv2d(C, C // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // r, C, 1, bias=False),
        )
        self.sig = nn.Sigmoid()
        self.avgP_small = nn.AdaptiveAvgPool2d(p2)
        self.maxP_small = nn.AdaptiveMaxPool2d(p2)
        self.conv = nn.Conv2d(C, C, kernel_size=p2, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        avgF = self.fc(self.avgP(x))
        maxF = self.fc(self.maxP(x))
        chnlAtt = avgF + maxF

        avgF_small = self.conv(self.avgP_small(x))
        maxF_small = self.conv(self.maxP_small(x))
        spatlAtt = self.sig(avgF_small + maxF_small)
        spatlAtt = F.interpolate(
            spatlAtt, x.shape[2:], mode="bilinear", align_corners=False
        )

        return self.sig(chnlAtt) * spatlAtt * x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, config: InceptionConfig):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, config.out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, config.red_3x3, kernel_size=1),
            nn.GroupNorm(1, config.red_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.red_3x3, config.out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, config.red_5x5, kernel_size=1),
            nn.GroupNorm(1, config.red_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.red_5x5, config.out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, config.out_1x1pool, kernel_size=1),
        )
        for m in [self.branch1, self.branch2, self.branch3, self.branch4]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        outputs = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        return torch.cat(outputs, 1)


class Network(nn.Module):
    def __init__(
        self,
        n_chan,
        chan_embed=64,
        inception_config=None,
        growth_rate=32,
        num_layers=1,
        num_res_blocks=3,
        latent_dim=28,
        dropout=0.05,
    ):
        super(Network, self).__init__()
        self.latent_dim = latent_dim
        self.n_chan = n_chan

        if inception_config is None:
            inception_config = InceptionConfig(
                out_1x1=chan_embed // 4,
                red_3x3=chan_embed // 8,
                out_3x3=chan_embed // 4,
                red_5x5=chan_embed // 16,
                out_5x5=chan_embed // 8,
                out_1x1pool=chan_embed // 4,
            )

        self.conv1 = nn.Conv2d(n_chan, chan_embed, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.conv1a = nn.Conv2d(chan_embed, chan_embed, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.conv1a.weight, mode="fan_out", nonlinearity="relu")

        self.inception1 = InceptionModule(chan_embed, inception_config)
        inception_output_channels = sum(
            [
                inception_config.out_1x1,
                inception_config.out_3x3,
                inception_config.out_5x5,
                inception_config.out_1x1pool,
            ]
        )

        self.dense_attention1 = DenseAttention(inception_output_channels)
        attention_output_channels = inception_output_channels * 2

        self.conv2 = nn.Conv2d(
            attention_output_channels, chan_embed * 2, kernel_size=3, padding=1
        )
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        self.cbam = CBAM(chan_embed * 2)

        self.res_blocks = nn.ModuleList(
            [
                DenseResidualBlock(
                    chan_embed * 2 + growth_rate * num_layers * i,
                    growth_rate,
                    num_layers,
                )
                for i in range(num_res_blocks)
            ]
        )

        final_block_output_channels = (
            chan_embed * 2 + growth_rate * num_layers * num_res_blocks
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        flattened_size = final_block_output_channels * 6 * 6

        self.ff = FFv3(
            in_features=flattened_size,
            out_features=n_chan * latent_dim,
            layers=num_layers,
            dropout_rate=dropout,
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1a(x))
        x = self.inception1(x)
        x = self.dense_attention1(x)
        x = F.relu(self.conv2(x))
        x = self.cbam(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.ff(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_chan, self.latent_dim)
        return x


class MoE(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_mixtures=4,
        kernel=4,
        sharpening_factor=1,
        clip_value=5,
        dropout=0.0,
        num_heads=3,
    ):
        super(MoE, self).__init__()

        self.ch = in_channels
        self.kernel = kernel
        self.num_mixtures = num_mixtures
        self.clip_value = clip_value

        self.c = nn.Parameter(torch.tensor(4.685))
        self.α = nn.Parameter(torch.tensor(sharpening_factor, dtype=torch.float))
        self.reg_param = nn.Parameter(torch.tensor(1e-6))

        self.dropout = nn.Dropout(p=dropout)

    def grid(self, height, width):
        xx = torch.linspace(0.0, 1.0, width)
        yy = torch.linspace(0.0, 1.0, height)
        grid_x, grid_y = torch.meshgrid(xx, yy, indexing="ij")
        grid = torch.stack((grid_x, grid_y), 2).float()
        return grid.reshape(height * width, 2)

    def forward(self, height, width, params):
        positive_α = F.softplus(self.α)

        μ_x = params[:, :, : self.kernel].reshape(-1, self.kernel, 1)
        μ_y = params[:, :, self.kernel : 2 * self.kernel].reshape(-1, self.kernel, 1)
        μ = torch.cat((μ_x, μ_y), 2).view(-1, self.kernel, 2)
        Σ = params[
            :, :, 3 * self.kernel : 3 * self.kernel + self.kernel * 2 * 2
        ].reshape(-1, self.kernel, 2, 2)
        w = params[:, :, 2 * self.kernel : 3 * self.kernel].reshape(-1, self.kernel)

        Σ = Σ + self.reg_param * torch.eye(*Σ.shape[-2:]).to(Σ.device)
        Σ = torch.tril(Σ)
        Σ = torch.mul(Σ, positive_α)

        grid = self.grid(height, width).to(params.device)
        μ = μ.unsqueeze(dim=2)
        grid_expand_dim = torch.unsqueeze(torch.unsqueeze(grid, dim=0), dim=0)
        x = torch.tile(grid_expand_dim, (μ.shape[0], μ.shape[1], 1, 1))
        x_sub_μ = torch.unsqueeze(x.float() - μ.float(), dim=-1)

        e = torch.exp(
            torch.negative(
                0.5 * torch.einsum("abcli,ablm,abnm,abcnj->abc", x_sub_μ, Σ, Σ, x_sub_μ)
            )
        )

        g = torch.sum(e, dim=1, keepdim=True)
        g_max = torch.max(torch.tensor(10e-8), g)
        e_norm = torch.divide(e, g_max)

        y_hat = torch.sum(e_norm * torch.unsqueeze(w, dim=-1), dim=1)
        y_hat = torch.clamp(y_hat, min=0, max=1)
        y_hat = y_hat.view(params.shape[0], self.ch, height, width)

        return y_hat

class AutoencoderN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 28,
        num_mixtures: int = 4,  # Tune this parameter based on task complexity
        kernel: int = 4,  # Tune this for desired patch size
        sharpening_factor: int = 1,
        stride: int = 16,  # Set stride such that it covers the image efficiently
        phw: int = 64,  # Set patch height and width
        dropout: int = 0.0,
        chan_embed: int = 64,
        growth_rate: int = 32,
        layers: int = 4,
        num_res_blocks: int = 3,
    ):
        super().__init__()

        self.phw = phw
        self.stride = stride
        self.latent_dim = latent_dim

        self.encoder = Network(
            n_chan=in_channels,
            chan_embed=chan_embed,
            growth_rate=growth_rate,
            num_layers=layers,
            num_res_blocks=num_res_blocks,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        self.decoder = MoE(
            in_channels=in_channels,
            num_mixtures=num_mixtures,
            kernel=kernel,
            sharpening_factor=sharpening_factor,
            dropout=dropout,
            num_heads=4,
        )

    @staticmethod
    def reconstruct(patches, shape, khw, stride):
        B, C, H, W = shape

        patches = patches.view(B, -1, C * khw * khw).transpose(1, 2)

        y_hat = F.fold(
            patches, output_size=(H, W), kernel_size=(khw, khw), stride=stride
        )

        one_patch = torch.ones((1, khw * khw, patches.shape[-1]), device=patches.device)
        overlap_count = F.fold(
            one_patch, output_size=(H, W), kernel_size=(khw, khw), stride=stride
        )

        y_hat /= overlap_count.clamp(min=1)

        return y_hat

    def forward(self, x, shape):
        if len(x.shape) == 5:
            x = x.view(-1, *x.size()[2:])

        bs = int(len(x) / 1024)
        x = torch.split(x, bs, dim=0)

        encoded = [self.encoder(batch) for batch in x]
        _, C, H, W = shape
        row, col = (W - self.phw) // self.stride + 1, (H - self.phw) // self.stride + 1

        params = torch.cat(encoded, dim=0)
        params = params.view(-1, row, col, C, self.latent_dim)
        params = torch.split(params.view(-1, *params.size()[3:]), bs, dim=0)

        decoded = [self.decoder(self.phw, self.phw, param) for param in params]
        decoded = torch.cat(decoded, dim=0)
        y_hat = self.reconstruct(decoded, shape, self.phw, self.stride)

        return y_hat

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x1, x2): # x1 (bs,out_ch,w1,h1) x2 (bs,in_ch,w2,h2)
        x2 = self.up_scale(x2) # (bs,out_ch,2*w2,2*h2)
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # (bs,out_ch,w1,h1)
        x = torch.cat([x2, x1], dim=1) # (bs,2*out_ch,w1,h1)
        return x

class up_layer(nn.Module):
    def __init__(self, in_ch, out_ch): # !! 2*out_ch = in_ch !!
        super(up_layer, self).__init__()
        self.up = up(in_ch, out_ch)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2): # x1 (bs,out_ch,w1,h1) x2 (bs,in_ch,w2,h2)
        a = self.up(x1, x2) # (bs,2*out_ch,w1,h1)
        x = self.conv(a) # (bs,out_ch,w1,h1) because 2*out_ch = in_ch
        return x

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_layer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x

class UNetV2(nn.Module):
    def __init__(self,
                 in_channels=1,
                 latent_dim:int=28,
                 layers:int=4,
                 dropout_rate:float=0.0
                 ):
        super(UNetV2, self).__init__()


        self.latent_dim = latent_dim
        self.in_channels = in_channels

        self.conv1 = double_conv(in_channels, 64)
        self.down1 = down_layer(64, 128)
        self.down2 = down_layer(128, 256)
        self.down3 = down_layer(256, 512)
        self.down4 = down_layer(512, 1024)

        self.up1 = up_layer(1024, 512)
        self.up2 = up_layer(512, 256)
        self.up3 = up_layer(256, 128)
        self.up4 = up_layer(128, 64)
        self.final_conv = nn.Conv2d(64, 3, 1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FFv3(in_features=3, out_features=in_channels * latent_dim, layers=layers, dropout_rate=dropout_rate)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1_up = self.up1(x4, x5)
        x2_up = self.up2(x3, x1_up)
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up)
        x_final = self.final_conv(x4_up)
        x_pooled = self.adaptive_pool(x_final)
        x_out = self.fc(x_pooled.view(x_pooled.size(0), -1))
        return x_out.view(-1,  self.in_channels, self.latent_dim)

class MoE_v1(nn.Module):
    def __init__(
        self,
        ch=1,
        num_mixtures=4,
        kernel=4,
        sharpening_factor=1,
        clip_value=5,
        dropout=0.0,
        num_heads=3,
    ):
        super(MoE_v1, self).__init__()

        self.ch = ch
        self.kernel = kernel
        self.num_mixtures = num_mixtures
        self.clip_value = clip_value

        self.α = nn.Parameter(torch.tensor(sharpening_factor, dtype=torch.float))

    def grid(self, height, width):
        xx = torch.linspace(0.0, 1.0, width)
        yy = torch.linspace(0.0, 1.0, height)
        grid_x, grid_y = torch.meshgrid(xx, yy, indexing="ij")
        grid = torch.stack((grid_x, grid_y), 2).float()
        return grid.reshape(height * width, 2)

    @staticmethod
    def _soft_clipping(x, beta=10):
        return 1 / (1 + torch.exp(-beta * (x - 0.5)))
    
    # @staticmethod
    # def sample_image_grid(
    #     shape: tuple[int, ...],
    #     device: torch.device = torch.device("cpu"),
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Get normalized (range 0 to 1) coordinates and integer indices for an image."""
    #     indices = [torch.arange(length, device=device) for length in shape]
    #     stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)
        
    #     coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    #     coordinates = reversed(coordinates)
    #     coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)
        
    #     return coordinates.reshape(-1, 2), stacked_indices
    
    def forward(self, height, width, params):

        # grid, _ = self.sample_image_grid((height, width), device=params.device)

        μ_x = params[:, :, : self.kernel].reshape(-1, self.kernel, 1)
        μ_y = params[:, :, self.kernel : 2 * self.kernel].reshape(-1, self.kernel, 1)
        μ = torch.cat((μ_x, μ_y), 2).view(-1, self.kernel, 2)
        Σ = params[
            :, :, 3 * self.kernel : 3 * self.kernel + self.kernel * 2 * 2
        ].reshape(-1, self.kernel, 2, 2)
        w = params[:, :, 2 * self.kernel : 3 * self.kernel].reshape(-1, self.kernel)

        Σ = torch.tril(Σ)
        # Σ = torch.mul(Σ, self.α)
        # Σ = torch.mul(Σ, 1.2)

        grid = self.grid(height, width).to(params.device)
        μ = μ.unsqueeze(dim=2)
        grid_expand_dim = torch.unsqueeze(torch.unsqueeze(grid, dim=0), dim=0)
        x = torch.tile(grid_expand_dim, (μ.shape[0], μ.shape[1], 1, 1))
        x_sub_μ = torch.unsqueeze(x.float() - μ.float(), dim=-1)

        e = torch.exp(
            torch.negative(
                0.5 * torch.einsum("abcli,ablm,abnm,abcnj->abc", x_sub_μ, Σ, Σ, x_sub_μ)
            )
        )

        g = torch.sum(e, dim=1, keepdim=True)
        g_max = torch.max(torch.tensor(10e-8), g)
        e_norm = torch.divide(e, g_max)

        y_hat = torch.sum(e_norm * torch.unsqueeze(w, dim=-1), dim=1)
        y_hat = torch.clamp(y_hat, min=0, max=1)

        # y_hat = y_hat.view(params.shape[0], self.ch, height, width)
        y_hat = y_hat.view(-1, self.ch, height, width)

        return y_hat

class UNet(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 latent_dim:int=28, 
                 size:int=16
                 ):
        super(UNet, self).__init__()

  
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        self.conv1 = double_conv(in_channels, 64)
        self.down1 = down_layer(64, 128)
        self.down2 = down_layer(128, 256)
        self.down3 = down_layer(256, 512)
        self.down4 = down_layer(512, 1024)
        
        self.up1 = up_layer(1024, 512)
        self.up2 = up_layer(512, 256)
        self.up3 = up_layer(256, 128)
        self.up4 = up_layer(128, 64)
        self.last_conv = nn.Conv2d(64, latent_dim, 1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
                nn.Linear((latent_dim*size**2), in_channels * latent_dim),
                nn.BatchNorm1d(in_channels * latent_dim),
                nn.ReLU())
            
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1_up = self.up1(x4, x5)
        x2_up = self.up2(x3, x1_up)
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up) 
        x_last = self.last_conv(x4_up)

        xf = self.flatten(x_last)
        x = self.fc1(xf)
        # x = x.view(-1, self.depth, self.latent_dim)
        x = x.view(-1, self.in_channels, self.latent_dim)
        return x

class AutoencoderU(nn.Module):
    def __init__(
        self,
        in_channels:int=1,
        latent_dim:int=28,
        num_mixtures:int=4,
        kernel:int=4,
        sharpening_factor:int=1,
        overlap:int=16,
        phw:int=64,
        dropout:int=0.0,
        scale_factor:int=16,
        pre_trained:str=None,
    ):
        super().__init__()

        self.phw = phw
        self.overlap = overlap
        self.latent_dim = latent_dim
        self.scale_factor = scale_factor

        self.encoder= UNet(in_channels=in_channels,latent_dim=latent_dim, size=phw)

        if pre_trained:
            model = torch.load(pre_trained)
            state_dict = {
                k.replace("encoder.", ""): v for k, v in model["state_dict"].items()
            }
            self.encoder.load_state_dict(state_dict)


        self.decoder = MoE_v1(
            ch=in_channels,
            num_mixtures=num_mixtures,
            kernel=kernel,
            sharpening_factor=sharpening_factor,
            dropout=dropout,
            num_heads=4,
        )

    @staticmethod
    def reconstruct(blocks, original_dims, block_size, overlap):
        
        batch_size, num_channels, height, width = original_dims
        step = block_size - overlap
        device = blocks.device

        recon_images = torch.zeros(batch_size, num_channels, height, width).to(device)
        count_matrix = torch.zeros(batch_size, num_channels, height, width).to(device)

        num_blocks_per_row = (width - block_size) // step + 1
        num_blocks_per_column = (height - block_size) // step + 1
        num_blocks_per_image = num_blocks_per_row * num_blocks_per_column

        for b in range(batch_size):
            idx_start = b * num_blocks_per_image
            current_blocks = blocks[idx_start:idx_start + num_blocks_per_image]
            idx = 0
            for i in range(0, height - block_size + 1, step):
                for j in range(0, width - block_size + 1, step):
                    recon_images[b, :, i:i+block_size, j:j+block_size] += current_blocks[idx]
                    count_matrix[b, :, i:i+block_size, j:j+block_size] += 1
                    idx += 1

        recon_images /= count_matrix.clamp(min=1)
        return recon_images

    def forward(self, x, shape):
        if len(x.shape) == 5:
            x = x.view(-1, *x.size()[2:])

        bs = len(x) // 1024
      
        encoded = self.encoder(x)

        B, C, H, W = shape
        scaled_phw = self.phw * self.scale_factor 
 
        params = torch.split(encoded, bs, dim=0)
        
        decoded = [self.decoder(scaled_phw, scaled_phw, param) for param in params]
        decoded = torch.cat(decoded, dim=0)


        y_hat = self.reconstruct(decoded, (B, C, H * self.scale_factor, W * self.scale_factor), scaled_phw, self.overlap*self.scale_factor)
        

        return y_hat


class FFv4(nn.Module):
    def __init__(self, in_features, out_features, layers=4, dropout_rate=0.0):
        super(FFv4, self).__init__()

        intermediate_features = [in_features * 2**i for i in range(1, layers - 1)]
        layer_sizes = [in_features] + intermediate_features + [out_features]

        sequence = []
        for i in range(len(layer_sizes) - 1):
            sequence.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            sequence.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            sequence.append(nn.ReLU())
            sequence.append(nn.Dropout(dropout_rate))

        self.layers = nn.Sequential(*sequence)

    def forward(self, x):
        return self.layers(x)

class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        latent_dim=28,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_chans = in_chans
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.fc = FFv4(
            in_features=dims[-1],
            out_features=in_chans * latent_dim,
            layers=4,
            dropout_rate=drop_path_rate,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        x = x.view(-1, self.in_chans, self.latent_dim)
        return x

class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class AutoencoderC(nn.Module):
    def __init__(
        self,
        in_channels=3,
        latent_dim=28,
        num_mixtures=4,
        kernel=4,
        sharpening_factor=1,
        stride=16,
        phw=64,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        dropout=0.0,
    ):
        super().__init__()

        self.phw = phw
        self.stride = stride
        self.latent_dim = latent_dim

        self.encoder = ConvNeXt(
            in_chans=in_channels, depths=depths, dims=dims, latent_dim=latent_dim
        )

        self.decoder = MoE(
            in_channels=in_channels,
            num_mixtures=num_mixtures,
            kernel=kernel,
            sharpening_factor=sharpening_factor,
            dropout=dropout,
            num_heads=4,
        )

    @staticmethod
    def reconstruct(patches, shape, khw, stride):
        B, C, H, W = shape

        patches = patches.view(B, -1, C * khw * khw).transpose(1, 2)

        y_hat = F.fold(
            patches, output_size=(H, W), kernel_size=(khw, khw), stride=stride
        )

        one_patch = torch.ones((1, khw * khw, patches.shape[-1]), device=patches.device)
        overlap_count = F.fold(
            one_patch, output_size=(H, W), kernel_size=(khw, khw), stride=stride
        )

        y_hat /= overlap_count.clamp(min=1)

        return y_hat

    def forward(self, x, shape):
        if len(x.shape) == 5:
            x = x.view(-1, *x.size()[2:])

        bs = int(len(x) / 1024)
        x = torch.split(x, bs, dim=0)

        encoded = [self.encoder(batch) for batch in x]

        _, C, H, W = shape
        row, col = (W - self.phw) // self.stride + 1, (H - self.phw) // self.stride + 1

        params = torch.cat(encoded, dim=0)
        params = params.view(-1, row, col, C, self.latent_dim)
        params = torch.split(params.view(-1, *params.size()[3:]), bs, dim=0)

        decoded = [self.decoder(self.phw, self.phw, param) for param in params]
        decoded = torch.cat(decoded, dim=0)

        y_hat = self.reconstruct(decoded, shape, self.phw, self.stride)

        return y_hat


def visualize_data(L, H, G):
    """
    Visualize restoration, noisy, and ground truth images with advanced scientific representation.

    Parameters:
    L (tensor): Restored image
    H (tensor): Noisy image
    G (tensor): Ground truth image
    """
    
    L_np = L.cpu().numpy()[0][0]
    H_np = H.cpu().numpy()[0][0]
    G_np = G.cpu().numpy()[0][0]

    if L_np.ndim == 3 and L_np.shape[0] == 1:
        L_np = L_np[0]
    if H_np.ndim == 3 and H_np.shape[0] == 1:
        H_np = H_np[0]
    if G_np.ndim == 3 and G_np.shape[0] == 1:
        G_np = G_np[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(H_np, cmap='gray', aspect='auto')
    axes[0].set_title('Noisy Image', fontsize=14)
    axes[0].grid(True, color='white', linestyle='--', linewidth=0.5)
    cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
    cbar0.set_label('Intensity', fontsize=10)
    axes[0].set_xlabel('X-axis (pixels)', fontsize=10)
    axes[0].set_ylabel('Y-axis (pixels)', fontsize=10)
    axes[0].tick_params(axis='both', which='major', labelsize=8)

    im1 = axes[1].imshow(L_np, cmap='gray', aspect='auto')
    axes[1].set_title('Restored Image', fontsize=14)
    axes[1].grid(True, color='white', linestyle='--', linewidth=0.5)
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    cbar1.set_label('Intensity', fontsize=10)
    axes[1].set_xlabel('X-axis (pixels)', fontsize=10)
    axes[1].set_ylabel('Y-axis (pixels)', fontsize=10)
    axes[1].tick_params(axis='both', which='major', labelsize=8)

    im2 = axes[2].imshow(G_np, cmap='gray', aspect='auto')
    axes[2].set_title('Ground Truth Image', fontsize=14)
    axes[2].grid(True, color='white', linestyle='--', linewidth=0.5)
    cbar2 = fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
    cbar2.set_label('Intensity', fontsize=10)
    axes[2].set_xlabel('X-axis (pixels)', fontsize=10)
    axes[2].set_ylabel('Y-axis (pixels)', fontsize=10)
    axes[2].tick_params(axis='both', which='major', labelsize=8)

    # Adding a unified title
    fig.suptitle('Comparison of Noisy, Restored, and Ground Truth Images', fontsize=16, y=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

@click.command()
@click.option("--kernels", default=9, help="Number of Kernels", type=int)
@click.option("--mixtures", help="Number of Experts", default=9, type=int)
@click.option("--sharpening_factor", default=1, type=int)
@click.option("--chan_embed", default=64, help="Channel embeded.", type=int)
@click.option("--layers", default=1, help="Number of layers.", type=int)
@click.option("--growth_rate", default=32, help="Growth rate.", type=int)
@click.option("--num_res_blocks", default=3, help="Number of res blocks.", type=int)
@click.option("--in_channels", default=1, help="Color channels.", type=int)
@click.option("--dropout", default=0.0, help="Dropout.", type=float)
@click.option(
    "--convert",
    default="L",
    help="Color Type.",
    type=click.Choice(
        [
            "L",
            "RGB",
        ]
    ),
)



# @click.option(
#     "--resume",
#     default="./zoo/conv-next-moe/convnext_hw-128_pHW-64_stride-8_color-L_opti-Adam_criter-MSE_lr-0.001_lr_min-0.0001_epochs-100_kernel-16_z-112_noise-lvl-25_noise-typ-gauss-jid-5394519_psnr22.76_ssim0.66_e5.pth",
#     help="Pre-trained.",
#     type=str,
# )

@click.option(
    "--resume",
    default="/mnt/d/zoo/Unet-MoE-v1/2023-09-27_00-30-02/Unet-MoE-v1_kernel-9_khw-16-Optimizer-Adam-Criterion-MSE-lr-0.001-lr_min-0.0001-warmup_epochs-3-epochs-50-batch_size-None-size-128-stride-4-kernel_params-7-scale_factor-1-sigma-15-noise_typegaussian_latest.pth",
    help="Pre-trained.",
    type=str,
)

# @click.option(
#     "--resume",
#     default="/mnt/d/zoo/cnn-inception-moe/cnn-inception-moe_hw-128_pHW-32_stride-8_color-L_opti-Adam_criter-MSE_lr-0.001_lr_min-0.0001_epochs-100_kernel-16_z-112_noise-lvl-50_noise-typ-gauss-jid-5395540_last_e12.pth",
#     help="Pre-trained.",
#     type=str,
# )
# @click.option(
#     "--resume",
#     default="/mnt/d/zoo/cnn-inception-moe/cnn-inception-moe_hw-128_pHW-32_stride-8_color-L_opti-Adam_criter-MSE_lr-0.001_lr_min-0.0001_epochs-100_kernel-16_z-112_noise-lvl-25_noise-typ-poiss-jid-5395536_last_e11.pth",
#     help="Pre-trained.",
#     type=str,
# )
@click.option(
    "--test_dir",
    default="/mnt/e/Medical/BrainMRI/Training/",
    help="Directory containing OpenImages dataset",
)
@click.option(
    "--output_dir",
    default="/mnt/e/out/BrainMRI",
    help="Directory containing OpenImages dataset",
)
@click.option("--stride", default=12, type=int)
@click.option("--overlap", default=12, type=int)
@click.option("--scale_factor", default=8, type=int)
@click.option("--hw", default=128, help="Image Size.", type=int)
@click.option("--phw", default=16, help="Patch Size.", type=int)
@click.option("--rank", default=0, help="Rank.", type=int)
@click.option(
    "--noise_type",
    default="gauss",
    help="Noise types.",
    type=click.Choice(
        [
            "gauss",
            "poiss",
            "speckle",
            "pepper",
        ]
    ),
)
@click.option(
    "--noise_level", default=25, help="Noise levels.", type=click.IntRange(min=1)
)
@click.option("--model_name", default="conv-next-moe", help="Model.", type=str)
def main(**kwargs):
    c = EasyDict(kwargs)
    torch.cuda.set_device(c.rank)
    device = torch.device("cuda", c.rank)
    current_time: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(c.model_name + c.output_dir, current_time)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = Compose(
        [
            ToTensor(),
        ]
    )

    test_dataset = OpenImageDataset(
        dir=c.test_dir,
        test=True,
        convert=c.convert,
        stride=c.stride,
        phw=c.phw,
        noise_type=c.noise_type,
        noise_level=c.noise_level,
        transform=transform,
        device=device, 
        length=1
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
    )

    z = 2 * c.kernels + 4 * c.mixtures + c.kernels
    # model1 = Autoencoder(
    #     in_channels=c.in_channels,
    #     chan_embed=c.chan_embed,
    #     growth_rate=c.growth_rate,
    #     latent_dim=z,
    #     kernel=c.kernels,
    #     num_mixtures=c.mixtures,
    #     phw=c.phw,
    #     stride=c.stride,
    #     layers=c.layers,
    #     dropout=c.dropout,
    # )
    # model1.eval()
    # model1.to(device)
    # p_number = network_parameters(model1)

    # if c.resume:
    #     model1.load_state_dict(torch.load(c.resume, map_location=device))
    #     print(f"Pre-trained model loaded: {c.resume}")
    
    model2 = AutoencoderU(
        in_channels=c.in_channels,
        latent_dim=z,
        kernel=c.kernels,
        num_mixtures=c.mixtures,
        phw=c.phw,
        overlap=c.overlap,
        dropout=c.dropout,
        pre_trained=c.resume,
        scale_factor=c.scale_factor,
    )

    model2.eval()
    model2.to(device)
    p_number = network_parameters(model2)

    # model3 = AutoencoderC(
    #     in_channels=c.in_channels,
    #     latent_dim=z,
    #     kernel=c.kernels,
    #     num_mixtures=c.mixtures,
    #     phw=c.phw,
    #     stride=c.stride,
    # )

    # if c.resume:
    #     model3.load_state_dict(torch.load(c.resume, map_location=device))
    #     print(f"Pre-trained model loaded: {c.resume}")

    # model3.eval()
    # model3.to(device)
    # p_number = network_parameters(model3)

    model_name = f"{c.model_name}_pHW-{c.phw}_stride-{c.stride}_kernel-{c.kernels}_z-{z}_noise-lvl-{c.noise_level}_noise-typ-{c.noise_type}"
    psnr_vals, ssim_vals = [], []
    restored_imgs = []

    ground_truth_imgs = []
    noisy_imgs = []
    restored_imgs = []
    psnr_vals = []
    ssim_vals = []
    psnr_vals_n = []
    ssim_vals_n = []
    with torch.no_grad():
        for data in test_dataloader:
            y_gt, y_n_p, y_n = data  
            restored = model2(y_n_p, y_gt.size())
            visualize_data(restored, y_n, y_gt)
            ground_truth_imgs.append(y_gt.cpu())
            noisy_imgs.append(y_n.cpu())
            restored_imgs.append(restored.cpu())
            psnr_vals.append(piq.psnr(restored, y_gt).item())
            ssim_vals.append(piq.ssim(restored, y_gt).item())
            psnr_vals_n.append(piq.psnr(y_n, y_gt).item())
            ssim_vals_n.append(piq.ssim(y_n, y_gt).item())

    
    font = ImageFont.load_default()
    annotated_imgs = []

    for gt, noisy, restored, psnr_n, ssim_n, psnr_r, ssim_r in zip(ground_truth_imgs, noisy_imgs, restored_imgs, psnr_vals_n, ssim_vals_n, psnr_vals, ssim_vals):
        for img, title, psnr, ssim in zip([gt, noisy, restored], ["Ground Truth", "Noisy", "Restored"], [None, psnr_n, psnr_r], [None, ssim_n, ssim_r]):
            img_tensor = ToTensor()(img.squeeze()) if not isinstance(img, torch.Tensor) else img
            if img_tensor.dim() == 2:  
                img_tensor = img_tensor.unsqueeze(0)
            if img_tensor.dim() == 4:  
                img_tensor = img_tensor[0]
            if img_tensor.dim() == 3 and img_tensor.size(0) == 1: 
                img_tensor = img_tensor.repeat(3, 1, 1)
            
            img_pil = ToPILImage()(img_tensor)
            draw = ImageDraw.Draw(img_pil)
            
            text = title  
            if psnr is not None and ssim is not None:
                text += f", PSNR: {psnr:.4f}, SSIM: {ssim:.4f}"
            
            text_position = (10, img_pil.height - 10)  
            draw.text(text_position, text, font=font, fill="green")
            
            annotated_imgs.append(ToTensor()(img_pil))
        
        
        error_map = torch.abs(gt - restored)
        error_map /= error_map.max()
        

        error_map_pil = ToPILImage()(error_map.squeeze())
        
        error_map_colored = cm.viridis(np.array(error_map_pil))
        error_map_colored = (error_map_colored[:, :, :3] * 255).astype(np.uint8)
        error_map_colored_pil = Image.fromarray(error_map_colored)
        
        draw = ImageDraw.Draw(error_map_colored_pil)
        text = "Error Map"
        text_position = (10, error_map_colored_pil.height - 10)
        draw.text(text_position, text, font=font, fill="green")
        
        annotated_imgs.append(ToTensor()(error_map_colored_pil))

    grid = make_grid(annotated_imgs, nrow=4) 
    save_image(grid, f"{output_dir}/{model_name}.png")
    print(f"Test images saved: {output_dir}/{model_name}.png")
if __name__ == "__main__":
    main()
