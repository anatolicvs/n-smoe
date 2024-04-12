import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

class ForwardPool(nn.Module):
    def __init__(self, in_features, out_features, layers=4, dropout_rate=0.1):
        super().__init__()

        intermediate_features = [in_features * 2**i for i in range(1, layers - 1)]
        layer_sizes = [in_features] + intermediate_features + [out_features]

        sequence = []
        for i in range(len(layer_sizes) - 1):
            sequence.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            sequence.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            sequence.append(nn.SELU())
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

class Encoder(nn.Module):
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
        self.fc = ForwardPool(
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
        # LL, LH, HL, HH = dwt_channelwise(x)
        x = self.forward_features(x)
        x = self.fc(x)
        x = x.view(-1, self.in_chans, self.latent_dim)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_mixtures=4,
        kernel=4,
        sharpening_factor=1,
        clip_value=5,
    ):
        super().__init__()

        self.ch = in_channels
        self.kernel = kernel
        self.num_mixtures = num_mixtures
        self.clip_value = clip_value

        self.α = sharpening_factor

    def grid(self, height, width):
        xx = torch.linspace(0.0, 1.0, width)
        yy = torch.linspace(0.0, 1.0, height)
        grid_x, grid_y = torch.meshgrid(xx, yy, indexing="ij")
        grid = torch.stack((grid_x, grid_y), 2).float()
        return grid.reshape(height * width, 2)

    @staticmethod
    def _soft_clipping(x, beta=10):
        return 1 / (1 + torch.exp(-beta * (x - 0.5)))

    def forward(self, height, width, params):
        
        μ_x = params[:, :, : self.kernel].reshape(-1, self.kernel, 1)
        μ_y = params[:, :, self.kernel : 2 * self.kernel].reshape(-1, self.kernel, 1)
        μ = torch.cat((μ_x, μ_y), 2).view(-1, self.kernel, 2)
        Σ = params[
            :, :, 3 * self.kernel : 3 * self.kernel + self.kernel * 2 * 2
        ].reshape(-1, self.kernel, 2, 2)
        w = params[:, :, 2 * self.kernel : 3 * self.kernel].reshape(-1, self.kernel)

        Σ = torch.tril(Σ)
        Σ = torch.mul(Σ, self.α)

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

        
        y_hat = y_hat.view(-1, self.ch, height, width)

        return y_hat

class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        latent_dim=28,
        num_mixtures=4,
        scale_factor=1,
        kernel=4,
        sharpening_factor=1,
        stride=16,
        phw=64,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ):
        super().__init__()

        self.phw = phw
        self.stride = stride
        self.latent_dim = latent_dim
        self.scale_factor = scale_factor

        self.encoder = Encoder(
            in_chans=in_channels, depths=depths, dims=dims, latent_dim=latent_dim
        )

        self.decoder = Decoder(
            in_channels=in_channels,
            num_mixtures=num_mixtures,
            kernel=kernel,
            sharpening_factor=sharpening_factor
        )

    @staticmethod
    def _reconstruct(decoded_patches, row, col, in_channels, stride, patch_size, scale_factor, height, width, batch_size=1, device='cuda:0'):
        i_indices = torch.arange(0, row * stride, stride, device=device) * scale_factor
        j_indices = torch.arange(0, col * stride, stride, device=device) * scale_factor
        start_i_indices, start_j_indices = torch.meshgrid(i_indices, j_indices, indexing='ij')

        end_i_indices = (start_i_indices + patch_size).flatten()
        end_j_indices = (start_j_indices + patch_size).flatten()
        
        out = torch.zeros(batch_size, in_channels, height * scale_factor, width * scale_factor, device=device)
        count = torch.zeros_like(out)

        patches_per_image = row * col

        for b in range(batch_size):
            for i in range(patches_per_image):
                patch_idx = b * patches_per_image + i

                start_i, end_i = start_i_indices.flatten()[i], end_i_indices[i]
                start_j, end_j = start_j_indices.flatten()[i], end_j_indices[i]
                
                end_i = min(end_i, height * scale_factor)
                end_j = min(end_j, width * scale_factor)
                
                patch = decoded_patches[patch_idx, :, :end_i-start_i, :end_j-start_j]
                out[b, :, start_i:end_i, start_j:end_j] += patch
                count[b, :, start_i:end_i, start_j:end_j] += 1

        out /= count.clamp(min=1) 
        return out
    
    def forward(self, x, shape):
        if len(x.shape) == 5:
            x = x.view(-1, *x.size()[2:])

        encoded = self.encoder(x)
        B, C, H, W = shape
        scaled_phw = self.phw * self.scale_factor  
        row, col = (W - self.phw) // self.stride + 1, (H - self.phw) // self.stride + 1
        
        params = encoded.view(-1, row, col, C, self.latent_dim)
        params = params.view(-1, *params.size()[3:])
        decoded = self.decoder(scaled_phw, scaled_phw, params)
        
        y_hat = self._reconstruct(decoded, row, col, C, self.stride, self.phw, self.scale_factor, H, W, batch_size=B)
        return y_hat