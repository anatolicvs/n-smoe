import functools
import math
from dataclasses import dataclass
from typing import Generic, Literal, TypedDict, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange

import torchvision.transforms as transforms
from PIL import Image

class MullerResizer(nn.Module):
    """Learned Laplacian resizer in PyTorch, fixed Gaussian blur for channel handling."""
    def __init__(self, base_resize_method='bilinear', antialias=False,
                 kernel_size=5, stddev=1.0, num_layers=2, avg_pool=False,
                 dtype=torch.float32, init_weights=None, name='muller_resizer'):
        super(MullerResizer, self).__init__()
        self.name = name
        self.base_resize_method = base_resize_method
        self.antialias = antialias  # Note: PyTorch does not support antialiasing in resizing.
        self.kernel_size = kernel_size
        self.stddev = stddev
        self.num_layers = num_layers
        self.avg_pool = avg_pool
        self.dtype = dtype

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for layer in range(num_layers):
            weight = nn.Parameter(torch.zeros(1, dtype=dtype) if init_weights is None else torch.tensor([init_weights[2*layer]], dtype=dtype))
            bias = nn.Parameter(torch.zeros(1, dtype=dtype) if init_weights is None else torch.tensor([init_weights[2*layer+1]], dtype=dtype))
            self.weights.append(weight)
            self.biases.append(bias)

    def _base_resizer(self, inputs, target_size):
        if self.avg_pool:
            stride_h = inputs.shape[2] // target_size[0]
            stride_w = inputs.shape[3] // target_size[1]
            if stride_h > 1 and stride_w > 1:
                inputs = F.avg_pool2d(inputs, kernel_size=(stride_h, stride_w), stride=(stride_h, stride_w))
        return F.interpolate(inputs, size=target_size, mode=self.base_resize_method, align_corners=False)

    def _gaussian_blur(self, inputs):
        sigma = max(self.stddev, 0.5)  # Ensure sigma is not too small
        radius = self.kernel_size // 2
        kernel_size = 2 * radius + 1
        x_coord = torch.arange(kernel_size, dtype=inputs.dtype, device=inputs.device) - radius
        y_grid = x_coord.repeat(kernel_size, 1)
        x_grid = x_coord.view(-1, 1).repeat(1, kernel_size)
        xy_grid = torch.sqrt(x_grid**2 + y_grid**2)
        kernel = torch.exp(-xy_grid**2 / (2 * sigma**2))
        kernel_sum = kernel.sum()
        if kernel_sum.item() == 0:
            kernel += 1e-10  
        kernel /= kernel_sum

        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(inputs.shape[1], 1, 1, 1)
        blurred = F.conv2d(inputs, kernel, padding=radius, groups=inputs.shape[1])
        return blurred

    def forward(self, inputs, target_size):
        inputs = inputs.to(dtype=self.dtype)
        net = self._base_resizer(inputs, target_size)
        for weight, bias in zip(self.weights, self.biases):
            blurred = self._gaussian_blur(inputs)
            residual_image = blurred - inputs
            resized_residual = self._base_resizer(residual_image, target_size)
            scaled_residual = weight * resized_residual + bias
            net += F.relu(scaled_residual.clamp(min=0, max=1))
            inputs = blurred
        return net

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])

class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads_channels, output_dim=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        Ho, Wo = H // self.patch_size, W // self.patch_size
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size ** 2)
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type='ln'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0,selfatt=True,
        kv_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True),
                            norm_type='ln'
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout), norm_type='ln'),
                    ]
                )
            )

    def forward(self, x, **kwargs):
        for attn, ff in self.layers:
            attn_out, _ = attn(x, x, x)
            x = x + attn_out
            x = x + ff(x, **kwargs)
        return x

class BatchedViews(TypedDict):
    image: torch.Tensor

T = TypeVar("T")

class Backbone(nn.Module, Generic[T]):
    def __init__(self, cfg: T):
        super().__init__()
        self.cfg = cfg

    def forward(self, context: BatchedViews) -> torch.Tensor:
        raise NotImplementedError

    @property
    def d_out(self) -> int:
        raise NotImplementedError

@dataclass
class BackboneResnetCfg:
    name: Literal["resnet"]
    model: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "dino_resnet50"]
    num_layers: int
    use_first_pool: bool
    d_in: int
    d_out: int

class BackboneResnet(Backbone[BackboneResnetCfg]):
    def __init__(self, cfg: BackboneResnetCfg,):
        super().__init__(cfg)
        
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.model = getattr(torchvision.models, cfg.model)(pretrained=False, norm_layer=norm_layer)

        self.model.conv1 = nn.Conv2d(cfg.d_in, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size, stride=self.model.conv1.stride, padding=self.model.conv1.padding, bias=False)

        self.projections = nn.ModuleDict()
        previous_output_channels = self.model.conv1.out_channels  
        self.projections['layer0'] = nn.Conv2d(previous_output_channels, cfg.d_out, 1)

        layers = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        for i, layer_group in enumerate(layers[:cfg.num_layers - 1]):
            output_channels = layer_group[-1].conv3.out_channels if hasattr(layer_group[-1], 'conv3') else layer_group[-1][-1].out_channels
            self.projections[f'layer{i+1}'] = nn.Conv2d(output_channels, cfg.d_out, 1)

    def forward(self, context: BatchedViews) -> torch.Tensor:
        x = context['image']
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        
        features = [self.projections['layer0'](x)]
        layers = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        for index in range(1, self.cfg.num_layers):
            x = layers[index - 1](x)
            features.append(self.projections[f'layer{index}'](x))
        
        h, w = context['image'].shape[2:]
        features = [F.interpolate(feature, (h, w), mode='bilinear', align_corners=True) for feature in features]
        output = torch.stack(features).sum(dim=0)
        return output

    @property
    def d_out(self) -> int:
        return self.cfg.d_out

@dataclass
class EncoderConfig:
    in_chans: int 
    latent_dim: int
    embed_dim: int
    depth: int
    heads: int
    mlp_dim: int
    dropout: float
    avg_pool: bool
    scale_factor: int
    resizer_num_layers: int
    patch_size: int
    backbone_cfg: BackboneResnetCfg 


class Encoder(Backbone[EncoderConfig]):
    def __init__(self, cfg: EncoderConfig):
        super().__init__(cfg)
        self.in_chans = cfg.in_chans
        self.latent_dim = cfg.latent_dim
        self.embed_dim = cfg.embed_dim
        self.scale_factor = cfg.scale_factor

        if cfg.backbone_cfg is not None:
            self.backbone = BackboneResnet(cfg.backbone_cfg)
        else:
            self.backbone = None

        self.patch_embed = PatchEmbed(patch_size=cfg.patch_size, in_chans=cfg.latent_dim, embed_dim=cfg.embed_dim)
        self.transformer = Transformer(cfg.embed_dim, cfg.depth, cfg.heads, cfg.mlp_dim, cfg.dropout)
        
        self.fc = nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.in_chans * cfg.latent_dim),
                nn.ReLU())

        self.resizer = MullerResizer(
            base_resize_method='bicubic', kernel_size=5, stddev=1.0, num_layers=cfg.resizer_num_layers,
            dtype=torch.float32, avg_pool=cfg.avg_pool
        )

    def _interpolate(self, x, scale_factor):
        _, _, H, W = x.size()
        target_h = int(H * scale_factor)
        target_w = int(W * scale_factor)
        target_size = (target_h, target_w)
        x_resized = self.resizer(x, target_size)
        return x_resized

    def forward(self, x):
        x = self._interpolate(x, self.scale_factor)
        
        x = self.backbone({'image': x})
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.fc(x)
        x = rearrange(x, "b n c -> b c n")
        x = torch.mean(x, dim=2, keepdim=True)
        x = rearrange(x, 'b (c latent) 1 -> b c latent', c=self.in_chans, latent=self.latent_dim)
        return x


@dataclass
class MoEConfig:
    in_chans: int = 1
    num_mixtures: int = 4
    kernel: int = 4
    sharpening_factor: float = 1.0

class MoE(Backbone[MoEConfig]):
    def __init__(
        self,
       cfg: MoEConfig
    ):
        super(MoE, self).__init__(cfg)

        self.ch = cfg.in_chans
        self.kernel = cfg.kernel
        self.num_mixtures = cfg.num_mixtures
        self.α = cfg.sharpening_factor

    def grid(self, height, width):
        xx = torch.linspace(0.0, 1.0, width)
        yy = torch.linspace(0.0, 1.0, height)
        grid_x, grid_y = torch.meshgrid(xx, yy, indexing="ij")
        grid = torch.stack((grid_x, grid_y), 2).float()
        return grid.reshape(height * width, 2)
 
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


@dataclass
class AutoencoderConfig:
    EncoderConfig: EncoderConfig
    DecoderConfig: MoEConfig
    phw: int = 32,
    overlap: int = 24

class Autoencoder(Backbone[AutoencoderConfig]):
     def __init__(self, 
                  cfg: AutoencoderConfig):
        super().__init__(cfg)
        
        self.phw = cfg.phw
        self.overlap = cfg.overlap
        
        self.encoder = Encoder(
            cfg.EncoderConfig
        )

        self.decoder = MoE(
            cfg.DecoderConfig
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
        encoded = self.encoder(x)
        B, C, H, W = shape
        sf = self.encoder.scale_factor
        scaled_phw = self.phw * sf 
 
        decoded = self.decoder(scaled_phw, scaled_phw, encoded)
        y_hat = self.reconstruct(decoded, (B, C, H * sf, W * sf), scaled_phw, self.overlap*sf)
        return y_hat


# if __name__ == "__main__":
#     dummy_inputs = [torch.randn(1, 1, 128, 128), torch.randn(1, 1, 256, 256)]
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def extract_blocks(img_tensor, block_size, overlap):
#         blocks = []
#         step = block_size - overlap
#         for i in range(0, img_tensor.shape[1] - block_size + 1, step):
#             for j in range(0, img_tensor.shape[2] - block_size + 1, step):
#                 block = img_tensor[:, i:i+block_size, j:j+block_size]
#                 blocks.append(block)
#         return torch.stack(blocks)
#     def load_image(image_path):
#         image = Image.open(image_path)
#         transform = transforms.ToTensor()
#         return transform(image).to(device)  # Move tensor to GPU
    
#     image_path = '/home/ozkan/works/n-smoe/utils/test.png'
#     image_tensor = load_image(image_path)
    
#     blocks = extract_blocks(image_tensor, 32, 16)
#     image_tensor = image_tensor.unsqueeze(0)

#     encoder_cfg = EncoderConfig(
#         in_chans=3,
#         latent_dim=63,
#         embed_dim=16,
#         depth=4,
#         heads=1,
#         mlp_dim=16,
#         dropout=0.01,
#         patch_size=4,
#         avg_pool=False,
#         scale_factor=2,
#         resizer_num_layers=2,
#         backbone_cfg = BackboneResnetCfg(
#             name="resnet",
#             model="resnet50", 
#             num_layers=1, 
#             use_first_pool=True,
#             d_in=3,
#             d_out=63
#         )
#     )
#     decoder_cfg = MoEConfig(
#         in_chans=3,
#         num_mixtures=9,
#         kernel=9,
#         sharpening_factor=1.0
#     )


#     autoenocer_cfg = AutoencoderConfig(
#         EncoderConfig=encoder_cfg,
#         DecoderConfig=decoder_cfg,
#         phw=32,
#         overlap=16
#     )

#     model = Autoencoder(
#         cfg=autoenocer_cfg
#     )

#     params = sum(p.numel() for p in model.parameters())
#     print(f"Total number of parameters: {params}")

#     model = model.to(device)

#     output = model(blocks, image_tensor.shape)
#     print(f"Input shape: {blocks.shape} -> Output shape: {output.shape}")

    # model = Encoder(
    #     cfg=encoder_cfg
    # )
    # for dummy_input in dummy_inputs:
    #     output = model(dummy_input.cuda())
    #     print(f"Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
