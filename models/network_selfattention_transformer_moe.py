import functools
from dataclasses import dataclass
from functools import partial
from typing import Generic, Literal, Optional, TypedDict, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int64
from torch import Tensor
import cuda_block_ops

# import torchvision.transforms as transforms
# from PIL import Image

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

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, selfatt=True, kv_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.selfatt = selfatt  # Ensure selfatt is accessible throughout the class

        self.attend = nn.Softmax(dim=-1)
        if selfatt:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            if kv_dim is None:
                raise ValueError("kv_dim must be set when selfatt is False")
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, z=None):
        if self.selfatt:  # Use internal selfatt state
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        else:
            if z is None:
                raise ValueError("z must be provided when selfatt is False")
            q = self.to_q(x)
            k, v = self.to_kv(z).chunk(2, dim=-1)
            qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class PositionalEncoding(nn.Module):
    """For the sake of simplicity, this encodes values in the range [0, 1]."""

    frequencies: Float[Tensor, "frequency phase"]
    phases: Float[Tensor, "frequency phase"]

    def __init__(self, num_octaves: int):
        super().__init__()
        octaves = torch.arange(num_octaves).float()

        # The lowest frequency has a period of 1.
        frequencies = 2 * torch.pi * 2**octaves
        frequencies = repeat(frequencies, "f -> f p", p=2)
        self.register_buffer("frequencies", frequencies, persistent=False)

        # Choose the phases to match sine and cosine.
        phases = torch.tensor([0, 0.5 * torch.pi], dtype=torch.float32)
        phases = repeat(phases, "p -> f p", f=num_octaves)
        self.register_buffer("phases", phases, persistent=False)

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        samples = einsum(samples, self.frequencies, "... d, f p -> ... d f p")
        return rearrange(torch.sin(samples + self.phases), "... d f p -> ... (d f p)")

    def d_out(self, dimensionality: int):
        return self.frequencies.numel() * dimensionality

def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    Float[Tensor, "*shape dim"],  # float coordinates (xy indexing)
    Int64[Tensor, "*shape dim"],  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices

@dataclass
class ImageSelfAttentionCfg:
    patch_size: int
    num_octaves: int
    num_layers: int
    num_heads: int
    d_token: int
    d_dot: int
    d_mlp: int

class Transformer(nn.Module):
    def __init__(
            self, 
            dim, 
            depth, 
            heads, 
            dim_head, 
            mlp_dim, 
            dropout=0.0,
            selfatt=True,
            feed_forward_layer=FeedForward,
            kv_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                             Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                selfatt=selfatt,
                                kv_dim=kv_dim,
                            ),
                        ),
                        PreNorm(dim, feed_forward_layer(dim, mlp_dim, dropout=dropout), norm_type='ln'),
                    ]
                )
            )

    def forward(self, x, **kwargs):
        for attn, ff in self.layers:
            attn_out = attn(x)
            x = x + attn_out
            x = x + ff(x, **kwargs)
        return x

class ImageSelfAttention(nn.Module):
    positional_encoding: nn.Sequential
    patch_embedder: nn.Sequential
    transformer: Transformer

    def __init__(
        self,
        cfg: ImageSelfAttentionCfg,
        d_in: int,
        d_out: int,
    ):
        super().__init__()
        self.positional_encoding = nn.Sequential(
            (pe := PositionalEncoding(cfg.num_octaves)),
            nn.Linear(pe.d_out(2), cfg.d_token),
        )
        self.patch_embedder = nn.Sequential(
            nn.Conv2d(d_in, cfg.d_token, cfg.patch_size, cfg.patch_size),
            nn.ReLU(),
        )
        self.transformer = Transformer(
            cfg.d_token,
            cfg.num_layers,
            cfg.num_heads,
            cfg.d_dot,
            cfg.d_mlp,
        )
        self.resampler = nn.ConvTranspose2d(
            cfg.d_token,
            d_out,
            cfg.patch_size,
            cfg.patch_size,
        )

    def forward(
        self,
        image: Float[Tensor, "batch d_in height width"],
    ) -> Float[Tensor, "batch d_out height width"]:
        # Embed patches so they become tokens.
        tokens = self.patch_embedder.forward(image)

        # Append positional information to the tokens.
        _, _, nh, nw = tokens.shape
        xy, _ = sample_image_grid((nh, nw), device=image.device)
        xy = self.positional_encoding.forward(xy)
        tokens = tokens + rearrange(xy, "nh nw c -> c nh nw")

        # Put the tokens through a transformer.
        _, _, nh, nw = tokens.shape
        tokens = rearrange(tokens, "b c nh nw -> b (nh nw) c")
        tokens = self.transformer.forward(tokens)

        # Resample the tokens back to the original resolution.
        tokens = rearrange(tokens, "b (nh nw) c -> b c nh nw", nh=nh, nw=nw)
        tokens = self.resampler.forward(tokens)

        return tokens

class ImageSelfAttentionWrapper(nn.Module):
    def __init__(
        self,
        self_attention_cfg: ImageSelfAttentionCfg,
        d_in: int,
        d_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = ImageSelfAttention(self_attention_cfg, d_in, d_in)

    def forward(
        self,
        x: Float[Tensor, "batch token dim"],
        b: int,
        h: int,
        w: int,
    ) -> Float[Tensor, "batch token dim"]:
        x = rearrange(x, "(b h w) () c -> b c h w", b=b, h=h, w=w)
        x = self.self_attention(x) + x
        return rearrange(x, "b c h w -> (b h w) () c", b=b, h=h, w=w)

@dataclass
class SelfAttentionTransformerCfg:
    self_attention: ImageSelfAttentionCfg
    num_layers: int
    num_heads: int
    d_dot: int
    d_mlp: int
    downscale: int
    # kv_dim: int  # Add kv_dim to the configuration

class SelfAttentionTransformer(nn.Module):
    cfg: SelfAttentionTransformerCfg
    transformer: Transformer
    downscaler: Optional[nn.Conv2d]
    upscaler: Optional[nn.ConvTranspose2d]
    upscale_refinement: Optional[nn.Sequential]

    def __init__(
        self,
        cfg: SelfAttentionTransformerCfg,
        d_in: int,
    ) -> None:
        super().__init__()
        
        self.cfg = cfg
       
        feed_forward_layer = partial(ImageSelfAttentionWrapper, cfg.self_attention)
        self.transformer = Transformer(
            d_in,
            cfg.num_layers,
            cfg.num_heads,
            cfg.d_dot,
            cfg.d_mlp,
            selfatt=True,
            kv_dim=None,
            feed_forward_layer=feed_forward_layer,
        )

        if cfg.downscale:
            self.downscaler = nn.Conv2d(d_in, d_in, cfg.downscale, cfg.downscale)
            self.upscaler = nn.ConvTranspose2d(d_in, d_in, cfg.downscale, cfg.downscale)
            self.upscale_refinement = nn.Sequential(
                nn.Conv2d(d_in, d_in * 2, 7, 1, 3),
                nn.GELU(),
                nn.Conv2d(d_in * 2, d_in, 7, 1, 3),
            )

    def forward(
        self,
        features: Float[Tensor, "batch channel height width"],
    ) -> tuple[Float[Tensor, "batch channel height width"]]:
        b, _, h, w = features.shape

        # If needed, apply downscaling.
        if self.downscaler is not None:
            features = rearrange(features, "b c h w -> b c h w")
            features = self.downscaler(features)
            features = rearrange(features, "b c h w -> b c h w", b=b)

    
        # Run the transformer.
        q = rearrange(features, "b c h w -> (b h w) () c")
        features = self.transformer.forward(
            q,
            b=b,
            h=h // self.cfg.downscale,
            w=w // self.cfg.downscale,
        )
        features = rearrange(
            features,
            "(b h w) () c -> b c h w",
            b=b,
            h=h // self.cfg.downscale,
            w=w // self.cfg.downscale,
        )

        # If needed, apply upscaling.
        if self.upscaler is not None:
            features = rearrange(features, "b  c h w -> (b ) c h w")
            features = self.upscaler(features)
            features = self.upscale_refinement(features) + features
            features = rearrange(features, "(b ) c h w -> b c h w", b=b)

        return features


@dataclass
class BackboneResnetCfg:
    name: Literal["resnet"]
    model: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "dino_resnet50"]
    num_layers: int
    use_first_pool: bool

class BackboneResnet(Backbone[BackboneResnetCfg]):
    def __init__(self, cfg: BackboneResnetCfg,d_in:int,d_out:int):
        super().__init__(cfg)
        
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.model = getattr(torchvision.models, cfg.model)(pretrained=False, norm_layer=norm_layer)

        self.model.conv1 = nn.Conv2d(d_in, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size, stride=self.model.conv1.stride, padding=self.model.conv1.padding, bias=False)

        self.projections = nn.ModuleDict()
        previous_output_channels = self.model.conv1.out_channels  
        self.projections['layer0'] = nn.Conv2d(previous_output_channels, d_out, 1)

        layers = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        for i, layer_group in enumerate(layers[:cfg.num_layers - 1]):
            output_channels = layer_group[-1].conv3.out_channels if hasattr(layer_group[-1], 'conv3') else layer_group[-1][-1].out_channels
            self.projections[f'layer{i+1}'] = nn.Conv2d(output_channels, d_out, 1)

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
class BackboneDinoCfg:
    name: Literal["dino"]
    model: Literal["dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8"]
    backbone_cfg: BackboneResnetCfg 

class BackboneDino(Backbone[BackboneDinoCfg]):
    def __init__(self, cfg: BackboneDinoCfg, d_in: int, d_out: int) -> None:
        super().__init__(cfg)
        self.dino = torch.hub.load("facebookresearch/dino:main", cfg.model)
        self._configure_dino_patch_embedding(d_in)
        self.resnet_backbone = BackboneResnet(cfg.backbone_cfg, d_in, d_out)
        dino_dim = self.get_dino_feature_dim()
        self.global_token_mlp = self._create_mlp(dino_dim, d_out)
        self.local_token_mlp = self._create_mlp(dino_dim, d_out)

    def get_dino_feature_dim(self):
        feature_dims = {
            "dino_vits16": 384,
            "dino_vits8": 384,
            "dino_vitb16": 768,
            "dino_vitb8": 768
        }
        model_key = self.cfg.model.split(':')[-1]
        return feature_dims.get(model_key, 768)

    def _configure_dino_patch_embedding(self, d_in: int):
        old_conv = self.dino.patch_embed.proj
        params = {
            'out_channels': old_conv.out_channels,
            'kernel_size': self._ensure_tuple(old_conv.kernel_size),
            'stride': self._ensure_tuple(old_conv.stride),
            'padding': self._ensure_tuple(old_conv.padding),
            'bias': old_conv.bias is not None
        }
        self.dino.patch_embed.proj = nn.Conv2d(d_in, **params)

    def _ensure_tuple(self, value):
        return value if isinstance(value, tuple) else tuple(value.tolist())

    def _create_mlp(self, input_dim: int, output_dim: int):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, context: BatchedViews) -> torch.Tensor:
        resnet_features = self.resnet_backbone(context)
        b, _, h, w = context["image"].shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0

        tokens = self.dino.get_intermediate_layers(context["image"])[0]
        global_token = self.global_token_mlp(tokens[:, 0])
        local_tokens = self.local_token_mlp(tokens[:, 1:])

        global_token = repeat(global_token, "b c -> b c h w", b=b, h=h, w=w)
        local_tokens = repeat(local_tokens, "b (h w) c -> b c (h hps) (w wps)", b=b, h=h // self.patch_size, hps=self.patch_size, w=w // self.patch_size, wps=self.patch_size)

        return resnet_features + local_tokens + global_token

    @property
    def patch_size(self) -> int:
        return int("".join(filter(str.isdigit, self.cfg.model)))

@dataclass
class EncoderConfig:
    embed_dim: int
    dropout: float
    avg_pool: bool
    scale_factor: int
    resizer_num_layers: int
    patch_size: int
    backbone_cfg: BackboneDinoCfg 
    transformer_cfg : SelfAttentionTransformerCfg


class Encoder(Backbone[EncoderConfig]):
    def __init__(self, cfg: EncoderConfig, d_in:int, d_out:int):
        super().__init__(cfg)
        self.d_in = d_in
        self.latent = d_out
     
        self.embed_dim = cfg.embed_dim
        self.scale_factor = cfg.scale_factor

        if cfg.backbone_cfg is not None:
              self.backbone = BackboneDino(cfg.backbone_cfg, d_in=d_in,d_out=d_out)
        else:
            self.backbone = None

        self.transformer = SelfAttentionTransformer(cfg.transformer_cfg, d_in=d_out)
        self.patch_embed = PatchEmbed(patch_size=cfg.patch_size, in_chans=d_out, embed_dim=cfg.embed_dim)

        self.fc = nn.Sequential(
                nn.Linear(cfg.embed_dim, d_in * d_out),
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
        x = self.transformer(x)
        x = self.patch_embed(x)
        x = self.fc(x)
        x = rearrange(x, "b n c -> b c n")
        x = torch.mean(x, dim=2, keepdim=True)
        x = rearrange(x, 'b (c latent) 1 -> b c latent', c=self.d_in, latent=self.latent)
        return x
    

@dataclass
class MoEConfig:
    num_mixtures: int = 4
    kernel: int = 4
    sharpening_factor: float = 1.0

class MoE(Backbone[MoEConfig]):
    def __init__(
        self,
        cfg: MoEConfig,
        d_in: int,
    ):
        super(MoE, self).__init__(cfg)

        self.ch = d_in
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
    d_in: int
    d_out: int
    phw: int = 32,
    overlap: int = 24

# class Autoencoder(Backbone[AutoencoderConfig]):
#      def __init__(self, 
#                   cfg: AutoencoderConfig):
#         super().__init__(cfg)
        
#         self.phw = cfg.phw
#         self.overlap = cfg.overlap
        
#         self.encoder = Encoder(
#             cfg.EncoderConfig,
#             d_in=cfg.d_in,
#             d_out=cfg.d_out
#         )

#         self.decoder = MoE(
#             cfg.DecoderConfig,
#             d_in=cfg.d_in
#         )

#      @staticmethod
#      def reconstruct(blocks, original_dims, block_size, overlap):
#         batch_size, num_channels, height, width = original_dims
#         step = block_size - overlap
#         device = blocks.device

#         recon_images = torch.zeros(batch_size, num_channels, height, width).to(device)
#         count_matrix = torch.zeros(batch_size, num_channels, height, width).to(device)

#         num_blocks_per_row = (width - block_size) // step + 1
#         num_blocks_per_column = (height - block_size) // step + 1
#         num_blocks_per_image = num_blocks_per_row * num_blocks_per_column

#         for b in range(batch_size):
#             idx_start = b * num_blocks_per_image
#             current_blocks = blocks[idx_start:idx_start + num_blocks_per_image]
#             idx = 0
#             for i in range(0, height - block_size + 1, step):
#                 for j in range(0, width - block_size + 1, step):
#                     recon_images[b, :, i:i+block_size, j:j+block_size] += current_blocks[idx]
#                     count_matrix[b, :, i:i+block_size, j:j+block_size] += 1
#                     idx += 1

#         recon_images /= count_matrix.clamp(min=1)
#         return recon_images
     
#      def forward(self, x, shape):
#         if len(x.shape) == 5:
#             x = x.view(-1, *x.size()[2:])
#         encoded = self.encoder(x)
#         B, C, H, W = shape
#         sf = self.encoder.scale_factor
#         scaled_phw = self.phw * sf 
 
#         decoded = self.decoder(scaled_phw, scaled_phw, encoded)
#         y_hat = self.reconstruct(decoded, (B, C, H * sf, W * sf), scaled_phw, self.overlap*sf)
#         return y_hat


class Autoencoder(Backbone[AutoencoderConfig]):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__(cfg)
        self.phw = cfg.phw
        self.overlap = cfg.overlap
        self.encoder = Encoder(cfg.EncoderConfig, d_in=cfg.d_in, d_out=cfg.d_out)
        self.decoder = MoE(cfg.DecoderConfig, d_in=cfg.d_in)

    @staticmethod
    def reconstruct(blocks, original_dims, block_size, overlap):

        batch_size, num_channels, height, width = original_dims
        step = block_size - overlap
        device = blocks.device

        recon_images = torch.zeros(batch_size, num_channels, height, width, device=device)

        cuda_block_ops.reconstruct(
            blocks.contiguous(),  
            recon_images,        
            batch_size,           
            num_channels,        
            height,
            width,
            block_size,
            step
        )

        return recon_images
    
    def mem_lim(self):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        if dev == 'cuda':
            torch.cuda.set_device(0)
            tot_mem = torch.cuda.get_device_properties(0).total_memory
            used_mem = torch.cuda.memory_allocated(0)
            free_mem = tot_mem - used_mem

            thresholds = [0.5, 0.3, 0.1]
            for percent in thresholds:
                threshold = tot_mem * percent
                if free_mem > threshold:
                    return threshold

            min_threshold = max(1 * 2**30, tot_mem * 0.05)
            return min_threshold
        else:
            return 1 * 2**30

    def forward(self, x, s):
        if x.ndim == 5:
            x = x.reshape(-1, *x.shape[2:])

        es = x.element_size()
        ml = self.mem_lim()
        bm = x.shape[1:].numel() * es
        mx_bs = ml // bm
        n = max(1, min(x.shape[0] // 1024, mx_bs))
        cs = (x.shape[0] + n - 1) // n

        b = torch.split(x, cs)

        enc = torch.cat([self.encoder(bt) for bt in b], dim=0)

        B, C, H, W = s
        sp = self.phw * self.encoder.scale_factor

        dec = torch.cat([self.decoder(sp, sp, bt) for bt in torch.split(enc, cs)], dim=0)

        y = self.reconstruct(dec, (B, C, H * self.encoder.scale_factor, W * self.encoder.scale_factor), sp, self.overlap * self.encoder.scale_factor)
        return y

if __name__ == "__main__":
    # dino_cfg = BackboneDinoCfg(name="dino", model="dino_vitb8", d_out=512,d_in=3, backbone_cfg=BackboneResnetCfg(name="resnet", model="resnet50", num_layers=1, use_first_pool=True))
    # backbone_dino = BackboneDino(dino_cfg)
    # sample_input = torch.rand(1, 3, 128, 128)  # Simulate an input image tensor
    # batched_views = BatchedViews(image=sample_input)

    # # # Run the test
    # output = backbone_dino(batched_views)
    # print("Output shape:", output.shape)

    # dummy_inputs = [torch.randn(1, 1, 128, 128), torch.randn(1, 1, 256, 256)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_blocks(img_tensor, block_size, overlap):
        blocks = []
        step = block_size - overlap
        for i in range(0, img_tensor.shape[1] - block_size + 1, step):
            for j in range(0, img_tensor.shape[2] - block_size + 1, step):
                block = img_tensor[:, i:i+block_size, j:j+block_size]
                blocks.append(block)
        return torch.stack(blocks)
    
    # def load_image(image_path):
    #     image = Image.open(image_path)
    #     transform = transforms.ToTensor()
    #     return transform(image).to(device)  # Move tensor to GPU
    
    # image_path = '/home/ozkan/works/n-smoe/utils/test.png'
    # image_tensor = load_image(image_path)
    
    image_tensor = torch.randn(1, 512, 512).cuda()

    blocks = extract_blocks(image_tensor, 32, 16)
    image_tensor = image_tensor.unsqueeze(0)

    # epipolar_transformer:
    #     self_attention:
    #         patch_size: 4
    #         num_octaves: 10
    #         num_layers: 2
    #         num_heads: 4
    #         d_token: 128
    #         d_dot: 128
    #         d_mlp: 256
    #     num_layers: 2
    #     num_heads: 4
    #     num_samples: 32
    #     d_dot: 128
    #     d_mlp: 256
    #     downscale: 4

    encoder_cfg = EncoderConfig(
        embed_dim=128,
        dropout=0.01,
        patch_size=4,
        scale_factor=2,
        avg_pool=False,
        resizer_num_layers=2,
        backbone_cfg = BackboneDinoCfg(
                name="dino", 
                model="dino_vitb8", 
                backbone_cfg=BackboneResnetCfg(name="resnet", model="resnet50", 
                                               num_layers=1, use_first_pool=True)),

        transformer_cfg = SelfAttentionTransformerCfg(
                self_attention=ImageSelfAttentionCfg(
                    patch_size=4,
                    num_octaves=10,
                    num_layers=2,
                    num_heads=2,
                    d_token=128,
                    d_dot=128,
                    d_mlp=128
                ),
                num_layers=2,
                num_heads=2,
                d_dot=128,
                d_mlp=128,
                downscale=4))
    
    decoder_cfg = MoEConfig(
        num_mixtures=9,
        kernel=9,
        sharpening_factor=1.0
    )

    autoenocer_cfg = AutoencoderConfig(
        EncoderConfig=encoder_cfg,
        DecoderConfig=decoder_cfg,
        d_in=1,
        d_out=63,
        phw=32,
        overlap=16
    )

    model = Autoencoder(
        cfg=autoenocer_cfg
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {params}")

    model = model.to(device)

    output = model(blocks, image_tensor.shape)
    print(f"Input shape: {blocks.shape} -> Output shape: {output.shape}")

    # model = Encoder(
    #     cfg=encoder_cfg
    # )
    # for dummy_input in dummy_inputs:
    #     output = model(dummy_input.cuda())
    #     print(f"Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
