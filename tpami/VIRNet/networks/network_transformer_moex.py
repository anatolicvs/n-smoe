import functools
import math
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Tuple, Literal, Optional, TypedDict, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from torch.nn.attention import SDPBackend

from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

from .DnCNN import DnCNN
from .KNet import KernelNet
from .nn import normalization


backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

OLD_GPU = True
USE_FLASH_ATTN = False
MATH_KERNEL_ON = True


class BatchedViews(TypedDict):
    image: torch.Tensor


T = TypeVar("T")
log_min = math.log(1e-10)
log_max = math.log(1e2)


class Backbone(nn.Module, Generic[T]):
    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def d_out(self) -> int:
        raise NotImplementedError


class MullerResizer(nn.Module):
    def __init__(
        self,
        d_in=3,
        base_resize_method="bicubic",
        kernel_size=5,
        stddev=1.0,
        num_layers=2,
        avg_pool=False,
        init_weights=None,
        dtype=torch.float32,
    ):
        super(MullerResizer, self).__init__()
        self.d_in = d_in
        self.kernel_size = kernel_size
        self.stddev = stddev
        self.num_layers = num_layers
        self.avg_pool = avg_pool
        self.dtype = dtype
        interpolation_methods = {
            "bilinear": "bilinear",
            "nearest": "nearest",
            "bicubic": "bicubic",
        }
        self.interpolation_method = interpolation_methods.get(
            base_resize_method, "bilinear"
        )
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        if init_weights is not None:
            for i in range(num_layers):
                self.weights.append(
                    nn.Parameter(torch.tensor(init_weights[2 * i], dtype=dtype))
                )
                self.biases.append(
                    nn.Parameter(torch.tensor(init_weights[2 * i + 1], dtype=dtype))
                )
        else:
            for _ in range(num_layers):
                weight = nn.Parameter(torch.empty(1, dtype=dtype))
                bias = nn.Parameter(torch.empty(1, dtype=dtype))
                nn.init.uniform_(weight, a=-0.1, b=0.1)
                nn.init.zeros_(bias)
                self.weights.append(weight)
                self.biases.append(bias)
        # self.gaussian_kernel = self.create_gaussian_kernel(kernel_size, stddev)
        self.register_buffer(
            "gaussian_kernel", self.create_gaussian_kernel(kernel_size, stddev)
        )

    def create_gaussian_kernel(self, kernel_size, stddev):
        t = torch.arange(kernel_size, dtype=self.dtype) - (kernel_size - 1) / 2
        gaussian_kernel = torch.exp(-t.pow(2) / (2 * (stddev**2)))
        gaussian_kernel /= gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(
            1, 1, kernel_size, 1
        ) * gaussian_kernel.view(1, 1, 1, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.d_in, 1, 1, 1)
        return gaussian_kernel

    def _apply_gaussian_blur(self, input):
        padding = self.kernel_size // 2
        x = F.pad(input, (padding, padding, padding, padding), mode="reflect")
        gaussian_kernel = self.gaussian_kernel.to(x.device)
        return F.conv2d(x, gaussian_kernel, groups=self.d_in)

    def forward(self, input_tensor, target_size):
        x = input_tensor.to(dtype=self.dtype)
        if self.avg_pool:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        net = F.interpolate(
            x, size=target_size, mode=self.interpolation_method, align_corners=False
        )
        for weight, bias in zip(self.weights, self.biases):
            blurred = self._apply_gaussian_blur(x)
            residual = blurred - x
            resized_residual = F.interpolate(
                residual,
                size=target_size,
                mode=self.interpolation_method,
                align_corners=False,
            )
            net = net + torch.tanh(weight * resized_residual + bias)
            x = blurred
        return net


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
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
    def __init__(self, dim, fn, norm_type="ln"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class Attention(nn.Module):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, selfatt=True, kv_dim=None
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
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
        kv_dim=None,
    ):
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
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout),
                            norm_type="ln",
                        ),
                    ]
                )
            )

    def forward(self, x, **kwargs):
        for attn, ff in self.layers:
            attn_out = attn(x)
            x = x + attn_out
            x = x + ff(x, **kwargs)
        return x


@dataclass
class BackboneResnetCfg:
    name: Literal["resnet"]
    model: Literal[
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "dino_resnet50"
    ]
    num_layers: int
    use_first_pool: bool
    pretrained: bool = False


class BackboneResnet(Backbone[BackboneResnetCfg]):
    def __init__(self, cfg: BackboneResnetCfg, d_in: int, d_out: int):
        super().__init__(cfg)

        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )

        model_weights = {
            "resnet18": ResNet18_Weights.DEFAULT,
            "resnet34": ResNet34_Weights.DEFAULT,
            "resnet50": ResNet50_Weights.DEFAULT,
            "resnet101": ResNet101_Weights.DEFAULT,
            "resnet152": ResNet152_Weights.DEFAULT,
            "dino_resnet50": ResNet50_Weights.DEFAULT,
        }

        if cfg.model not in model_weights:
            raise ValueError(f"Unsupported model name: {cfg.model}")

        weights = model_weights[cfg.model] if cfg.pretrained else None

        self.model = getattr(torchvision.models, cfg.model)(
            weights=weights, norm_layer=norm_layer
        )

        self.model.conv1 = nn.Conv2d(
            d_in,
            self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False,
        )

        self.projections = nn.ModuleDict()
        previous_output_channels = self.model.conv1.out_channels
        self.projections["layer0"] = nn.Conv2d(previous_output_channels, d_out, 1)

        layers = [
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        ]

        if cfg.num_layers < 1 or cfg.num_layers > len(layers) + 1:
            raise ValueError(
                f"cfg.num_layers must be between 1 and {len(layers) + 1}, got {cfg.num_layers}"
            )

        for i, layer_group in enumerate(layers[: cfg.num_layers - 1]):
            last_layer = layer_group[-1]
            if hasattr(last_layer, "conv3"):
                output_channels = last_layer.conv3.out_channels
            elif hasattr(last_layer, "conv2"):
                output_channels = last_layer.conv2.out_channels
            else:
                raise AttributeError(
                    f"Last layer of layer_group {i+1} does not have conv2 or conv3"
                )
            self.projections[f"layer{i+1}"] = nn.Conv2d(output_channels, d_out, 1)

    def forward(self, context: BatchedViews) -> torch.Tensor:
        x = context["image"]

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        features = [self.projections["layer0"](x)]

        layers = [
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        ]
        for index in range(1, self.cfg.num_layers):
            if index - 1 >= len(layers):
                raise IndexError(
                    f"Requested layer index {index} exceeds available layers {len(layers)}"
                )
            x = layers[index - 1](x)
            layer_key = f"layer{index}"
            if layer_key not in self.projections:
                raise KeyError(f"Projection layer '{layer_key}' is missing")
            features.append(self.projections[layer_key](x))

        h, w = context["image"].shape[2:]
        features = [
            F.interpolate(feature, (h, w), mode="bilinear", align_corners=True)
            for feature in features
        ]

        output = torch.stack(features).sum(dim=0)

        return output


@dataclass
class BackboneDinoCfg:
    name: Literal["dino"]
    model: Literal["dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8"]
    backbone_cfg: BackboneResnetCfg


class BackboneDino(Backbone[BackboneDinoCfg]):
    def __init__(self, cfg: BackboneDinoCfg, d_in: int, d_out: int) -> None:
        super().__init__(cfg)
        self.dino = torch.hub.load(
            "facebookresearch/dino:main", cfg.model, skip_validation=True
        )
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
            "dino_vitb8": 768,
        }
        model_key = self.cfg.model.split(":")[-1]
        return feature_dims.get(model_key, 768)

    def _configure_dino_patch_embedding(self, d_in: int):
        old_conv = self.dino.patch_embed.proj
        params = {
            "out_channels": old_conv.out_channels,
            "kernel_size": self._ensure_tuple(old_conv.kernel_size),
            "stride": self._ensure_tuple(old_conv.stride),
            "padding": self._ensure_tuple(old_conv.padding),
            "bias": old_conv.bias is not None,
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
        local_tokens = repeat(
            local_tokens,
            "b (h w) c -> b c (h hps) (w wps)",
            b=b,
            h=h // self.patch_size,
            hps=self.patch_size,
            w=w // self.patch_size,
            wps=self.patch_size,
        )

        return resnet_features + local_tokens + global_token

    @property
    def patch_size(self) -> int:
        return int("".join(filter(str.isdigit, self.cfg.model)))


@dataclass
class EncoderConfig:
    embed_dim: int
    depth: int
    heads: int
    dim_head: int
    mlp_dim: int
    dropout: float

    resizer_avg_pool: bool
    scale_factor: int
    resizer_num_layers: int
    patch_size: int
    backbone_cfg: BackboneDinoCfg

    noise_cond: bool
    kernel_cond: bool
    noise_avg: bool
    sigma_chn: int
    kernel_chn: int

    activation: str = "GELU"
    num_groups: Optional[int] = None


class AttentionPool2d(nn.Module):
    def __init__(self, embed_dim: int, group_dim: int, num_heads: int, output_dim: int):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim + 1, group_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(group_dim, group_dim)
        self.q_proj = nn.Linear(group_dim, group_dim)
        self.v_proj = nn.Linear(group_dim, group_dim)
        self.c_proj = nn.Linear(group_dim, output_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class Encoder(Backbone[EncoderConfig]):
    def __init__(self, cfg: EncoderConfig, d_in: int, d_out: int, phw: int):
        super().__init__(cfg)
        self.d_in = d_in
        self.latent = d_out

        self.embed_dim = cfg.embed_dim
        self.noise_cond = cfg.noise_cond
        self.noise_avg = cfg.noise_avg
        self.kernel_cond = cfg.kernel_cond
        self.sigma_chn = cfg.sigma_chn
        self.kernel_chn = cfg.kernel_chn

        extra_chn = 0
        if self.kernel_cond:
            extra_chn += self.kernel_chn
        if self.noise_cond:
            extra_chn += self.sigma_chn

        if hasattr(nn, cfg.activation):
            activation = getattr(nn, cfg.activation)()

        if cfg.backbone_cfg is not None:
            self.backbone = BackboneDino(
                cfg.backbone_cfg, d_in=d_in + extra_chn, d_out=d_out
            )
        else:
            self.backbone = None

        self.patch_embed = PatchEmbed(
            patch_size=cfg.patch_size, in_chans=d_out, embed_dim=cfg.embed_dim
        )
        self.transformer = Transformer(
            cfg.embed_dim, cfg.depth, cfg.heads, cfg.dim_head, cfg.mlp_dim, cfg.dropout
        )

        self.out = nn.Sequential(
            # normalization(cfg.num_groups, int(((phw * cfg.scale_factor)//cfg.patch_size))**2),
            normalization(
                channels=int(((phw * cfg.scale_factor) // cfg.patch_size)) ** 2
            ),
            # nn.LayerNorm(
            #     normalized_shape=int(((phw * cfg.scale_factor) // cfg.patch_size)) ** 2
            # ),
            activation,
            AttentionPool2d(
                cfg.embed_dim,
                # int((phw * cfg.scale_factor)),
                int(((phw * cfg.scale_factor) // cfg.patch_size)) ** 2,
                cfg.heads,
                int(self.d_in * self.latent),
            ),
        )

        self.resizer = MullerResizer(
            self.d_in,
            "bicubic",
            kernel_size=5,
            stddev=1.0,
            num_layers=cfg.resizer_num_layers,
            avg_pool=cfg.resizer_avg_pool,
        )

    def _interpolate(self, x, scale_factor):
        _, _, H, W = x.size()
        target_h = int(H * scale_factor)
        target_w = int(W * scale_factor)
        target_size = (target_h, target_w)
        x_resized = self.resizer(x, target_size)
        return x_resized

    def forward(
        self,
        x: torch.Tensor,
        sigma_est: Optional[torch.Tensor] = None,
        kinfo_est: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # sigma = torch.exp(torch.clamp(sigma_est, min=log_min, max=log_max))
        sigma = F.softplus(sigma_est) + 1e-8

        x_up = self._interpolate(x, self.scale_factor)
        h_up, w_up = x_up.shape[-2], x_up.shape[-1]

        if not self.noise_cond and not self.kernel_cond:
            extra_maps = None
        else:
            tmp_list = []
            if self.kernel_cond:
                tmp_list.append(kinfo_est.repeat(1, 1, h_up, w_up))
            if self.noise_cond:
                s_sqrt = sigma.sqrt()
                if self.noise_avg:
                    tmp_list.append(s_sqrt.repeat(1, 1, h_up, w_up))
                else:
                    tmp_list.append(
                        F.interpolate(
                            s_sqrt, scale_factor=self.scale_factor, mode="nearest"
                        )
                    )
            extra_maps = torch.cat(tmp_list, dim=1) if len(tmp_list) > 0 else None

        if extra_maps is not None:
            x_input = torch.cat([x_up, extra_maps], dim=1)
        else:
            x_input = x_up

        features = self.backbone({"image": x_input})
        features = self.patch_embed(features)
        features = self.transformer(features)

        gaussians = self.out(features)
        gaussians = rearrange(
            gaussians, "b (c latent) -> b c latent", c=self.d_in, latent=self.latent
        )

        return gaussians, kinfo_est.squeeze(-1).squeeze(-1), sigma

    @property
    def scale_factor(self) -> int:
        return self.cfg.scale_factor


class KernelType(Enum):
    GAUSSIAN = "gaussian"
    GAUSSIAN_CAUCHY = "gaussian_cauchy"


@dataclass
class MoEConfig:
    kernel: int = 4
    sharpening_factor: float = 1.0
    kernel_type: Optional[KernelType] = KernelType.GAUSSIAN_CAUCHY


@dataclass
class Gaussians:
    mu: Optional[torch.Tensor] = None
    sigma: Optional[torch.Tensor] = None
    w: Optional[torch.Tensor] = None
    theta: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None


@dataclass
class AutoencoderConfig:
    EncoderConfig: EncoderConfig
    DecoderConfig: MoEConfig
    d_in: int
    dep_S: int
    dep_K: int
    d_out: Optional[int] = None
    phw: int = 32
    overlap: int = 24


class MoE(Backbone[MoEConfig]):
    def __init__(self, cfg: MoEConfig):
        super().__init__(cfg)
        self.kernel = cfg.kernel
        self.sharpening_factor = cfg.sharpening_factor
        self.kernel_type = cfg.kernel_type
        self.min_diag = 1e-6
        self.min_denom = 1e-8

    def grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        xx = torch.linspace(0.0, 1.0, width, device=device)  # [W]
        yy = torch.linspace(0.0, 1.0, height, device=device)  # [H]
        gx, gy = torch.meshgrid(xx, yy, indexing="ij")  # Each [W, H]
        grid = torch.stack((gx, gy), dim=-1)  # [W, H, 2]
        return grid.float()  # [W, H, 2]

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        ct = torch.cos(theta).unsqueeze(-1)  # [B, ch, k, 1]
        st = torch.sin(theta).unsqueeze(-1)  # [B, ch, k, 1]
        R = torch.cat([ct, -st, st, ct], dim=-1)  # [B, ch, k, 4]
        return R.view(*theta.shape, 2, 2)  # [B, ch, k, 2, 2]

    def construct_lower_triangular_size1(self, params: torch.Tensor) -> torch.Tensor:
        B, ch, k, _ = params.shape  # [B, ch, k, 1]
        L = torch.zeros(
            B, ch, k, 1, 1, device=params.device, dtype=params.dtype
        )  # [B, ch, k, 1, 1]
        L[..., 0, 0] = F.softplus(params[..., 0]) + 1e-2  # [B, ch, k]
        return L  # [B, ch, k, 1, 1]

    def construct_lower_triangular_size2(self, params: torch.Tensor) -> torch.Tensor:
        B, ch, k, _ = params.shape  # [B, ch, k, 3]
        l11, l21, l22 = torch.split(params, 1, dim=-1)  # Each [B, ch, k, 1]
        l11 = F.softplus(l11) + 1e-2  # [B, ch, k, 1]
        l22 = F.softplus(l22) + 1e-2  # [B, ch, k, 1]
        L = torch.zeros(
            B, ch, k, 2, 2, device=params.device, dtype=params.dtype
        )  # [B, ch, k, 2, 2]
        L[..., 0, 0] = l11.squeeze(-1)  # [B, ch, k]
        L[..., 1, 0] = l21.squeeze(-1)  # [B, ch, k]
        L[..., 1, 1] = l22.squeeze(-1)  # [B, ch, k]
        return L  # [B, ch, k, 2, 2]

    def construct_lower_triangular_size3(self, params: torch.Tensor) -> torch.Tensor:
        B, ch, k, _ = params.shape  # [B, ch, k, 6]
        l11, l21, l22, l31, l32, l33 = torch.split(
            params, 1, dim=-1
        )  # Each [B, ch, k, 1]
        l11 = F.softplus(l11) + 1e-2  # [B, ch, k, 1]
        l22 = F.softplus(l22) + 1e-2  # [B, ch, k, 1]
        l33 = F.softplus(l33) + 1e-2  # [B, ch, k, 1]
        L = torch.zeros(
            B, ch, k, 3, 3, device=params.device, dtype=params.dtype
        )  # [B, ch, k, 3, 3]
        L[..., 0, 0] = l11.squeeze(-1)  # [B, ch, k]
        L[..., 1, 0] = l21.squeeze(-1)  # [B, ch, k]
        L[..., 1, 1] = l22.squeeze(-1)  # [B, ch, k]
        L[..., 2, 0] = l31.squeeze(-1)  # [B, ch, k]
        L[..., 2, 1] = l32.squeeze(-1)  # [B, ch, k]
        L[..., 2, 2] = l33.squeeze(-1)  # [B, ch, k]
        return L  # [B, ch, k, 3, 3]

    def construct_lower_triangular(self, p: torch.Tensor, s: int) -> torch.Tensor:
        if s == 1:
            return self.construct_lower_triangular_size1(p)
        if s == 2:
            return self.construct_lower_triangular_size2(p)
        if s == 3:
            return self.construct_lower_triangular_size3(p)
        raise ValueError(f"Unsupported size: {s}")

    def cov_mat(
        self,
        L_spatial: torch.Tensor,
        theta_xy: torch.Tensor,
        L_color: torch.Tensor,
        ch: int,
    ) -> torch.Tensor:
        R = self.ang_to_rot_mat(theta_xy)  # [B, ch, k, 2, 2]
        C_xy = torch.matmul(
            R, torch.matmul(L_spatial, L_spatial.transpose(-2, -1))
        )  # [B, ch, k, 2, 2]
        C_xy = torch.matmul(C_xy, R.transpose(-2, -1))  # [B, ch, k, 2, 2]
        C_xy = 0.5 * (C_xy + C_xy.mT)  # [B, ch, k, 2, 2] Ensure symmetry
        if ch == 1:
            C_color = torch.matmul(L_color, L_color.transpose(-2, -1))  # [B,1,k,1,1]
            C_color = C_color.squeeze(-1).squeeze(-1)  # [B,1,k]
            B_, ch_, k_ = C_xy.shape[:3]
            C_full = torch.zeros(
                B_, ch_, k_, 3, 3, device=C_xy.device, dtype=C_xy.dtype
            )  # [B,1,k,3,3]
            C_full[..., :2, :2] = C_xy  # [B,1,k,2,2]
            C_full[..., 2, 2] = C_color  # [B,1,k]
        elif ch == 3:
            C_color = torch.matmul(L_color, L_color.transpose(-2, -1))  # [B,3,k,3,3]
            B_, ch_, k_ = C_xy.shape[:3]
            C_full = torch.zeros(
                B_, ch_, k_, 5, 5, device=C_xy.device, dtype=C_xy.dtype
            )  # [B,3,k,5,5]
            C_full[..., :2, :2] = C_xy  # [B,3,k,2,2]
            C_full[..., 2:, 2:] = C_color  # [B,3,k,3,3]
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")

        return C_full * self.sharpening_factor  # [B, ch, k, D, D]

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        B, _, _ = p.shape  # [B, ch, k * param_per_kernel]
        p = p.view(B, ch, k, -1)  # [B, ch, k, param_per_kernel]
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            mu_x = p[..., 0].reshape(B, ch, k, 1)  # [B, ch, k, 1]
            mu_y = p[..., 1].reshape(B, ch, k, 1)  # [B, ch, k, 1]
            L_spatial_params = p[..., 2:5].reshape(B, ch, k, 3)  # [B, ch, k, 3]
            L_spatial = self.construct_lower_triangular(
                L_spatial_params, s=2
            )  # [B, ch, k, 2, 2]
            theta_xy = (p[..., 5].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi  # [B, ch, k]
            w = F.softmax(p[..., 6].reshape(B, ch, k), dim=-1)  # [B, ch, k]
            alpha = torch.sigmoid(p[..., 7].reshape(B, ch, k))  # [B, ch, k]
            c = F.softplus(p[..., 8].reshape(B, ch, k)) + self.min_diag  # [B, ch, k]
            if ch == 1:
                L_color_params = p[..., 9:10].reshape(B, ch, k, 1)  # [B,1,k,1]
                color_mean = torch.zeros_like(mu_x)  # [B,1,k,1]
            elif ch == 3:
                L_color_params = p[..., 9:15].reshape(B, ch, k, 6)  # [B,3,k,6]
                color_mean = p[..., 15:18].reshape(B, ch, k, 3)  # [B,3,k,3]
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            color_cov_size = 1 if ch == 1 else 3
            L_color = self.construct_lower_triangular(
                L_color_params, s=color_cov_size
            )  # [B, ch, k, size, size]
            mu_xy = torch.cat([mu_x, mu_y, color_mean], dim=-1)  # [B, ch, k, D]
            cov_matrix = self.cov_mat(
                L_spatial, theta_xy, L_color, ch
            )  # [B, ch, k, D, D]
            return mu_xy, cov_matrix, w, alpha, c
        elif self.kernel_type == KernelType.GAUSSIAN:
            mu_x = p[..., 0].reshape(B, ch, k, 1)  # [B, ch, k, 1]
            mu_y = p[..., 1].reshape(B, ch, k, 1)  # [B, ch, k, 1]
            L_spatial_params = p[..., 2:5].reshape(B, ch, k, 3)  # [B, ch, k, 3]
            L_spatial = self.construct_lower_triangular(
                L_spatial_params, s=2
            )  # [B, ch, k, 2, 2]
            theta_xy = (p[..., 5].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi  # [B, ch, k]
            w = F.softmax(p[..., 6].reshape(B, ch, k), dim=-1)  # [B, ch, k]
            if ch == 1:
                L_color_params = p[..., 7:8].reshape(B, ch, k, 1)  # [B,1,k,1]
                color_mean = torch.zeros_like(mu_x)  # [B,1,k,1]
            elif ch == 3:
                L_color_params = p[..., 7:13].reshape(B, ch, k, 6)  # [B,3,k,6]
                color_mean = p[..., 13:16].reshape(B, ch, k, 3)  # [B,3,k,3]
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            color_cov_size = 1 if ch == 1 else 3
            L_color = self.construct_lower_triangular(
                L_color_params, s=color_cov_size
            )  # [B, ch, k, size, size]
            mu_xy = torch.cat([mu_x, mu_y, color_mean], dim=-1)  # [B, ch, k, D]
            cov_matrix = self.cov_mat(
                L_spatial, theta_xy, L_color, ch
            )  # [B, ch, k, D, D]
            return mu_xy, cov_matrix, w, None, None
        else:
            raise NotImplementedError(
                f"Kernel type {self.kernel_type} not implemented."
            )

    def gaussian_cauchy_kernel(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        Sigma_inv: torch.Tensor,
        alpha: Optional[torch.Tensor],
        c: Optional[torch.Tensor],
    ) -> torch.Tensor:
        d = x - mu  # [B, ch, k, W, H, D]
        e = -0.5 * torch.einsum(
            "bckwhd,bckde,bckwhe->bckwh", d, Sigma_inv, d
        )  # [B, ch, k, W, H]
        mx = e.max(dim=2, keepdim=True).values  # [B, ch, 1, W, H]
        e = e - mx  # [B, ch, k, W, H]
        G_sigma = torch.exp(e)  # [B, ch, k, W, H]
        norm_x = torch.linalg.norm(d[..., :2], dim=-1)  # [B, ch, k, W, H]
        Sigma_inv_diag = Sigma_inv[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
        denom = c.unsqueeze(-1).unsqueeze(-1) * Sigma_inv_diag.clamp(
            min=self.min_diag
        )  # [B, ch, k, 1, 1]
        denom = denom.clamp(min=self.min_denom)
        C_csigma = 1.0 / (1.0 + norm_x**2 / denom)  # [B, ch, k, W, H]
        combined = (
            alpha.unsqueeze(-1).unsqueeze(-1) * G_sigma
            + (1 - alpha.unsqueeze(-1).unsqueeze(-1)) * C_csigma
        )  # [B, ch, k, W, H]
        return combined  # [B, ch, k, W, H]

    def gaussian_kernel(
        self, x: torch.Tensor, mu_spatial: torch.Tensor, Sigma_inv_spatial: torch.Tensor
    ) -> torch.Tensor:
        d = x - mu_spatial  # [B, ch, k, W, H, 2]
        e = -0.5 * torch.einsum(
            "bckwhd,bckde,bckwhe->bckwh", d, Sigma_inv_spatial, d
        )  # [B, ch, k, W, H]
        mx = e.max(dim=2, keepdim=True).values  # [B, ch, 1, W, H]
        e = e - mx  # [B, ch, k, W, H]
        G_sigma = torch.exp(e)  # [B, ch, k, W, H]
        return G_sigma  # [B, ch, k, W, H]

    def forward_spatial(self, h: int, w: int, params: torch.Tensor) -> torch.Tensor:
        B, ch, _ = params.shape  # [B, ch, k * param_per_kernel]
        k = self.kernel  # int

        mu, cov, wt, alp, cst = self.extract_parameters(params, k, ch)
        # mu_xy: [B, ch, k, D]
        # cov_matrix: [B, ch, k, D, D]
        # w: [B, ch, k]
        # alpha: [B, ch, k] or None
        # c: [B, ch, k] or None

        d = cov.shape[-1]
        # I = torch.eye(d, device=cov.device).view(1, 1, 1, d, d)
        I = torch.eye(d, device=cov.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        eig = torch.linalg.eigvalsh(cov)
        m = eig.min(dim=-1).values[..., None, None]
        eps = F.softplus(-m) + 1e-8
        cov_reg = cov + (1e-6 + eps) * I

        L_chol = torch.linalg.cholesky(cov_reg)  # [B, ch, k, D, D]
        SI = torch.cholesky_inverse(L_chol)  # [B, ch, k, D, D]

        # Sigma_inv = torch.linalg.solve(cov_matrix + eps * eye_d, eye_d)

        G = self.grid(h, w, params.device)  # [W, H, 2]
        G_exp = G.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # [1,1,1,W,H,2]
        G_exp = G_exp.repeat(B, ch, k, 1, 1, 1)

        mu_full = mu.unsqueeze(3).unsqueeze(4)  # [B, ch, k, 1, 1, D]

        if ch == 1:
            CZ = (
                torch.zeros_like(mu[..., -1:]).unsqueeze(3).unsqueeze(4)
            )  # [B,1,k,1,1,1]
            CZ = CZ.expand(-1, -1, -1, h, w, -1)  # [B,1,k,H,W,1]
            X = torch.cat([G_exp, CZ], dim=-1)  # [B,1,k,W,H,3]
        elif ch == 3:
            CM = mu[..., -3:].unsqueeze(3).unsqueeze(4)  # [B,3,k,1,1,3]
            CM_exp = CM.expand(-1, -1, -1, h, w, -1)  # [B,3,k,H,W,3]
            X = torch.cat([G_exp, CM_exp], dim=-1)  # [B,3,k,W,H,5]
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")

        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            K_out = self.gaussian_cauchy_kernel(
                X, mu_full, SI, alp, cst
            )  # [B, ch, k, W, H]
        else:
            mu_sp = mu[..., :2].reshape(B, ch, k, 1, 1, 2)  # [B, ch, k,1,1,2]
            SI_sp = SI[..., :2, :2]  # [B, ch, k,2,2]
            K_out = self.gaussian_kernel(G_exp, mu_sp, SI_sp)  # [B, ch, k, W, H]

        K_out = K_out * wt.unsqueeze(-1).unsqueeze(-1)  # [B, ch, k, W, H]
        KS = K_out.sum(dim=2, keepdim=True)  # [B, ch, 1, W, H]
        K_norm = K_out / (KS + 1e-8)  # [B, ch, k, W, H]
        out = K_norm.sum(dim=2)  # [B, ch, W, H]
        return torch.clamp(out, min=0.0, max=1.0)  # [B, ch, W, H]

    def extract_dynamic(
        self, x: torch.Tensor, cnt: torch.Tensor, p: int
    ) -> torch.Tensor:
        # x: [B, C, L] where L = padded length = max_possible * p; cnt: [B]
        B, C, _ = x.shape
        K = int(cnt.max().item())
        lst = []
        for i in range(B):
            k_i = int(cnt[i].item())
            xi = x[i, :, : k_i * p].view(C, k_i, p)  # [C, k_i, p]
            if k_i < K:
                pad = torch.zeros(C, K - k_i, p, device=x.device, dtype=x.dtype)
                xi = torch.cat([xi, pad], dim=1)
            lst.append(xi.unsqueeze(0))

        return torch.cat(lst, dim=0)  # [B, C, K, p]

    def forward_spatial_(
        self, h: int, w: int, params: torch.Tensor, cnt: torch.Tensor
    ) -> torch.Tensor:
        B, ch, L = params.shape  # L = padded length = p * (padded kernel count)
        p = L // (cnt.max().item() if cnt.numel() > 0 else 1)
        x_dyn = self.extract_dynamic(params, cnt, p)
        x_flat = x_dyn.view(B, ch, -1)
        mu, cov, wt, alp, cst = self.extract_parameters(x_flat, x_dyn.shape[2], ch)
        d = cov.shape[-1]
        # I = torch.eye(d, device=cov.device).view(1, 1, 1, d, d)
        I = torch.eye(d, device=cov.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        eig = torch.linalg.eigvalsh(cov)
        m = eig.min(dim=-1).values[..., None, None]
        eps = F.softplus(-m) + 1e-8
        cov_reg = cov + (1e-6 + eps) * I
        L_chol = torch.linalg.cholesky(cov_reg)
        SI = torch.cholesky_inverse(L_chol)
        G = self.grid(h, w, params.device)
        G_exp = (
            G.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(2)
            .repeat(B, ch, x_dyn.shape[2], 1, 1, 1)
        )
        mu_full = mu.unsqueeze(3).unsqueeze(4)
        if ch == 1:
            CZ = torch.zeros_like(mu[..., -1:]).unsqueeze(3).unsqueeze(4)
            CZ = CZ.expand(-1, -1, -1, h, w, -1)
            X = torch.cat([G_exp, CZ], dim=-1)
        elif ch == 3:
            CM = mu[..., -3:].unsqueeze(3).unsqueeze(4)
            CM_exp = CM.expand(-1, -1, -1, h, w, -1)
            X = torch.cat([G_exp, CM_exp], dim=-1)
        else:
            raise ValueError(f"Unsupported ch: {ch}")
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            K_out = self.gaussian_cauchy_kernel(X, mu_full, SI, alp, cst)
        else:
            mu_sp = mu[..., :2].reshape(B, ch, x_dyn.shape[2], 1, 1, 2)
            SI_sp = SI[..., :2, :2]
            K_out = self.gaussian_kernel(G_exp, mu_sp, SI_sp)

        K_out = K_out * wt.unsqueeze(-1).unsqueeze(-1)
        KS = K_out.sum(dim=2, keepdim=True)
        K_norm = K_out / (KS + 1e-8)
        out = K_norm.sum(dim=2)
        return torch.clamp(out, 0.0, 1.0)

    def forward(
        self, h: int, w: int, params: torch.Tensor, cnt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if cnt is None:
            return self.forward_spatial(h, w, params)
        else:
            return self.forward_spatial_(h, w, params, cnt)


class Autoencoder(Backbone[AutoencoderConfig]):
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__(cfg)
        self.phw: int = cfg.phw
        self.overlap: int = cfg.overlap

        d_out, params_per_kernel = self.num_params(
            cfg.DecoderConfig.kernel_type, cfg.d_in, cfg.DecoderConfig.kernel
        )

        self.snet = DnCNN(
            in_channels=cfg.d_in,
            out_channels=cfg.EncoderConfig.sigma_chn,
            dep=cfg.dep_S,
            noise_avg=cfg.EncoderConfig.noise_avg,
        )
        self.knet = KernelNet(
            in_nc=cfg.d_in,
            out_chn=cfg.EncoderConfig.kernel_chn,
            num_blocks=cfg.dep_K,
        )

        self.encoder = Encoder(
            cfg=cfg.EncoderConfig, phw=cfg.phw, d_in=cfg.d_in, d_out=d_out
        )
        self.decoder = MoE(cfg=cfg.DecoderConfig)

        self.params_per_kernel = params_per_kernel
        self.register_buffer("ml", torch.tensor(self.mem_lim()))

    @staticmethod
    def hann_window(
        block_size: int, C: int, step: int, device: torch.device
    ) -> torch.Tensor:
        hann_1d = torch.hann_window(block_size, periodic=False, device=device)
        hann_2d = hann_1d.unsqueeze(1) * hann_1d.unsqueeze(0)
        window = hann_2d.view(1, 1, block_size * block_size)
        window = window.repeat(C, 1, 1)
        window = window.view(1, C * block_size * block_size, 1)
        window = window * (step / block_size)
        return window

    def num_params(self, kernel_type: KernelType, ch: int, kernel: int) -> int:
        num_parms = self.get_params_per_kernel(kernel_type, ch)
        num_params_per_kernel = num_parms * kernel
        return num_params_per_kernel, num_parms

    @staticmethod
    def get_params_per_kernel(kernel_type: any, ch: int) -> int:
        if kernel_type == KernelType.GAUSSIAN:
            return 10 if ch == 1 else 17
        elif kernel_type == KernelType.GAUSSIAN_CAUCHY:
            return 12 if ch == 1 else 18
        else:
            raise NotImplementedError(f"Unsupported kernel type: {kernel_type}")

    # @torch.compile
    def extract_blocks(
        self, img_tensor: torch.Tensor, block_size: int, overlap: int
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
        B, C, H, W = img_tensor.shape
        step = block_size - overlap
        pad = (block_size - step) // 2 + step
        img_padded = F.pad(img_tensor, (pad, pad, pad, pad), mode="reflect")
        patches = F.unfold(img_padded, kernel_size=block_size, stride=step)
        window = self.hann_window(block_size, C, step, img_tensor.device)
        L = patches.shape[-1]
        window = window.repeat(1, 1, L)
        patches = patches * window
        patches = (
            patches.view(B, C, block_size, block_size, L)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        return patches, (B, L, C, H, W, pad)

    # @torch.compile
    def reconstruct(
        self,
        decoded_patches: torch.Tensor,
        dims: Tuple[int, int, int, int, int, int],
        block_size_out: int,
        overlap_out: int,
    ) -> torch.Tensor:
        B, L, C, H, W, pad_out = dims
        step_out = block_size_out - overlap_out

        decoded_patches = (
            decoded_patches.reshape(B, L, C * block_size_out * block_size_out)
            .permute(0, 2, 1)
            .contiguous()
        )

        recon_padded = F.fold(
            decoded_patches,
            output_size=(H + 2 * pad_out, W + 2 * pad_out),
            kernel_size=block_size_out,
            stride=step_out,
        )
        window = self.hann_window(block_size_out, C, step_out, decoded_patches.device)
        window_sum = F.fold(
            torch.ones_like(decoded_patches) * window,
            output_size=(H + 2 * pad_out, W + 2 * pad_out),
            kernel_size=block_size_out,
            stride=step_out,
        )
        recon_padded = recon_padded / window_sum.clamp_min(1e-8)
        recon = recon_padded[:, :, pad_out : pad_out + H, pad_out : pad_out + W]
        return recon

    @staticmethod
    def mem_lim():
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            torch.cuda.set_device(device)
            free_mem, tot_mem = torch.cuda.mem_get_info(device)
            thresholds = [0.9, 0.8, 0.7]
            for percent in thresholds:
                threshold = tot_mem * percent
                if free_mem > threshold:
                    return threshold
            return max(1 * 2**30, tot_mem * 0.05)
        return 1 * 2**30

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x_p, dims = self.extract_blocks(x, self.phw, self.overlap)

        if x_p.ndim == 5:
            x_p = x_p.reshape(-1, *x_p.shape[2:])

        es: int = x_p.element_size()
        bm: int = x_p.shape[1:].numel() * es
        mx_bs = self.ml // bm
        n: int = int(max(1, min(x_p.shape[0] // 1024, mx_bs)))
        cs: int = int((x_p.shape[0] + n - 1) // n)

        res = [
            self.encoder(chunk, self.snet(chunk), self.knet(chunk))
            for chunk in torch.split(x_p, cs)
        ]

        gaussians, kinfo, sigma = map(lambda arr: torch.cat(arr, dim=0), zip(*res))

        B, L, C, H, W, pad = dims  # => (B, L, C, H, W, pad)
        sp: int = self.phw * self.encoder.scale_factor

        dec: torch.Tensor = torch.cat(
            [self.decoder(sp, sp, bt) for bt in torch.split(gaussians, cs)], dim=0
        )

        y: torch.Tensor = self.reconstruct(
            dec,
            (
                B,
                L,
                C,
                H * self.encoder.scale_factor,
                W * self.encoder.scale_factor,
                pad * self.encoder.scale_factor,
            ),
            sp,
            self.overlap * self.encoder.scale_factor,
        )  # => (B, C, H, W)

        kinfo = kinfo.view(B, L, -1).mean(dim=1)
        sigma = sigma.view(B, L, 1, 1, 1).mean(dim=1)

        return y, kinfo, sigma
