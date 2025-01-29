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


# torch.set_float32_matmul_precision("high")

backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

OLD_GPU = True
USE_FLASH_ATTN = False
MATH_KERNEL_ON = True

T = TypeVar("T")
log_min = math.log(1e-10)
log_max = math.log(1e2)


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
        self.gaussian_kernel = self.create_gaussian_kernel(kernel_size, stddev)

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
            nn.InstanceNorm2d, affine=True, track_running_stats=False
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

        sigma = torch.exp(torch.clamp(sigma_est, min=log_min, max=log_max))

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


class MoE_(Backbone[MoEConfig]):
    def __init__(self, cfg: MoEConfig):
        super(MoE, self).__init__(cfg)
        self.kernel = cfg.kernel
        self.sharpening_factor = cfg.sharpening_factor
        self.kernel_type = cfg.kernel_type

    def grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        xx = torch.linspace(0.0, 1.0, width, device=device)
        yy = torch.linspace(0.0, 1.0, height, device=device)
        gx, gy = torch.meshgrid(xx, yy, indexing="ij")
        return torch.stack((gx, gy), dim=-1).float()

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        ct = torch.cos(theta).unsqueeze(-1)
        st = torch.sin(theta).unsqueeze(-1)
        R = torch.cat([ct, -st, st, ct], dim=-1)
        return R.view(*theta.shape, 2, 2)

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        B, _, _ = p.shape
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            mu_x = p[:, :, 0:k].reshape(B, ch, k, 1)
            mu_y = p[:, :, k : 2 * k].reshape(B, ch, k, 1)
            scale_xy = F.softplus(p[:, :, 2 * k : 4 * k].reshape(B, ch, k, 2)) + 1e-3
            theta_xy = (p[:, :, 4 * k : 5 * k].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            w = F.softmax(p[:, :, 5 * k : 6 * k].reshape(B, ch, k), dim=-1)
            alpha = torch.sigmoid(p[:, :, 6 * k : 7 * k].reshape(B, ch, k))
            c = F.softplus(p[:, :, 7 * k : 8 * k].reshape(B, ch, k)) + 1e-3

            scale_color = (
                F.softplus(p[:, :, 8 * k : 11 * k].reshape(B, ch, k, 3)) + 1e-3
            )
            if ch == 3:
                rho_color = torch.tanh(p[:, :, 11 * k : 14 * k].reshape(B, ch, k, 3))
            else:
                rho_color = torch.tanh(p[:, :, 11 * k : 12 * k].reshape(B, ch, k, 1))

            L = F.softplus(p[:, :, 14 * k : 20 * k].reshape(B, ch, k, 3, 2))
            l33 = F.softplus(p[:, :, 20 * k : 23 * k].reshape(B, ch, k, 3, 1))
            L_full = torch.cat([L, l33], dim=-1)
        else:
            mu_x = p[:, :, 0:k].reshape(B, ch, k, 1)
            mu_y = p[:, :, k : 2 * k].reshape(B, ch, k, 1)
            scale_xy = F.softplus(p[:, :, 2 * k : 4 * k].reshape(B, ch, k, 2)) + 1e-3
            theta_xy = (p[:, :, 4 * k : 5 * k].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            w = F.softmax(p[:, :, 5 * k : 6 * k].reshape(B, ch, k), dim=-1)

            scale_color = F.softplus(p[:, :, 6 * k : 9 * k].reshape(B, ch, k, 3)) + 1e-3
            if ch == 3:
                rho_color = torch.tanh(p[:, :, 9 * k : 12 * k].reshape(B, ch, k, 3))
            else:
                rho_color = torch.tanh(p[:, :, 9 * k : 10 * k].reshape(B, ch, k, 1))

            L = F.softplus(p[:, :, 12 * k : 18 * k].reshape(B, ch, k, 3, 2))
            l33 = F.softplus(p[:, :, 18 * k : 21 * k].reshape(B, ch, k, 3, 1))
            L_full = torch.cat([L, l33], dim=-1)

            alpha = None
            c = None

        mu = torch.cat([mu_x, mu_y], dim=-1)
        cov_matrix = (
            self.cov_mat(scale_xy, theta_xy, scale_color, rho_color, L_full, ch)
            * self.sharpening_factor
        )

        return mu, cov_matrix, w, alpha, c

    def cov_mat(
        self,
        scale: torch.Tensor,
        theta_xy: torch.Tensor,
        scale_color: torch.Tensor,
        rho_color: Optional[torch.Tensor],
        L_full: torch.Tensor,
        ch: int,
    ) -> torch.Tensor:
        R = self.ang_to_rot_mat(theta_xy)
        S = torch.diag_embed(scale)
        C_xy = torch.matmul(R, torch.matmul(S, S.transpose(-2, -1)))

        if ch == 3:
            C_rgb = torch.matmul(L_full, L_full.transpose(-2, -1))
            if rho_color is not None:
                rho_color = rho_color.unsqueeze(-1)  # Shape: (B, ch, k, 3, 1)
                rho_color_t = rho_color.transpose(-2, -1)  # Shape: (B, ch, k, 1, 3)
                C_rgb += torch.matmul(rho_color, rho_color_t)  # Shape: (B, ch, k, 3, 3)
            C_rgb += 1e-3 * torch.eye(3, device=scale.device).view(1, 1, 1, 3, 3)

            C_full = torch.zeros(*C_xy.shape[:-2], 5, 5, device=scale.device)
            C_full[..., :2, :2] = C_xy
            C_full[..., 2:, 2:] = C_rgb
        else:
            C_color = scale_color.squeeze(-1).squeeze(-1)
            C_full = torch.zeros(*C_xy.shape[:-2], 3, 3, device=scale.device)
            C_full[..., :2, :2] = C_xy
            C_full[..., 2, 2] = C_color
            C_full += 1e-3 * torch.eye(3, device=scale.device).view(1, 1, 1, 3, 3)

        return C_full

    def gaussian_cauchy_kernel(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        Sigma_inv: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_sub_mu = x - mu
        x_sub_mu_t = x_sub_mu.unsqueeze(-1)

        exp_terms = -0.5 * torch.einsum(
            "bnkhwli,bnkhwlj,bnkhwmi->bnkhw",
            x_sub_mu_t,
            Sigma_inv,
            x_sub_mu_t,
        ).squeeze(-1)

        max_exp_terms = torch.max(exp_terms, dim=2, keepdim=True).values
        exp_terms = exp_terms - max_exp_terms
        G_sigma = torch.exp(exp_terms)

        norm_x = torch.linalg.norm(x_sub_mu, dim=-1)
        H, W = norm_x.shape[-2], norm_x.shape[-1]
        c_expanded = c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)
        Sigma_diag = Sigma_inv[..., 0, 0]

        denominator = c_expanded * Sigma_diag.clamp(min=1e-8)
        C_csigma = 1 / (1 + norm_x**2 / denominator)
        alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1)

        blended_kers = alpha_expanded * G_sigma + (1 - alpha_expanded) * C_csigma

        return blended_kers

    def gaussian_kernel(
        self, x: torch.Tensor, mu: torch.Tensor, Sigma_inv: torch.Tensor
    ) -> torch.Tensor:
        x_sub_mu = x - mu
        x_sub_mu_t = x_sub_mu.unsqueeze(-1)
        exp_terms = -0.5 * torch.einsum(
            "bnkhwli,bnkhwlj,bnkhwmi->bnkhw", x_sub_mu_t, Sigma_inv, x_sub_mu_t
        ).squeeze(-1)
        max_exp_terms = torch.max(exp_terms, dim=2, keepdim=True).values
        exp_terms = exp_terms - max_exp_terms
        return torch.exp(exp_terms)

    def forward(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        return self.forward_spatial(height, width, params)

    def forward_spatial(
        self, height: int, width: int, params: torch.Tensor
    ) -> torch.Tensor:
        B, ch, _ = params.shape
        k = self.kernel

        mu, cov_matrix, w, alpha, c = self.extract_parameters(params, k, ch)

        eps = 1e-6
        d = cov_matrix.shape[-1]

        eye_d = (
            torch.eye(d, device=cov_matrix.device).unsqueeze(0).unsqueeze(0)
        )  # Shape: (1, 1, d, d)

        cov_matrix = (cov_matrix + cov_matrix.transpose(-2, -1)) / 2

        cov_matrix_reg = cov_matrix + eps * eye_d  # Shape: (B, ch, d, d)

        I = (
            torch.eye(d, device=cov_matrix.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1,1,1,d,d)
        I = I.expand(B, ch, k, d, d)  # (B, ch, k, d, d)
        Sigma_inv = torch.linalg.solve(cov_matrix_reg, I)  # (B, ch, k, d, d)

        device = params.device

        g = self.grid(height, width, device)  # Shape: (height, width, 2)

        g_color = torch.zeros(
            height, width, ch, device=device
        )  # Shape: (height, width, ch)

        g_full = torch.cat([g, g_color], dim=-1)  # Shape: (height, width, 2 + ch)

        g_full = (
            g_full.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )  # Shape: (1, 1, 1, height, width, 2 + ch)

        mu_color = torch.zeros(B, ch, k, ch, device=device)  # Shape: (B, ch, k, ch)

        mu_full = (
            torch.cat([mu, mu_color], dim=-1).unsqueeze(3).unsqueeze(4)
        )  # Shape: (B, ch, k, 1, 1, 2 + ch)

        S = Sigma_inv.unsqueeze(3).unsqueeze(4)  # Shape: (B, ch, k, 1, 1, d, d)

        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            ker = self.gaussian_cauchy_kernel(g_full, mu_full, S, alpha, c)
        else:
            ker = self.gaussian_kernel(g_full, mu_full, S)

        ker = ker * w.view(B, ch, k, 1, 1)  # Shape: (B, ch, k, height, width)

        ker = ker / (
            ker.sum(dim=2, keepdim=True) + 1e-8
        )  # Shape: (B, ch, k, height, width)

        out = ker.sum(dim=2)  # Shape: (B, ch, height, width)

        return out.clamp(min=0, max=1)


class MoE__(Backbone[MoEConfig]):
    def __init__(self, cfg: MoEConfig):
        super(MoE, self).__init__()
        self.kernel = cfg.kernel
        self.sharpening_factor = cfg.sharpening_factor
        self.kernel_type = cfg.kernel_type
        self.learnable_eps = True
        self.eps_log = nn.Parameter(torch.tensor(-6.9))
        self.clamp_alpha = True
        self.alpha_min, self.alpha_max = 0.01, 0.99
        self.clamp_c = True
        self.c_min, self.c_max = 1e-4, 1e4
        self.clamp_scale = True
        self.scale_min, self.scale_max = 1e-4, 1e2
        self.clamp_rho_color = True
        self.rho_color_max = 1.0
        self.enable_reg = True
        self.reg_logdet_weight = 1e-5

    def grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        xx = torch.linspace(0.0, 1.0, width, device=device)
        yy = torch.linspace(0.0, 1.0, height, device=device)
        gx, gy = torch.meshgrid(xx, yy, indexing="ij")
        return torch.stack((gx, gy), dim=-1).float()

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        ct = torch.cos(theta).unsqueeze(-1)
        st = torch.sin(theta).unsqueeze(-1)
        R = torch.cat([ct, -st, st, ct], dim=-1)
        return R.view(*theta.shape, 2, 2)

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        B, _, _ = p.shape
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            mu_x = p[:, :, 0:k].reshape(B, ch, k, 1)
            mu_y = p[:, :, k : 2 * k].reshape(B, ch, k, 1)
            scale_xy = F.softplus(p[:, :, 2 * k : 4 * k].reshape(B, ch, k, 2)) + 1e-3
            theta_xy = (p[:, :, 4 * k : 5 * k].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            w = F.softmax(p[:, :, 5 * k : 6 * k].reshape(B, ch, k), dim=-1)
            alpha = torch.sigmoid(p[:, :, 6 * k : 7 * k].reshape(B, ch, k))
            c = F.softplus(p[:, :, 7 * k : 8 * k].reshape(B, ch, k)) + 1e-3
            scale_color = (
                F.softplus(p[:, :, 8 * k : 11 * k].reshape(B, ch, k, 3)) + 1e-3
            )
            if ch == 3:
                rho_color = torch.tanh(p[:, :, 11 * k : 14 * k].reshape(B, ch, k, 3))
            else:
                rho_color = torch.tanh(p[:, :, 11 * k : 12 * k].reshape(B, ch, k, 1))
            L = F.softplus(p[:, :, 14 * k : 20 * k].reshape(B, ch, k, 3, 2))
            l33 = F.softplus(p[:, :, 20 * k : 23 * k].reshape(B, ch, k, 3, 1))
            L_full = torch.cat([L, l33], dim=-1)
        else:
            mu_x = p[:, :, 0:k].reshape(B, ch, k, 1)
            mu_y = p[:, :, k : 2 * k].reshape(B, ch, k, 1)
            scale_xy = F.softplus(p[:, :, 2 * k : 4 * k].reshape(B, ch, k, 2)) + 1e-3
            theta_xy = (p[:, :, 4 * k : 5 * k].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            w = F.softmax(p[:, :, 5 * k : 6 * k].reshape(B, ch, k), dim=-1)
            scale_color = F.softplus(p[:, :, 6 * k : 9 * k].reshape(B, ch, k, 3)) + 1e-3
            if ch == 3:
                rho_color = torch.tanh(p[:, :, 9 * k : 12 * k].reshape(B, ch, k, 3))
            else:
                rho_color = torch.tanh(p[:, :, 9 * k : 10 * k].reshape(B, ch, k, 1))
            L = F.softplus(p[:, :, 12 * k : 18 * k].reshape(B, ch, k, 3, 2))
            l33 = F.softplus(p[:, :, 18 * k : 21 * k].reshape(B, ch, k, 3, 1))
            L_full = torch.cat([L, l33], dim=-1)
            alpha = None
            c = None
        if self.clamp_scale:
            scale_xy = scale_xy.clamp(self.scale_min, self.scale_max)
            scale_color = scale_color.clamp(self.scale_min, self.scale_max)
        if alpha is not None and self.clamp_alpha:
            alpha = alpha.clamp(self.alpha_min, self.alpha_max)
        if c is not None and self.clamp_c:
            c = c.clamp(self.c_min, self.c_max)
        if self.clamp_rho_color and rho_color is not None:
            rho_color = rho_color.clamp(min=-self.rho_color_max, max=self.rho_color_max)
        mu = torch.cat([mu_x, mu_y], dim=-1)
        cov_matrix = (
            self.cov_mat(scale_xy, theta_xy, scale_color, rho_color, L_full, ch)
            * self.sharpening_factor
        )
        return mu, cov_matrix, w, alpha, c

    def cov_mat(
        self,
        scale: torch.Tensor,
        theta_xy: torch.Tensor,
        scale_color: torch.Tensor,
        rho_color: Optional[torch.Tensor],
        L_full: torch.Tensor,
        ch: int,
    ) -> torch.Tensor:
        R = self.ang_to_rot_mat(theta_xy)
        S = torch.diag_embed(scale)
        C_xy = torch.matmul(R, torch.matmul(S, S.transpose(-2, -1)))
        if ch == 3:
            C_rgb = torch.matmul(L_full, L_full.transpose(-2, -1))
            if rho_color is not None:
                rho_color_ = rho_color.unsqueeze(-1)
                rho_color_t = rho_color_.transpose(-2, -1)
                C_rgb += torch.matmul(rho_color_, rho_color_t)
            if self.learnable_eps:
                eps_val = torch.exp(self.eps_log).clamp(1e-8, 1e-2)
            else:
                eps_val = 1e-3
            C_rgb += eps_val * torch.eye(3, device=scale.device).view(1, 1, 1, 3, 3)
            C_full = torch.zeros(*C_xy.shape[:-2], 5, 5, device=scale.device)
            C_full[..., :2, :2] = C_xy
            C_full[..., 2:, 2:] = C_rgb
        else:
            C_color = scale_color.squeeze(-1).squeeze(-1)
            if self.learnable_eps:
                eps_val = torch.exp(self.eps_log).clamp(1e-8, 1e-2)
            else:
                eps_val = 1e-3
            C_full = torch.zeros(*C_xy.shape[:-2], 3, 3, device=scale.device)
            C_full[..., :2, :2] = C_xy
            C_full[..., 2, 2] = C_color
            C_full += eps_val * torch.eye(3, device=scale.device).view(1, 1, 1, 3, 3)
        return C_full

    def gaussian_cauchy_kernel_einsum(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        Sigma_inv: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_sub_mu = x - mu
        x_sub_mu_t = x_sub_mu.unsqueeze(-1)
        exp_terms = -0.5 * torch.einsum(
            "bnkhwli,bnkhwlj,bnkhwmi->bnkhw",
            x_sub_mu_t,
            Sigma_inv,
            x_sub_mu_t,
        ).squeeze(-1)
        max_exp_terms = torch.max(exp_terms, dim=2, keepdim=True).values
        exp_terms = exp_terms - max_exp_terms
        G_sigma = torch.exp(exp_terms)
        norm_x = torch.linalg.norm(x_sub_mu, dim=-1)
        H, W = norm_x.shape[-2], norm_x.shape[-1]
        if c is None or alpha is None:
            return G_sigma
        c_expanded = c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)
        Sigma_diag = Sigma_inv[..., 0, 0]
        denominator = c_expanded * Sigma_diag.clamp(min=1e-8)
        C_csigma = 1 / (1 + norm_x**2 / denominator)
        alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1)
        blended_kers = alpha_expanded * G_sigma + (1 - alpha_expanded) * C_csigma
        return blended_kers
    
    def gaussian_cauchy_kernel(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        Sigma_inv: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None):
        d = x - mu
        x1 = d.unsqueeze(-2)
        x2 = torch.matmul(Sigma_inv, d.unsqueeze(-1))
        e = -0.5 * torch.matmul(x1, x2).squeeze(-1).squeeze(-1)
        mx = e.max(dim=2, keepdim=True).values
        e = e - mx
        G_sigma = torch.exp(e)
        norm_x = torch.linalg.norm(d, dim=-1)
        H, W = norm_x.shape[-2], norm_x.shape[-1]
        c_e = c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)
        diag = Sigma_inv[..., 0, 0]
        denominator = c_e * diag.clamp(min=1e-8)
        C_csigma = 1 / (1 + norm_x**2 / denominator)
        a_e = alpha.unsqueeze(-1).unsqueeze(-1)
        blend = a_e * G_sigma + (1 - a_e) * C_csigma
        # s = blend.sum(dim=2, keepdim=True)
        # b = blend / (s + 1e-8)
        return blend
    
    def gaussian_kernel_einsum(
        self, x: torch.Tensor, mu: torch.Tensor, Sigma_inv: torch.Tensor
    ) -> torch.Tensor:
        x_sub_mu = x - mu
        x_sub_mu_t = x_sub_mu.unsqueeze(-1)
        exp_terms = -0.5 * torch.einsum(
            "bnkhwli,bnkhwlj,bnkhwmi->bnkhw", x_sub_mu_t, Sigma_inv, x_sub_mu_t
        ).squeeze(-1)
        max_exp_terms = torch.max(exp_terms, dim=2, keepdim=True).values
        exp_terms = exp_terms - max_exp_terms
        return torch.exp(exp_terms)

    def gaussian_kernel(
        self, x: torch.Tensor, mu: torch.Tensor, Sigma_inv: torch.Tensor
    ) -> torch.Tensor:
        d = x - mu
        x1 = d.unsqueeze(-2)
        x2 = torch.matmul(Sigma_inv, d.unsqueeze(-1))
        e = -0.5 * torch.matmul(x1, x2).squeeze(-1).squeeze(-1)
        mx = e.max(dim=2, keepdim=True).values
        e = e - mx
        G_sigma = torch.exp(e)
        return G_sigma

    def forward(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        return self.forward_spatial(height, width, params)

    def forward_spatial(
        self, height: int, width: int, params: torch.Tensor
    ) -> torch.Tensor:
        B, ch, _ = params.shape
        k = self.kernel
        mu, cov_matrix, w, alpha, c = self.extract_parameters(params, k, ch)
        cov_matrix = (cov_matrix + cov_matrix.transpose(-2, -1)) / 2
        d = cov_matrix.shape[-1]
        eye_d = torch.eye(d, device=cov_matrix.device).view(1, 1, 1, d, d)
        cov_matrix_reg = cov_matrix + 1e-6 * eye_d
        I = (
            torch.eye(d, device=cov_matrix.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        I = I.expand(B, ch, k, d, d)
        Sigma_inv = torch.linalg.solve(cov_matrix_reg, I)
        device = params.device
        g = self.grid(height, width, device)
        g_color = torch.zeros(height, width, ch, device=device)
        g_full = torch.cat([g, g_color], dim=-1)
        g_full = g_full.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mu_color = torch.zeros(B, ch, k, ch, device=device)
        mu_full = torch.cat([mu, mu_color], dim=-1).unsqueeze(3).unsqueeze(4)
        S = Sigma_inv.unsqueeze(3).unsqueeze(4)
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            ker = self.gaussian_cauchy_kernel(g_full, mu_full, S, alpha, c)
        else:
            ker = self.gaussian_kernel(g_full, mu_full, S)
        ker = ker * w.view(B, ch, k, 1, 1)
        ker = ker / (ker.sum(dim=2, keepdim=True) + 1e-8)
        out = ker.sum(dim=2)
        return out.clamp(min=0, max=1)


class MoE(Backbone[MoEConfig]):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.kernel = cfg.kernel
        self.sharpening_factor = cfg.sharpening_factor
        self.kernel_type = cfg.kernel_type
        self.learnable_eps = True
        self.eps_log = nn.Parameter(torch.tensor(-6.9))
        self.clamp_alpha = True
        self.alpha_min, self.alpha_max = 0.01, 0.99
        self.clamp_c = True
        self.c_min, self.c_max = 1e-4, 1e4
        self.clamp_scale = True
        self.scale_min, self.scale_max = 1e-4, 1e2
        self.clamp_rho_color = True
        self.rho_color_max = 1.0
       
        

    def grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        xx = torch.linspace(0.0, 1.0, width, device=device)
        yy = torch.linspace(0.0, 1.0, height, device=device)
        gx, gy = torch.meshgrid(xx, yy, indexing="ij")
        return torch.stack((gx, gy), dim=-1).float()

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        ct = torch.cos(theta).unsqueeze(-1)
        st = torch.sin(theta).unsqueeze(-1)
        R = torch.cat([ct, -st, st, ct], dim=-1)
        return R.view(*theta.shape, 2, 2)

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        B, _, _ = p.shape
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            mu_x = p[:, :, 0:k].reshape(B, ch, k, 1)
            mu_y = p[:, :, k : 2 * k].reshape(B, ch, k, 1)
            scale_xy = F.softplus(p[:, :, 2 * k : 4 * k].reshape(B, ch, k, 2)) + 1e-3
            theta_xy = (p[:, :, 4 * k : 5 * k].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            w = F.softmax(p[:, :, 5 * k : 6 * k].reshape(B, ch, k), dim=-1)
            alpha = torch.sigmoid(p[:, :, 6 * k : 7 * k].reshape(B, ch, k))
            c = F.softplus(p[:, :, 7 * k : 8 * k].reshape(B, ch, k)) + 1e-3
            scale_color = (
                F.softplus(p[:, :, 8 * k : 11 * k].reshape(B, ch, k, 3)) + 1e-3
            )
            if ch == 3:
                rho_color = torch.tanh(p[:, :, 11 * k : 14 * k].reshape(B, ch, k, 3))
            else:
                rho_color = torch.tanh(p[:, :, 11 * k : 12 * k].reshape(B, ch, k, 1))
            L = F.softplus(p[:, :, 14 * k : 20 * k].reshape(B, ch, k, 3, 2))
            l33 = F.softplus(p[:, :, 20 * k : 23 * k].reshape(B, ch, k, 3, 1))
            L_full = torch.cat([L, l33], dim=-1)
        else:
            mu_x = p[:, :, 0:k].reshape(B, ch, k, 1)
            mu_y = p[:, :, k : 2 * k].reshape(B, ch, k, 1)
            scale_xy = F.softplus(p[:, :, 2 * k : 4 * k].reshape(B, ch, k, 2)) + 1e-3
            theta_xy = (p[:, :, 4 * k : 5 * k].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            w = F.softmax(p[:, :, 5 * k : 6 * k].reshape(B, ch, k), dim=-1)
            scale_color = F.softplus(p[:, :, 6 * k : 9 * k].reshape(B, ch, k, 3)) + 1e-3
            if ch == 3:
                rho_color = torch.tanh(p[:, :, 9 * k : 12 * k].reshape(B, ch, k, 3))
            else:
                rho_color = torch.tanh(p[:, :, 9 * k : 10 * k].reshape(B, ch, k, 1))
            L = F.softplus(p[:, :, 12 * k : 18 * k].reshape(B, ch, k, 3, 2))
            l33 = F.softplus(p[:, :, 18 * k : 21 * k].reshape(B, ch, k, 3, 1))
            L_full = torch.cat([L, l33], dim=-1)
            alpha = None
            c = None
        if self.clamp_scale:
            scale_xy = scale_xy.clamp(self.scale_min, self.scale_max)
            scale_color = scale_color.clamp(self.scale_min, self.scale_max)
        if alpha is not None and self.clamp_alpha:
            alpha = alpha.clamp(self.alpha_min, self.alpha_max)
        if c is not None and self.clamp_c:
            c = c.clamp(self.c_min, self.c_max)
        if self.clamp_rho_color and rho_color is not None:
            rho_color = rho_color.clamp(min=-self.rho_color_max, max=self.rho_color_max)
        mu = torch.cat([mu_x, mu_y], dim=-1)
        cov_matrix = (
            self.cov_mat(scale_xy, theta_xy, scale_color, rho_color, L_full, ch)
            * self.sharpening_factor
        )
        return mu, cov_matrix, w, alpha, c

    def cov_mat(
        self,
        scale: torch.Tensor,
        theta_xy: torch.Tensor,
        scale_color: torch.Tensor,
        rho_color: Optional[torch.Tensor],
        L_full: torch.Tensor,
        ch: int,
    ) -> torch.Tensor:
        R = self.ang_to_rot_mat(theta_xy)
        S = torch.diag_embed(scale)
        C_xy = torch.matmul(R, torch.matmul(S, S.transpose(-2, -1)))
        if ch == 3:
            C_rgb = torch.matmul(L_full, L_full.transpose(-2, -1))
            if rho_color is not None:
                rho_color_ = rho_color.unsqueeze(-1)
                rho_color_t = rho_color_.transpose(-2, -1)
                C_rgb += torch.matmul(rho_color_, rho_color_t)
            if self.learnable_eps:
                eps_val = torch.exp(self.eps_log).clamp(1e-8, 1e-2)
            else:
                eps_val = 1e-3
            C_rgb += eps_val * torch.eye(3, device=scale.device).view(1, 1, 1, 3, 3)
            C_full = torch.zeros(*C_xy.shape[:-2], 5, 5, device=scale.device)
            C_full[..., :2, :2] = C_xy
            C_full[..., 2:, 2:] = C_rgb
        else:
            C_color = scale_color.squeeze(-1).squeeze(-1)
            if self.learnable_eps:
                eps_val = torch.exp(self.eps_log).clamp(1e-8, 1e-2)
            else:
                eps_val = 1e-3
            C_full = torch.zeros(*C_xy.shape[:-2], 3, 3, device=scale.device)
            C_full[..., :2, :2] = C_xy
            C_full[..., 2, 2] = C_color
            C_full += eps_val * torch.eye(3, device=scale.device).view(1, 1, 1, 3, 3)
        return C_full

    def gaussian_cauchy_kernel(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        Sigma_inv: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None):
        d = x - mu
        x1 = d.unsqueeze(-2)
        x2 = torch.matmul(Sigma_inv, d.unsqueeze(-1))
        e = -0.5 * torch.matmul(x1, x2).squeeze(-1).squeeze(-1)
        mx = e.max(dim=2, keepdim=True).values
        e = e - mx
        G_sigma = torch.exp(e)
        norm_x = torch.linalg.norm(d, dim=-1)
        H, W = norm_x.shape[-2], norm_x.shape[-1]
        c_e = c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)
        diag = Sigma_inv[..., 0, 0]
        denominator = c_e * diag.clamp(min=1e-8)
        C_csigma = 1 / (1 + norm_x**2 / denominator)
        a_e = alpha.unsqueeze(-1).unsqueeze(-1)
        blend = a_e * G_sigma + (1 - a_e) * C_csigma
        # s = blend.sum(dim=2, keepdim=True)
        # b = blend / (s + 1e-8)
        return blend
    
    def gaussian_kernel(
        self, x: torch.Tensor, mu: torch.Tensor, Sigma_inv: torch.Tensor
    ) -> torch.Tensor:
        d = x - mu
        x1 = d.unsqueeze(-2)
        x2 = torch.matmul(Sigma_inv, d.unsqueeze(-1))
        e = -0.5 * torch.matmul(x1, x2).squeeze(-1).squeeze(-1)
        mx = e.max(dim=2, keepdim=True).values
        e = e - mx
        G_sigma = torch.exp(e)
        return G_sigma

    def forward(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        return self.forward_spatial(height, width, params)

    def forward_spatial(
        self, height: int, width: int, params: torch.Tensor
    ) -> torch.Tensor:
        B, ch, _ = params.shape
        k = self.kernel
        mu, cov_matrix, w, alpha, c = self.extract_parameters(params, k, ch)
        cov_matrix = (cov_matrix + cov_matrix.transpose(-2, -1)) / 2
        d = cov_matrix.shape[-1]
        eye_d = torch.eye(d, device=cov_matrix.device).view(1, 1, 1, d, d)
        cov_matrix_reg = cov_matrix + 1e-6 * eye_d
        I = (
            torch.eye(d, device=cov_matrix.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        I = I.expand(B, ch, k, d, d)
        Sigma_inv = torch.linalg.solve(cov_matrix_reg, I)
        device = params.device
        g = self.grid(height, width, device)
        g_color = torch.zeros(height, width, ch, device=device)
        g_full = torch.cat([g, g_color], dim=-1)
        g_full = g_full.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mu_color = torch.zeros(B, ch, k, ch, device=device)
        mu_full = torch.cat([mu, mu_color], dim=-1).unsqueeze(3).unsqueeze(4)
        S = Sigma_inv.unsqueeze(3).unsqueeze(4)
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            ker = self.gaussian_cauchy_kernel(g_full, mu_full, S, alpha, c)
        else:
            ker = self.gaussian_kernel(g_full, mu_full, S)
        ker = ker * w.view(B, ch, k, 1, 1)
        ker = ker / (ker.sum(dim=2, keepdim=True) + 1e-8)
        out = ker.sum(dim=2)
        return out.clamp(min=0, max=1)




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


class Autoencoder(Backbone[AutoencoderConfig]):
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__(cfg)
        self.phw: int = cfg.phw
        self.overlap: int = cfg.overlap

        if cfg.DecoderConfig.kernel_type == KernelType.GAUSSIAN:
            # Existing: (7 * d_in) * kernel
            # Add 3 for Cholesky parameters per kernel
            d_out = (7 * cfg.d_in + 3) * cfg.DecoderConfig.kernel
        else:
            # Existing: (7 * d_in + 3) * kernel
            # Add 6 for Cholesky parameters per kernel
            d_out = (7 * cfg.d_in + 3 + 6) * cfg.DecoderConfig.kernel

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
        self.decoder = MoE(cfg.DecoderConfig)

    @staticmethod
    def hann_window(
        block_size: int, C: int, step: int, device: torch.device
    ) -> torch.Tensor:
        hann_1d = torch.hann_window(block_size, periodic=False, device=device)
        hann_2d = hann_1d.unsqueeze(1) * hann_1d.unsqueeze(0)
        window = hann_2d.view(1, 1, block_size * block_size)
        window = window.repeat(C, 1, 1)
        window = window.view(1, C * block_size * block_size, 1)
        window = window * (
            step / block_size
        )  # Dynamic scaling based on step and block_size
        return window

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
        ml = self.mem_lim()
        bm: int = x_p.shape[1:].numel() * es
        mx_bs = ml // bm
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
