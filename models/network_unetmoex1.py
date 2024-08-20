import math
from dataclasses import dataclass
from typing import Any, Generic, Optional, Tuple, TypeVar

import cv2
from grpc import StatusCode
import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import cupy as cp

from utils_n.nn import avg_pool_nd, checkpoint, conv_nd, zero_module, GroupNorm32


T = TypeVar("T")


class Backbone(nn.Module, Generic[T]):
    def __init__(self, cfg: T):
        super().__init__()
        self.cfg = cfg

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def d_out(self) -> int:
        raise NotImplementedError


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.resample_2d = resample_2d
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3 and self.resample_2d:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = (1, 2, 2) if dims == 3 and resample_2d else 2
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
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


def normalization(channels, groups) -> GroupNorm32:

    return GroupNorm32(groups, channels)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


@dataclass
class ResBlockConfig:
    channels: int = 3
    dropout: float = 0.0
    out_channels: Optional[int] = None
    use_conv: bool = False
    dims: int = 2
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False
    num_groups: int = 32
    resample_2d: bool = True

    def __post_init__(self):
        self.out_channels = self.out_channels or self.channels


class ResBlock(Backbone[ResBlockConfig]):

    def __init__(self, cfg: ResBlockConfig, activation: nn.Module = nn.GELU()):
        super().__init__(cfg=cfg)

        self.activation = activation

        self.in_layers = nn.Sequential(
            normalization(cfg.channels, cfg.num_groups),
            self.activation,
            conv_nd(cfg.dims, cfg.channels, cfg.out_channels, 3, padding=1),
        )

        self.updown = cfg.up or cfg.down

        if cfg.up:
            self.h_upd = Upsample(
                cfg.channels, False, cfg.dims, resample_2d=cfg.resample_2d
            )
            self.x_upd = Upsample(
                cfg.channels, False, cfg.dims, resample_2d=cfg.resample_2d
            )
        elif cfg.down:
            self.h_upd = Downsample(
                cfg.channels, False, cfg.dims, resample_2d=cfg.resample_2d
            )
            self.x_upd = Downsample(
                cfg.channels, False, cfg.dims, resample_2d=cfg.resample_2d
            )
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(cfg.out_channels, cfg.num_groups),
            self.activation,
            nn.Dropout(p=cfg.dropout),
            zero_module(
                conv_nd(cfg.dims, cfg.out_channels, cfg.out_channels, 3, padding=1)
            ),
        )

        if cfg.out_channels == cfg.channels:
            self.skip_connection = nn.Identity()
        elif cfg.use_conv:
            self.skip_connection = conv_nd(
                cfg.dims, cfg.channels, cfg.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(cfg.dims, cfg.channels, cfg.out_channels, 1)

    def forward(self, x: torch.Tensor):
        return checkpoint(
            self._forward, (x,), self.parameters(), self.cfg.use_checkpoint
        )

    def _forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        h = self.out_layers(h)
        return self.skip_connection(x) + h

    @property
    def d_out(self) -> int:
        return self.cfg.out_channels or 0


@dataclass
class AttentionBlockConfig:
    channels: int = 3
    num_heads: int = 1
    num_head_channels: int = -1
    use_checkpoint: bool = False
    use_new_attention_order: bool = False
    num_groups: int = 32


class AttentionBlock(Backbone[AttentionBlockConfig]):
    def __init__(self, cfg: AttentionBlockConfig):
        super().__init__(cfg)
        if cfg.num_head_channels == -1:
            self.num_heads = cfg.num_heads
        else:
            assert (
                cfg.channels % cfg.num_head_channels == 0
            ), f"q,k,v channels {cfg.channels} is not divisible by num_head_channels {cfg.num_head_channels}"
            self.num_heads = cfg.channels // cfg.num_head_channels

        self.norm = normalization(cfg.channels, cfg.num_groups)
        self.qkv = conv_nd(1, cfg.channels, cfg.channels * 3, 1)
        if cfg.use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, cfg.channels, cfg.channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class MullerResizer(nn.Module):
    def __init__(
        self,
        d_in=3,
        base_resize_method="bilinear",
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
                weight = nn.Parameter(torch.empty((), dtype=dtype))
                bias = nn.Parameter(torch.empty((), dtype=dtype))
                nn.init.uniform_(weight, a=-0.1, b=0.1)
                nn.init.zeros_(bias)
                self.weights.append(weight)
                self.biases.append(bias)

        self.gaussian_kernel = self.create_gaussian_kernel(kernel_size, stddev)

    def create_gaussian_kernel(self, kernel_size, stddev):
        t = torch.arange(kernel_size, dtype=self.dtype) - (kernel_size - 1) / 2
        gaussian_kernel = torch.exp(-t.pow(2) / (2 * stddev**2))
        gaussian_kernel /= gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(
            1, 1, kernel_size, 1
        ) * gaussian_kernel.view(1, 1, 1, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.d_in, 1, 1, 1)
        return gaussian_kernel

    def _apply_gaussian_blur(self, x):
        padding = self.kernel_size // 2
        x = F.pad(x, (padding, padding, padding, padding), mode="reflect")
        # Move the kernel to the same device as the input tensor
        gaussian_kernel = self.gaussian_kernel.to(x.device)
        return F.conv2d(x, gaussian_kernel, groups=self.d_in)

    def forward(self, x, target_size):
        x = x.to(dtype=self.dtype)
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


@dataclass
class EncoderConfig:
    model_channels: int
    num_res_blocks: int
    attention_resolutions: Optional[list] = None
    dropout: float = 0
    channel_mult: tuple = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    use_checkpoint: bool = False
    use_fp16: bool = False
    num_heads: int = 4
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    pool: str = "adaptive"
    num_groups: int = 32
    resample_2d: bool = True
    scale_factor: int = 2
    resizer_num_layers: int = 2
    resizer_avg_pool: bool = False
    activation: str = "GELU"


class Encoder(Backbone[EncoderConfig]):
    def __init__(self, cfg: EncoderConfig, phw: int, d_in: int, d_out: int):
        super().__init__(cfg)
        self.d_in = d_in
        self.latent: int = d_out
        self.phw = phw

        self.dtype = torch.float16 if cfg.use_fp16 else torch.float32

        if hasattr(nn, cfg.activation):
            self.activation = getattr(nn, cfg.activation)()

        self.input_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    conv_nd(cfg.dims, self.d_in, cfg.model_channels, 3, padding=1)
                )
            ]
        )

        self._feature_size = cfg.model_channels
        input_block_chans = [cfg.model_channels]
        ch = cfg.model_channels
        ds = 1

        for level, mult in enumerate(cfg.channel_mult):
            for _ in range(cfg.num_res_blocks):
                layers: list[Any] = [
                    ResBlock(
                        ResBlockConfig(
                            channels=ch,
                            dropout=cfg.dropout,
                            out_channels=mult * cfg.model_channels,
                            dims=cfg.dims,
                            use_checkpoint=cfg.use_checkpoint,
                            num_groups=cfg.num_groups,
                            resample_2d=cfg.resample_2d,
                        )
                    )
                ]
                ch = mult * cfg.model_channels
                if (
                    cfg.attention_resolutions is not None
                    and ds in cfg.attention_resolutions
                ):
                    layers.append(
                        AttentionBlock(
                            AttentionBlockConfig(
                                channels=ch,
                                use_checkpoint=cfg.use_checkpoint,
                                num_heads=cfg.num_heads,
                                num_head_channels=cfg.num_head_channels,
                                use_new_attention_order=cfg.use_new_attention_order,
                                num_groups=cfg.num_groups,
                            )
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(cfg.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlock(
                            ResBlockConfig(
                                channels=ch,
                                dropout=cfg.dropout,
                                out_channels=out_ch,
                                dims=cfg.dims,
                                use_checkpoint=cfg.use_checkpoint,
                                down=True,
                                num_groups=cfg.num_groups,
                                resample_2d=cfg.resample_2d,
                            )
                        )
                        if cfg.resblock_updown
                        else Downsample(
                            ch, cfg.conv_resample, dims=cfg.dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = nn.Sequential(
            ResBlock(
                ResBlockConfig(
                    channels=ch,
                    dropout=cfg.dropout,
                    dims=cfg.dims,
                    use_checkpoint=cfg.use_checkpoint,
                    num_groups=cfg.num_groups,
                    resample_2d=cfg.resample_2d,
                ),
                self.activation,
            ),
            AttentionBlock(
                AttentionBlockConfig(
                    channels=ch,
                    use_checkpoint=cfg.use_checkpoint,
                    num_heads=cfg.num_heads,
                    num_head_channels=cfg.num_head_channels,
                    use_new_attention_order=cfg.use_new_attention_order,
                    num_groups=cfg.num_groups,
                )
            ),
            ResBlock(
                ResBlockConfig(
                    channels=ch,
                    dropout=cfg.dropout,
                    dims=cfg.dims,
                    use_checkpoint=cfg.use_checkpoint,
                    num_groups=cfg.num_groups,
                    resample_2d=cfg.resample_2d,
                ),
                self.activation,
            ),
        )
        self._feature_size += ch
        self.pool = cfg.pool

        spatial_dims = (2, 3, 4, 5)[: cfg.dims]
        self.gap = lambda x: x.mean(dim=spatial_dims)

        if cfg.pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch, cfg.num_groups),
                self.activation,
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(cfg.dims, ch, self.d_in * self.latent, 1)),
                nn.Flatten(),
            )
        elif cfg.pool == "attention":
            assert cfg.num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch, cfg.num_groups),
                self.activation,
                AttentionPool2d(
                    int(((self.phw * (cfg.scale_factor / 2)) ** 2) // (ds * 2)),
                    ch,
                    cfg.num_head_channels,
                    int(self.d_in * self.latent),
                ),
            )
        elif cfg.pool == "spatial":
            self.out = nn.Linear(256, self.d_in * self.latent)
        elif cfg.pool == "spatial_v2":
            print(f"feture size: {self._feature_size}")
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048, cfg.num_groups),
                self.activation,
                nn.Linear(2048, self.d_in * self.latent),
            )
        else:
            raise NotImplementedError(f"Unexpected {cfg.pool} pooling")

        self.resizer = MullerResizer(
            self.d_in,
            "bilinear",
            kernel_size=5,
            stddev=1.0,
            num_layers=cfg.resizer_num_layers,
            dtype=self.dtype,
            avg_pool=cfg.resizer_avg_pool,
        )

    def _interpolate(self, x, scale_factor):
        B, C, H, W = x.size()
        target_h = int(H * scale_factor)
        target_w = int(W * scale_factor)
        target_size = (target_h, target_w)
        x_resized = self.resizer(x, target_size)
        return x_resized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._interpolate(x, self.cfg.scale_factor)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)

        h = self.middle_block(h)

        if self.pool.startswith("spatial"):
            h = self.gap(h)
            N = h.shape[0]
            h = h.reshape(N, -1)

            gaussians = self.out(h)
            gaussians = rearrange(
                gaussians, "b (c latent) -> b c latent", c=self.d_in, latent=self.latent
            )

            return gaussians
        else:
            h = h.type(x.dtype)

            gaussians = self.out(h)
            gaussians = rearrange(
                gaussians, "b (c latent) -> b c latent", c=self.d_in, latent=self.latent
            )

            return gaussians

    @property
    def d_out(self) -> int:
        return self.latent

    @property
    def scale_factor(self) -> int:
        return self.cfg.scale_factor


@dataclass
class MoEConfig:
    kernel: int = 4
    sharpening_factor: float = 1.0


@dataclass
class Gaussians:
    mu: torch.Tensor
    sigma: torch.Tensor
    w: torch.Tensor


class MoE(Backbone[MoEConfig]):
    def __init__(self, cfg: MoEConfig):
        super(MoE, self).__init__(cfg)
        self.kernel = cfg.kernel
        self.alpha = cfg.sharpening_factor

    def grid(self, height, width):
        xx = torch.linspace(0.0, 1.0, width)
        yy = torch.linspace(0.0, 1.0, height)
        grid_x, grid_y = torch.meshgrid(xx, yy, indexing="ij")
        grid = torch.stack((grid_x, grid_y), -1).float()
        return grid

    def cov_mat_2d(self, scale: torch.Tensor, theta: torch.Tensor, epsilon=1e-4):
        R = self.ang_to_rot_mat(theta)
        S = torch.diag_embed(scale) + epsilon * torch.eye(
            scale.size(-1), device=scale.device
        )

        RS = R @ S
        Sigma = RS @ RS.transpose(-2, -1)
        return Sigma

    def ang_to_rot_mat(self, theta: torch.Tensor):
        cos_theta = torch.cos(theta).unsqueeze(-1)
        sin_theta = torch.sin(theta).unsqueeze(-1)
        R = torch.cat([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1)
        return R.view(*theta.shape, 2, 2)

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Gaussians:
        mu_x = p[:, :, :k].reshape(-1, ch, k, 1)
        mu_y = p[:, :, k : 2 * k].reshape(-1, ch, k, 1)
        mu = torch.cat((mu_x, mu_y), dim=-1).view(-1, ch, k, 2)

        scale_idx = 3 * k
        scale = p[:, :, scale_idx : scale_idx + 2 * k].reshape(-1, p.shape[1], k, 2)
        rot_idx = scale_idx + 2 * k
        theta = p[:, :, rot_idx : rot_idx + k].reshape(-1, p.shape[1], k)

        cov_matrix = self.cov_mat_2d(scale, theta)
        cov_matrix = torch.mul(cov_matrix, self.alpha)

        w = p[:, :, 2 * k : 3 * k].reshape(-1, ch, k)

        return Gaussians(mu, cov_matrix, w)

    def forward(self, height, width, params):
        batch_size, num_channels, _ = params.shape
        gauss = self.extract_parameters(params, self.kernel, num_channels)

        grid = self.grid(height, width).to(params.device)
        grid_expanded = (
            grid.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, num_channels, self.kernel, height, width, 2)
        )

        mu_expanded = (
            gauss.mu.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, height, width, -1)
        )
        x_sub_mu = grid_expanded - mu_expanded

        Sigma_expanded = (
            gauss.sigma.unsqueeze(3)
            .unsqueeze(4)
            .expand(-1, -1, -1, height, width, -1, -1)
        )

        x_sub_mu_t = x_sub_mu.unsqueeze(-1)

        exp_terms = -0.5 * torch.einsum(
            "bnkhwli,bnkhwlj,bnkhwmi->bnkhw",
            x_sub_mu_t,
            Sigma_expanded,
            x_sub_mu_t,
        ).squeeze(-1)

        e = torch.exp(exp_terms)

        max_e = torch.max(e, dim=2, keepdim=True)[0]
        log_sum_exp_e = max_e + torch.log(
            torch.sum(torch.exp(e - max_e), dim=2, keepdim=True)
        )
        e_norm = torch.exp(e - log_sum_exp_e)

        y_hat = torch.sum(e_norm * gauss.w.unsqueeze(-1).unsqueeze(-1), dim=2)
        y_hat = torch.clamp(y_hat, min=0, max=1)

        return y_hat


@dataclass
class AutoencoderConfig:
    EncoderConfig: EncoderConfig
    DecoderConfig: MoEConfig
    d_in: int
    d_out: int
    phw: int = 32
    overlap: int = 24


class Autoencoder(Backbone[AutoencoderConfig]):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__(cfg)
        self.phw = cfg.phw
        self.overlap = cfg.overlap
        self.encoder = Encoder(
            cfg.EncoderConfig, cfg.phw, d_in=cfg.d_in, d_out=cfg.d_out
        )
        self.decoder = MoE(cfg.DecoderConfig)

    @staticmethod
    def reconstruct(
        blocks: torch.Tensor,
        original_dims: Tuple[int, int, int, int],
        block_size: int,
        overlap: int,
    ) -> torch.Tensor:
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
            current_blocks = blocks[idx_start : idx_start + num_blocks_per_image]
            idx = 0
            for i in range(0, height - block_size + 1, step):
                for j in range(0, width - block_size + 1, step):
                    recon_images[
                        b, :, i : i + block_size, j : j + block_size
                    ] += current_blocks[idx]
                    count_matrix[b, :, i : i + block_size, j : j + block_size] += 1
                    idx += 1

        recon_images /= count_matrix.clamp(min=1)
        return recon_images

    @staticmethod
    def mem_lim():
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":
            device = torch.cuda.current_device()
            torch.cuda.set_device(device)
            tot_mem = torch.cuda.get_device_properties(device).total_memory
            used_mem = torch.cuda.memory_reserved(device)
            free_mem = tot_mem - used_mem

            thresholds = [0.7, 0.5, 0.3, 0.1]
            for percent in thresholds:
                threshold = tot_mem * percent
                if free_mem > threshold:
                    return threshold

            return max(1 * 2**30, tot_mem * 0.05)
        else:
            return 1 * 2**30

    def forward(self, x: torch.Tensor, s: Tuple[int, int, int, int]) -> torch.Tensor:
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

        dec = torch.cat(
            [self.decoder(sp, sp, bt) for bt in torch.split(enc, cs)], dim=0
        )

        y = self.reconstruct(
            dec,
            (B, C, H * self.encoder.scale_factor, W * self.encoder.scale_factor),
            sp,
            self.overlap * self.encoder.scale_factor,
        )
        return y
