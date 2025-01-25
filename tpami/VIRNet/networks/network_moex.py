# # type: ignore

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
from .nn import GroupNorm32, avg_pool_nd, checkpoint, conv_nd, zero_module
from torch import compile
from enum import Enum
from .DnCNN import DnCNN
from .KNet import KernelNet
from utils import util_net


torch.set_float32_matmul_precision("high")

backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

OLD_GPU = True
USE_FLASH_ATTN = False
MATH_KERNEL_ON = True

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


class Upsample(nn.Module):
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

    def forward(self, input_x) -> torch.Tensor:
        x = input_x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x: torch.Tensor = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        pos_emb = self.positional_embedding[: x.shape[0], :].to(x.dtype)
        x = x + pos_emb[:, None, :]
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


def normalization(num_channels: int, num_groups: int) -> GroupNorm32:
    return GroupNorm32(num_groups, num_channels)


def count_flops_attn(model, _x, y) -> None:
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


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
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


def init_t_xy(q_len: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    end_x = int(math.floor(math.sqrt(q_len)))
    end_y: int = math.ceil(q_len / end_x)
    t: torch.Tensor = torch.arange(end_x * end_y, dtype=torch.float32)[:q_len]
    t_x: torch.Tensor = (t % end_x).float()
    t_y: torch.Tensor = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y, end_x, end_y


def compute_axial_cis(dim: int, q_len: int, theta: float = 10000.0) -> torch.Tensor:
    t_x, t_y, end_x, end_y = init_t_xy(q_len)
    freqs_x: torch.Tensor = 1.0 / (
        theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)
    )
    freqs_y: torch.Tensor = 1.0 / (
        theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)
    )
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x: torch.Tensor = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y: torch.Tensor = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_: torch.Tensor = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_: torch.Tensor | None = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )

    if freqs_cis.shape[0] != xq_.shape[-2]:
        freqs_cis = freqs_cis[: xq_.shape[-2]]

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out: torch.Tensor = torch.view_as_real(xq_ * freqs_cis).flatten(3)

    if xk_ is None:
        return xq_out.type_as(xq).to(xq.device), xk

    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)

    xk_out: torch.Tensor = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape: list[int] = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: Any = None,
    ) -> None:
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.kv_in_dim: Any | int = (
            kv_in_dim if kv_in_dim is not None else embedding_dim
        )
        self.internal_dim: int = embedding_dim // downsample_rate
        self.num_heads: int = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p: float = dropout

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p: float = self.dropout_p if self.training else 0.0

        with sdpa_kernel(backends):
            out: torch.Tensor = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p
            )

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention(Attention):
    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(32, 32),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        self.freqs_cis = None
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        batch_size, channels, seq_len = q.shape
        q = q.reshape(batch_size * seq_len, -1)
        k = k.reshape(batch_size * seq_len, -1)
        v = v.reshape(batch_size * seq_len, -1)

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.reshape(batch_size, seq_len, self.internal_dim)
        k = k.reshape(batch_size, seq_len, self.internal_dim)
        v = v.reshape(batch_size, seq_len, self.internal_dim)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        q_len, k_len = q.shape[-2], k.shape[-2]

        self.freqs_cis: torch.Tensor = self.compute_cis(q_len=q_len).to(q.device)

        num_k_rope: int = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p: float = self.dropout_p if self.training else 0.0

        with sdpa_kernel(backends):
            out: torch.Tensor = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p
            )

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


@dataclass
class ResBlockConfig:
    channels: int = 3
    dropout: float = 0.0
    out_channels: int = 0
    use_conv: bool = False
    dims: int = 2
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False
    num_groups: int = 32
    resample_2d: bool = True

    def __post_init__(self) -> None:
        self.out_channels = self.out_channels or self.channels


class ResBlock(Backbone[ResBlockConfig]):
    def __init__(self, cfg: ResBlockConfig, activation: nn.Module = nn.GELU()):
        super().__init__(cfg=cfg)

        self.activation: nn.Module = activation

        self.in_layers = nn.Sequential(
            normalization(num_channels=cfg.channels, num_groups=cfg.num_groups),
            self.activation,
            conv_nd(cfg.dims, cfg.channels, cfg.out_channels, 3, padding=1),
        )

        self.updown: bool = cfg.up or cfg.down

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
            normalization(num_channels=cfg.out_channels, num_groups=cfg.num_groups),
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
            self._forward, (x,), list(self.parameters()), self.cfg.use_checkpoint
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
    num_heads: int = 4
    num_groups: int = 32
    num_head_channels: int = -1
    rope_theta: float = 10000.0
    dropout_rate: float = 0.1
    use_checkpoint: bool = True
    dims: int = 1
    attention_type: str = "cross_attention"  # "attention" or "cross_attention"


class AttentionBlock(Backbone[AttentionBlockConfig]):
    def __init__(self, cfg: AttentionBlockConfig, phw: int, scale_factor: int = 1):
        super().__init__(cfg)
        self.cfg: AttentionBlockConfig = cfg
        self.scale_factor: int = scale_factor
        self.phw: int = phw

        self.norm: GroupNorm32 = normalization(
            num_channels=cfg.channels, num_groups=cfg.num_groups
        )
        self.qkv: nn.Conv1d | nn.Conv2d | nn.Conv3d = conv_nd(
            1, cfg.channels, cfg.channels * 3, 1
        )

        self.attention_type: str = (
            cfg.attention_type if hasattr(cfg, "attention_type") else "attention"
        )

        self.attention_map = {
            "attention": QKVAttention(cfg.num_heads),
            "cross_attention": RoPEAttention(
                embedding_dim=cfg.channels,
                num_heads=cfg.num_heads,
                rope_theta=cfg.rope_theta,
                rope_k_repeat=True,
                feat_sizes=(phw, phw),
            ),
        }

        self.attention = self.attention_map.get(self.attention_type, None)

        self.proj_out: nn.Conv1d | nn.Conv2d | nn.Conv3d = zero_module(
            conv_nd(cfg.dims, cfg.channels, cfg.channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(
            self._forward, (x,), list(self.parameters()), self.cfg.use_checkpoint
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))

        if self.attention_type == "attention" and self.attention is not None:
            h = self.attention(qkv)
        else:
            h = self.attention(*qkv.chunk(3, dim=1))
            h = h.transpose(1, 2)

        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


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


@dataclass
class EncoderConfig:
    noise_cond: bool
    kernel_cond: bool
    noise_avg: bool
    sigma_chn: int
    kernel_chn: int
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
    resblock_updown: bool = False
    num_groups: int = 32
    resample_2d: bool = True
    scale_factor: int = 2
    resizer_num_layers: int = 2
    resizer_avg_pool: bool = False
    activation: str = "GELU"
    rope_theta: float = 10000.0
    attention_type: str = "cross_attention"  # "attention" or "cross_attention"


class Encoder(Backbone[EncoderConfig]):
    def __init__(self, cfg: EncoderConfig, phw: int, d_in: int, d_out: int) -> None:
        super().__init__(cfg)
        self.d_in: int = d_in
        self.latent: int = d_out
        self.phw: int = phw

        self.dtype: torch.dtype = torch.float16 if cfg.use_fp16 else torch.float32

        if hasattr(nn, cfg.activation):
            self.activation = getattr(nn, cfg.activation)()

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

        self.depth = len(cfg.channel_mult)

        self.input_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    conv_nd(
                        cfg.dims, d_in + extra_chn, cfg.model_channels, 3, padding=1
                    )
                )
            ]
        )

        self._feature_size: int = cfg.model_channels
        input_block_chans: list[int] = [cfg.model_channels]
        ch: int = cfg.model_channels
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
                        ),
                        activation=self.activation,
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
                                num_groups=cfg.num_groups,
                                rope_theta=cfg.rope_theta,
                                dropout_rate=cfg.dropout,
                                attention_type=cfg.attention_type,
                            ),
                            phw,
                            scale_factor=cfg.scale_factor,
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
                            ),
                            activation=self.activation,
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
                    num_groups=cfg.num_groups,
                    rope_theta=cfg.rope_theta,
                    dropout_rate=cfg.dropout,
                    attention_type=cfg.attention_type,
                ),
                phw,
                scale_factor=cfg.scale_factor,
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

        self.out = nn.Sequential(
            normalization(num_channels=ch, num_groups=cfg.num_groups),
            self.activation,
            AttentionPool2d(
                int(((self.phw * (cfg.scale_factor / 2)) ** 2)),
                # int(((self.phw * (cfg.scale_factor / 2)) ** 2) // (ds)),
                ch,
                cfg.num_heads,
                int(self.d_in * self.latent),
            ),
        )

        self.resizer = MullerResizer(
            self.d_in,
            "bicubic",
            kernel_size=5,
            stddev=1.0,
            num_layers=cfg.resizer_num_layers,
            dtype=self.dtype,
            avg_pool=cfg.resizer_avg_pool,
        )

    def _interpolate(self, x, scale_factor):
        _, _, H, W = x.size()
        target_h = int(H * scale_factor)
        target_w = int(W * scale_factor)
        target_size: Tuple[int, int] = (target_h, target_w)
        x_resized = self.resizer(x, target_size)
        return x_resized

    def pad_x(self, x: torch.Tensor) -> torch.Tensor:
        return util_net.pad_input(x, 2 ** (self.depth - 1))

    def forward(
        self,
        x: torch.Tensor,
        sigma_est: Optional[torch.Tensor] = None,
        kinfo_est: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        sigma = torch.exp(torch.clamp(sigma_est, min=log_min, max=log_max))

        x_pad = self.pad_x(x)
        x_up = self._interpolate(x_pad, self.scale_factor)
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

        h = x_input.to(dtype=self.dtype)
        for module in self.input_blocks:
            h = module(h)
        h = self.middle_block(h)
        h = h.to(dtype=x_input.dtype)

        gaussians = self.out(h)
        gaussians = rearrange(
            gaussians, "b (c latent) -> b c latent", c=self.d_in, latent=self.latent
        )
        return gaussians, kinfo_est.squeeze(-1).squeeze(-1), sigma

    @property
    def d_out(self) -> int:
        return self.latent

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


class MoE(Backbone[MoEConfig]):
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

    def cov_mat(
        self,
        scale: torch.Tensor,
        theta_xy: torch.Tensor,
        scale_color: torch.Tensor,
        rho_color: torch.Tensor,
        ch: int,
    ) -> torch.Tensor:
        R = self.ang_to_rot_mat(theta_xy)
        S = torch.diag_embed(scale)
        C_xy = R @ S @ S.transpose(-2, -1) @ R.transpose(-2, -1)
        if ch == 1:
            C_color = scale_color.squeeze(-1).squeeze(-1)
            C_full = torch.zeros(*C_xy.shape[:-2], 3, 3, device=scale.device)
            C_full[..., :2, :2] = C_xy
            C_full[..., 2, 2] = C_color
        elif ch == 3:
            rho = rho_color.unsqueeze(-1)
            C_rgb = torch.diag_embed(scale_color) + rho @ rho.transpose(-2, -1)
            C_full = torch.zeros(*C_xy.shape[:-2], 5, 5, device=scale.device)
            C_full[..., :2, :2] = C_xy
            C_full[..., 2:, 2:] = C_rgb
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")

        return C_full

    def extract_parameters(
        self, p: torch.Tensor, k: int, ch: int
    ) -> Tuple[torch.Tensor, torch.Tensor, ...]:
        B, _, _ = p.shape
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            mu_x = p[:, :, 0:k].reshape(B, ch, k, 1)
            mu_y = p[:, :, k : 2 * k].reshape(B, ch, k, 1)
            scale_xy = F.softplus(p[:, :, 2 * k : 4 * k].reshape(B, ch, k, 2))
            theta_xy = (p[:, :, 4 * k : 5 * k].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            w = F.softmax(p[:, :, 5 * k : 6 * k].reshape(B, ch, k), dim=-1)
            alpha = torch.sigmoid(p[:, :, 6 * k : 7 * k].reshape(B, ch, k))
            c = F.softplus(p[:, :, 7 * k : 8 * k].reshape(B, ch, k))
            if ch == 1:
                scale_color = F.softplus(p[:, :, 8 * k : 9 * k].reshape(B, ch, k, 1))
            elif ch == 3:
                scale_color = F.softplus(p[:, :, 8 * k : 11 * k].reshape(B, ch, k, 3))
            rho_color = torch.tanh(p[:, :, 11 * k : 12 * k].reshape(B, ch, k, 1))
        else:  # KernelType.GAUSSIAN
            mu_x = p[:, :, 0:k].reshape(B, ch, k, 1)
            mu_y = p[:, :, k : 2 * k].reshape(B, ch, k, 1)
            scale_xy = F.softplus(p[:, :, 2 * k : 4 * k].reshape(B, ch, k, 2))
            theta_xy = (p[:, :, 4 * k : 5 * k].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            w = F.softmax(p[:, :, 5 * k : 6 * k].reshape(B, ch, k), dim=-1)
            if ch == 1:
                scale_color = F.softplus(p[:, :, 6 * k : 7 * k].reshape(B, ch, k, 1))
            elif ch == 3:
                scale_color = F.softplus(p[:, :, 6 * k : 9 * k].reshape(B, ch, k, 3))
            rho_color = torch.tanh(p[:, :, 9 * k : 10 * k].reshape(B, ch, k, 1))
            alpha = None
            c = None

        mu = torch.cat([mu_x, mu_y], dim=-1)
        cov_matrix = (
            self.cov_mat(scale_xy, theta_xy, scale_color, rho_color, ch)
            * self.sharpening_factor
        )
        return mu, cov_matrix, w, alpha, c

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
        eye_d = torch.eye(d, device=cov_matrix.device).view(1, 1, 1, d, d)
        Sigma_inv = torch.linalg.solve(cov_matrix + eps * eye_d, eye_d)
        device = params.device

        g = self.grid(height, width, device)  # (height, width, 2)
        g_color = torch.zeros(
            height, width, ch, device=device
        )  # shape => (height, width, 2 + ch)
        g_full = (
            torch.cat([g, g_color], dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )  # => (B, ch, k, height, width, 2+ch) after broadcast

        mu_color = torch.zeros(B, ch, k, ch, device=device)
        mu_full = (
            torch.cat([mu, mu_color], dim=-1).unsqueeze(3).unsqueeze(4)
        )  # => (B, ch, k, 1, 1, 2+ch)
        S = Sigma_inv.unsqueeze(3).unsqueeze(4)  # => (B, ch, k, 1, 1, d, d)

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
            d_out = (7 * cfg.d_in) * cfg.DecoderConfig.kernel
        else:
            d_out = (7 * cfg.d_in + 3) * cfg.DecoderConfig.kernel

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

        self.encoder = Encoder(cfg.EncoderConfig, cfg.phw, d_in=cfg.d_in, d_out=d_out)
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
