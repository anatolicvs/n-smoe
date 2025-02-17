# # type: ignore

import math
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Generic, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils import spectral_norm

from .DnCNN import DnCNN
from .KNet import KernelNet
from .nn import GroupNorm32, avg_pool_nd, checkpoint, conv_nd, zero_module


backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

T = TypeVar("T")
log_min = math.log(1e-6)
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


class AttentionPool2d_(nn.Module):
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


class FourierPosition(nn.Module):
    def __init__(self, in_dim: int = 2, mapping_size: int = 128, scale: float = 1.0):
        super().__init__()
        assert mapping_size % 2 == 0, "mapping_size must be even"
        self.mapping_size = mapping_size
        self.B = nn.Parameter(torch.randn(in_dim, mapping_size // 2) * scale)
        self.norm = nn.LayerNorm(mapping_size)  # Normalize Fourier features

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * torch.pi * coords @ self.B
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.norm(features)


class AttentionPool2d(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None,
        fourier_size: int = 128,
        fourier_scale: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim or embed_dim
        self.fourier_pos = FourierPosition(
            in_dim=2, mapping_size=fourier_size, scale=fourier_scale
        )
        self.linear_pos = nn.Linear(fourier_size, embed_dim)
        self.global_pos = nn.Parameter(torch.empty(embed_dim))
        nn.init.xavier_uniform_(self.global_pos.unsqueeze(0))
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, self.output_dim)

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = input_x.shape
        if C != self.embed_dim:
            raise ValueError(
                f"Input channel count {C} must equal embed_dim ({self.embed_dim})"
            )
        x = input_x.flatten(start_dim=2).permute(2, 0, 1)
        global_token = x.mean(dim=0, keepdim=True)
        x = torch.cat([global_token, x], dim=0)
        grid_y = torch.linspace(0, 1, H, device=input_x.device)
        grid_x = torch.linspace(0, 1, W, device=input_x.device)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).view(-1, 2)
        fourier_features = self.fourier_pos(coords)
        pos_spatial = self.linear_pos(fourier_features)
        pos_global = self.global_pos.unsqueeze(0)
        pos_emb = torch.cat([pos_global, pos_spatial], dim=0)
        x = x + pos_emb.unsqueeze(1)
        attn_output, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=None,
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
        return attn_output.squeeze(0)


def normalization(C, G_max=32, C_min=8, eps=1e-5, affine=True):
    G = min(G_max, C // C_min)
    G = max(G, 1)
    return nn.GroupNorm(G, C, eps=eps, affine=affine)


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

        self.scale_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)

        # scale = 1 / math.sqrt(math.sqrt(ch))
        # scale = 1 / math.sqrt(ch)
        scale = self.scale_param / math.sqrt(ch)

        # raw_scores = torch.einsum(
        #     "bct,bcs->bts",
        #     (q * scale).view(bs * self.n_heads, ch, length),
        #     (k * scale).view(bs * self.n_heads, ch, length),
        # )

        # norm_scores = spectral_normalize_tensor(raw_scores)
        # weight = torch.softmax(norm_scores.float(), dim=-1).type(raw_scores.dtype)
        # a = torch.einsum(
        #     "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        # )

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
            conv_nd(cfg.dims, cfg.channels, cfg.out_channels, 3, padding=1),
            normalization(C=cfg.out_channels, G_max=cfg.num_groups),
            self.activation,
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

        conv_final = conv_nd(cfg.dims, cfg.out_channels, cfg.out_channels, 3, padding=1)
        nn.init.kaiming_normal_(conv_final.weight, mode="fan_in", nonlinearity="linear")
        conv_final.weight.data.mul_(0.1)  # Scale down the weights by 0.1.
        nn.init.zeros_(conv_final.bias)

        self.out_layers = nn.Sequential(
            normalization(C=cfg.out_channels, G_max=cfg.num_groups),
            self.activation,
            nn.Dropout(p=cfg.dropout),
            # zero_module(
            #     conv_nd(cfg.dims, cfg.out_channels, cfg.out_channels, 3, padding=1)
            # ),
            conv_final,
        )

        if cfg.out_channels == cfg.channels:
            self.skip_connection = nn.Identity()
        elif cfg.use_conv:
            self.skip_connection = conv_nd(
                cfg.dims, cfg.channels, cfg.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(cfg.dims, cfg.channels, cfg.out_channels, 1)

        self.res_scale = nn.Parameter(torch.tensor(1.0))

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
        # return self.skip_connection(x) + h
        return self.skip_connection(x) + self.res_scale * h

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

        self.norm: GroupNorm32 = normalization(C=cfg.channels, G_max=cfg.num_groups)
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
        self.stddev = nn.Parameter(torch.tensor(stddev, dtype=dtype))
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

    model_channels: int = 64  # Number of channels in the first layer
    channel_mult: tuple = (1, 2, 4, 8)  # E.g., [64, 128, 256, 512]
    num_res_blocks: int = 2
    attention_resolutions: Optional[list] = None
    dropout: float = 0
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

        self.kernel_min = getattr(cfg, "kernel_min", 8)
        self.kernel_max = getattr(cfg, "kernel_max", 16)

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
            current_checkpoint = level >= len(cfg.channel_mult) - 2
            for _ in range(cfg.num_res_blocks):
                layers: list[Any] = [
                    ResBlock(
                        ResBlockConfig(
                            channels=ch,
                            dropout=cfg.dropout,
                            out_channels=mult * cfg.model_channels,
                            dims=cfg.dims,
                            use_checkpoint=current_checkpoint,
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
                                use_checkpoint=current_checkpoint,
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
                                use_checkpoint=current_checkpoint,
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

        # self.kernel_estimator = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling: [B, ch, 1, 1]
        #     nn.Flatten(),  # -> [B, ch_est]
        #     nn.Linear(ch, 32),
        #     self.activation,
        #     nn.Linear(32, 1),
        #     nn.Sigmoid(),
        # )

        # self.out = nn.Sequential(
        #     normalization(num_channels=ch, num_groups=cfg.num_groups),
        #     self.activation,
        #     AttentionPool2d(
        #         int(((self.phw * (cfg.scale_factor / 2)) ** 2)),
        #         # int(((self.phw * (cfg.scale_factor / 2)) ** 2) // (ds)),
        #         ch,
        #         cfg.num_heads,
        #         int(self.d_in * self.latent),
        #     ),
        # )

        self.out = nn.Sequential(
            normalization(C=ch, G_max=cfg.num_groups),
            self.activation,
            AttentionPool2d(
                embed_dim=ch,
                num_heads=cfg.num_heads,
                output_dim=int(self.d_in * self.latent),
            ),
        )

        self.global_gate = nn.Conv2d(ch, ch, kernel_size=1)

        # self.resizer = MullerResizer(
        #     self.d_in,
        #     "bicubic",
        #     kernel_size=5,
        #     stddev=1.0,
        #     num_layers=cfg.resizer_num_layers,
        #     dtype=self.dtype,
        #     avg_pool=cfg.resizer_avg_pool,
        # )

    # def _interpolate(self, x, scale_factor):
    #     _, _, H, W = x.size()
    #     target_h = int(H * scale_factor)
    #     target_w = int(W * scale_factor)
    #     target_size: Tuple[int, int] = (target_h, target_w)
    #     x_resized = self.resizer(x, target_size)
    #     return x_resized

    # def pad_x(self, x: torch.Tensor) -> torch.Tensor:
    #     return util_net.pad_input(x, 2 ** (self.depth - 1))

    def forward(
        self,
        x: torch.Tensor,
        sigma_est: Optional[torch.Tensor] = None,
        kinfo_est: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor] == None
    ]:

        # sigma = torch.exp(torch.clamp(sigma_est, min=log_min, max=log_max))
        sigma = F.softplus(sigma_est) + 1e-8

        # x_pad = self.pad_x(x)
        # x_up = self._interpolate(x, self.scale_factor)

        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic")

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

        global_ctx = F.adaptive_avg_pool2d(h, (1, 1))
        gate = torch.sigmoid(self.global_gate(global_ctx))
        h = h * gate

        h = h.to(dtype=x_input.dtype)

        # pool_h = F.adaptive_avg_pool2d(h, (1, 1))
        # kernel_ratio = self.kernel_estimator(pool_h)  # Shape: [B, 1]
        # Scale to [kernel_min, kernel_max] and round to integer.
        # kernel_count = (
        #     torch.round(
        #         self.kernel_min + (self.kernel_max - self.kernel_min) * kernel_ratio
        #     )
        #     .long()
        #     .squeeze(1)
        # )  # Shape: [B]

        gaussians = self.out(h)
        gaussians = rearrange(
            gaussians, "b (c latent) -> b c latent", c=self.d_in, latent=self.latent
        )
        # return gaussians, kinfo_est.squeeze(-1).squeeze(-1), sigma, kernel_count # dynamic kernel count
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
    activation: str = "GELU"
    min_diag: float = 1e-5
    min_denom: float = 1e-5
    initial_temp: float = 0.5
    tau_min: float = 0.1
    reg_lambda: float = 1e-5
    grid_cache: Optional[torch.Tensor] = None  # to cache grid


@dataclass
class Gaussians:
    mu: Optional[torch.Tensor] = None
    sigma: Optional[torch.Tensor] = None
    w: Optional[torch.Tensor] = None
    theta: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None


class MoE(Backbone[MoEConfig]):
    def __init__(self, cfg: MoEConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.kernel = cfg.kernel
        self.sharpening_factor = cfg.sharpening_factor
        self.kernel_type = cfg.kernel_type
        self.min_diag = cfg.min_diag
        self.min_denom = cfg.min_denom
        self.tau_min = cfg.tau_min
        self.reg_lambda = nn.Parameter(torch.tensor(cfg.reg_lambda), requires_grad=True)

        self.log_temp = nn.Parameter(
            torch.log(torch.tensor(cfg.initial_temp)), requires_grad=True
        )

        self.spatial_mapper = spectral_norm(nn.Linear(3, 3))
        self.color_mapper_1 = spectral_norm(nn.Linear(1, 1))
        self.color_mapper_3 = spectral_norm(nn.Linear(6, 6))

        self.diag_size = 2

    def grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:

        if self.cfg.grid_cache is not None:
            return self.cfg.grid_cache.to(device)
        xx = torch.linspace(0.0, 1.0, width, device=device)
        yy = torch.linspace(0.0, 1.0, height, device=device)
        gx, gy = torch.meshgrid(xx, yy, indexing="ij")
        grid = torch.stack((gx, gy), dim=-1).float()
        return grid

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Convert an angle tensor to a 2x2 rotation matrix.
        For each angle θ, compute:
          R(θ) = [[cosθ, -sinθ],
                  [sinθ,  cosθ]]
        """
        ct = torch.cos(theta).unsqueeze(-1)
        st = torch.sin(theta).unsqueeze(-1)
        R = torch.cat([ct, -st, st, ct], dim=-1)
        return R.view(*theta.shape, 2, 2)

    def construct_lower_triangular(self, params: torch.Tensor, s: int) -> torch.Tensor:
        """
        Construct a lower-triangular matrix L from raw parameters.
        For s=2 (a 2x2 matrix), we expect params[..., :3] where:
          L_11 = softplus(p0) + 1e-2,
          L_21 = p1,
          L_22 = softplus(p2) + 1e-2.
        This construction guarantees L is lower-triangular and its diagonal is strictly positive.
        """
        if s == 2:
            B, ch, k, _ = params.shape
            l11 = F.softplus(params[..., 0]) + self.min_diag
            l21 = params[..., 1]
            l22 = F.softplus(params[..., 2]) + self.min_diag
            L = torch.zeros(B, ch, k, 2, 2, device=params.device, dtype=params.dtype)
            L[..., 0, 0] = l11.squeeze(-1)
            L[..., 1, 0] = l21.squeeze(-1)
            L[..., 1, 1] = l22.squeeze(-1)
            return L
        elif s == 1:
            B, ch, k, _ = params.shape
            L = torch.zeros(B, ch, k, 1, 1, device=params.device, dtype=params.dtype)
            L[..., 0, 0] = F.softplus(params[..., 0]) + self.min_diag
            return L
        elif s == 3:
            B, ch, k, _ = params.shape
            l11 = F.softplus(params[..., 0]) + self.min_diag
            l21 = params[..., 1]
            l22 = F.softplus(params[..., 2]) + self.min_diag
            l31 = params[..., 3]
            l32 = params[..., 4]
            l33 = F.softplus(params[..., 5]) + self.min_diag
            L = torch.zeros(B, ch, k, 3, 3, device=params.device, dtype=params.dtype)
            L[..., 0, 0] = l11.squeeze(-1)
            L[..., 1, 0] = l21.squeeze(-1)
            L[..., 1, 1] = l22.squeeze(-1)
            L[..., 2, 0] = l31.squeeze(-1)
            L[..., 2, 1] = l32.squeeze(-1)
            L[..., 2, 2] = l33.squeeze(-1)
            return L
        else:
            raise ValueError(f"Unsupported matrix size: {s}")

    def cov_mat(
        self,
        L_spatial: torch.Tensor,
        theta_xy: torch.Tensor,
        L_color: torch.Tensor,
        ch: int,
    ) -> torch.Tensor:
        """
        Compute the full covariance matrix.
        For spatial dimensions, compute:
          C_xy = R (L_spatial L_spatial^T) R^T,
        then symmetrize it.
        For the full covariance, embed C_xy into a larger matrix that includes the color channel(s).
        """
        R = self.ang_to_rot_mat(theta_xy)
        C_xy = torch.matmul(R, torch.matmul(L_spatial, L_spatial.transpose(-2, -1)))
        C_xy = torch.matmul(C_xy, R.transpose(-2, -1))
        C_xy = 0.5 * (C_xy + C_xy.transpose(-2, -1))
        if ch == 1:
            C_color = (
                torch.matmul(L_color, L_color.transpose(-2, -1)).squeeze(-1).squeeze(-1)
            )
            B_, _, k_ = C_xy.shape[:3]
            C_full = torch.zeros(B_, ch, k_, 3, 3, device=C_xy.device, dtype=C_xy.dtype)
            C_full[..., :2, :2] = C_xy
            C_full[..., 2, 2] = C_color
        elif ch == 3:
            C_color = torch.matmul(L_color, L_color.transpose(-2, -1))
            B_, _, k_ = C_xy.shape[:3]
            C_full = torch.zeros(B_, ch, k_, 5, 5, device=C_xy.device, dtype=C_xy.dtype)
            C_full[..., :2, :2] = C_xy
            C_full[..., 2:, 2:] = C_color
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")
        return C_full * self.sharpening_factor

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Extract the latent parameters from the raw tensor p.
        p is reshaped to [B, ch, k, param_per_kernel].
        For the Gaussian-Cauchy case, extract:
          - Spatial means (mu_x, mu_y)
          - Covariance parameters (which are mapped by the spatial_mapper and then converted into a lower–triangular matrix)
          - A rotation angle (theta_xy)
          - Expert logits for Gumbel-softmax to produce weights w (using an adaptive temperature)
          - Additional scalar parameters (alpha, c) for blending kernel functions.
        """
        B, _, _ = p.shape
        p = p.view(B, ch, k, -1)
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            mu_x = p[..., 0].reshape(B, ch, k, 1)
            mu_y = p[..., 1].reshape(B, ch, k, 1)
            raw_L_spatial = p[..., 2:5].reshape(B, ch, k, 3)
            L_spatial_params = self.spatial_mapper(raw_L_spatial.view(-1, 3)).view(
                B, ch, k, 3
            )
            L_spatial = self.construct_lower_triangular(L_spatial_params, s=2)
            theta_xy = (p[..., 5].reshape(B, ch, k) + math.pi) % (2 * math.pi) - math.pi
            # w = p[..., 6].reshape(B, ch, k)
            logits = p[..., 6].reshape(B, ch, k)
            tau = F.softplus(self.log_temp).clamp_min(self.tau_min)
            w = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            alpha = torch.sigmoid(p[..., 7].reshape(B, ch, k))
            c = F.softplus(p[..., 8].reshape(B, ch, k)) + self.min_diag
            if ch == 1:
                raw_L_color = p[..., 9:10].reshape(B, ch, k, 1)
                L_color_params = self.color_mapper_1(raw_L_color.view(-1, 1)).view(
                    B, ch, k, 1
                )
                color_mean = torch.zeros_like(mu_x)
            elif ch == 3:
                raw_L_color = p[..., 9:15].reshape(B, ch, k, 6)
                L_color_params = self.color_mapper_3(raw_L_color.view(-1, 6)).view(
                    B, ch, k, 6
                )
                color_mean = p[..., 15:18].reshape(B, ch, k, 3)
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            color_cov_size = 1 if ch == 1 else 3
            L_color = self.construct_lower_triangular(L_color_params, s=color_cov_size)
            mu_xy = torch.cat([mu_x, mu_y, color_mean], dim=-1)
            cov_matrix = self.cov_mat(L_spatial, theta_xy, L_color, ch)
            return mu_xy, cov_matrix, w, alpha, c
        elif self.kernel_type == KernelType.GAUSSIAN:
            mu_x = p[..., 0].reshape(B, ch, k, 1)
            mu_y = p[..., 1].reshape(B, ch, k, 1)
            raw_L_spatial = p[..., 2:5].reshape(B, ch, k, 3)
            L_spatial_params = self.spatial_mapper(raw_L_spatial.view(-1, 3)).view(
                B, ch, k, 3
            )
            L_spatial = self.construct_lower_triangular(L_spatial_params, s=2)
            theta_xy = (p[..., 5].reshape(B, ch, k) + math.pi) % (2 * math.pi) - math.pi
            # w = p[..., 6].reshape(B, ch, k)
            logits = p[..., 6].reshape(B, ch, k)
            tau = F.softplus(self.log_temp).clamp_min(self.tau_min)
            w = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            if ch == 1:
                raw_L_color = p[..., 7:8].reshape(B, ch, k, 1)
                L_color_params = self.color_mapper_1(raw_L_color.view(-1, 1)).view(
                    B, ch, k, 1
                )
                color_mean = torch.zeros_like(mu_x)
            elif ch == 3:
                raw_L_color = p[..., 7:13].reshape(B, ch, k, 6)
                L_color_params = self.color_mapper_3(raw_L_color.view(-1, 6)).view(
                    B, ch, k, 6
                )
                color_mean = p[..., 13:16].reshape(B, ch, k, 3)
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            color_cov_size = 1 if ch == 1 else 3
            L_color = self.construct_lower_triangular(L_color_params, s=color_cov_size)
            mu_xy = torch.cat([mu_x, mu_y, color_mean], dim=-1)
            cov_matrix = self.cov_mat(L_spatial, theta_xy, L_color, ch)
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
        """
        Compute a combined kernel from a Gaussian and a Cauchy component.
        Here, we compute:
          e = -0.5 (x - mu)^T Sigma_inv (x - mu)
        and then a Gaussian term G_sigma = exp(e) and a Cauchy term.
        The final kernel is a convex combination controlled by alpha.
        """
        d = x - mu  # [B, ch, k, W, H, D]
        e = -0.5 * torch.einsum("bckwhd,bckde,bckwhe->bckwh", d, Sigma_inv, d)
        mx = e.max(dim=2, keepdim=True).values
        e = e - mx
        G_sigma = torch.exp(e)
        norm_x = torch.linalg.norm(d[..., :2], dim=-1)
        Sigma_inv_diag = Sigma_inv[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
        denom = c.unsqueeze(-1).unsqueeze(-1) * Sigma_inv_diag.clamp(min=self.min_diag)
        denom = denom.clamp(min=self.min_denom)
        C_csigma = 1.0 / (1.0 + norm_x**2 / denom)
        combined = (
            alpha.unsqueeze(-1).unsqueeze(-1) * G_sigma
            + (1 - alpha.unsqueeze(-1).unsqueeze(-1)) * C_csigma
        )
        return combined

    def gaussian_kernel(
        self, x: torch.Tensor, mu_spatial: torch.Tensor, Sigma_inv_spatial: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the standard Gaussian kernel.
        """
        d = x - mu_spatial  # [B, ch, k, W, H, 2]
        e = -0.5 * torch.einsum("bckwhd,bckde,bckwhe->bckwh", d, Sigma_inv_spatial, d)
        mx = e.max(dim=2, keepdim=True).values
        e = e - mx
        G_sigma = torch.exp(e)
        return G_sigma

    def forward_spatial(self, h: int, w: int, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for static (non-dynamic) kernel estimation.
        This function:
          1. Extracts the latent parameters,
          2. Regularizes the covariance matrix,
          3. Computes its Cholesky decomposition and inverse,
          4. Computes the Gaussian (or Gaussian-Cauchy) kernels,
          5. Normalizes and sums over experts to produce the output.
        """
        B, ch, _ = params.shape  # [B, ch, k * param_per_kernel]
        k = self.kernel

        mu, cov, wt, alp, cst = self.extract_parameters(params, k, ch)
        d = cov.shape[-1]
        I = torch.eye(d, device=cov.device).view(1, 1, 1, d, d)
        eig = torch.linalg.eigvalsh(cov)
        m = eig.min(dim=-1).values[..., None, None]
        eps = torch.clamp(F.softplus(-m), min=self.reg_lambda) + self.reg_lambda

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            # cov_reg = cov + self.reg_lambda * I
            cov_reg = cov + eps * I
            L_chol = torch.linalg.cholesky(cov_reg)
            # L_chol = torch.linalg.cholesky_ex(cov_reg)

        SI = torch.cholesky_inverse(L_chol)

        G = self.grid(h, w, params.device)  # [W, H, 2]
        G_exp = G.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(B, ch, k, 1, 1, 1)
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
            raise ValueError(f"Unsupported number of channels: {ch}")

        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            K_out = self.gaussian_cauchy_kernel(X, mu_full, SI, alp, cst)
        else:
            mu_sp = mu[..., :2].reshape(B, ch, -1, 1, 1, 2)
            SI_sp = SI[..., :2, :2]
            K_out = self.gaussian_kernel(G_exp, mu_sp, SI_sp)

        K_out = K_out * wt.unsqueeze(-1).unsqueeze(-1)
        KS = K_out.sum(dim=2, keepdim=True)
        K_norm = K_out / (KS + 1e-8)
        out = K_norm.sum(dim=2)
        return torch.clamp(out, min=0.0, max=1.0)

    def extract_dynamic(
        self, x: torch.Tensor, cnt: torch.Tensor, p: int
    ) -> torch.Tensor:
        """
        Extract dynamic kernel parameters when the number of kernels varies per sample.
        """
        B, C, _ = x.shape
        K = int(cnt.max().item())
        lst = []
        for i in range(B):
            k_i = int(cnt[i].item())
            xi = x[i, :, : k_i * p].view(C, k_i, p)
            if k_i < K:
                pad = torch.zeros(C, K - k_i, p, device=x.device, dtype=x.dtype)
                xi = torch.cat([xi, pad], dim=1)
            lst.append(xi.unsqueeze(0))
        return torch.cat(lst, dim=0)

    def forward_spatial_(
        self, h: int, w: int, params: torch.Tensor, cnt: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the dynamic case when each sample may use a different number of kernels.
        """
        B, ch, L = params.shape
        p = L // (cnt.max().item() if cnt.numel() > 0 else 1)
        x_dyn = self.extract_dynamic(params, cnt, p)
        x_flat = x_dyn.view(B, ch, -1)
        mu, cov, wt, alp, cst = self.extract_parameters(x_flat, x_dyn.shape[2], ch)
        d = cov.shape[-1]
        I = torch.eye(d, device=cov.device).view(1, 1, 1, d, d)
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
            raise ValueError(f"Unsupported number of channels: {ch}")
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
        """
        The forward method chooses between the static and dynamic kernel cases.
        """
        if cnt is None:
            return self.forward_spatial(h, w, params)
        else:
            return self.forward_spatial_(h, w, params, cnt)


class MoE_(Backbone[MoEConfig]):
    def __init__(self, cfg: MoEConfig):
        super().__init__(cfg)
        self.kernel = cfg.kernel
        self.sharpening_factor = cfg.sharpening_factor
        self.kernel_type = cfg.kernel_type

        self.min_diag = 1e-10
        self.min_denom = 1e-10

        self.temp = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.reg_lambda = nn.Parameter(torch.tensor(1e-10), requires_grad=True)

        self.spatial_mapper = spectral_norm(nn.Linear(3, 3))

        self.color_mapper_1 = spectral_norm(nn.Linear(1, 1))
        self.color_mapper_3 = spectral_norm(nn.Linear(6, 6))

    def grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        xx = torch.linspace(0.0, 1.0, width, device=device)
        yy = torch.linspace(0.0, 1.0, height, device=device)
        gx, gy = torch.meshgrid(xx, yy, indexing="ij")
        grid = torch.stack((gx, gy), dim=-1)
        return grid.float()

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        ct = torch.cos(theta).unsqueeze(-1)
        st = torch.sin(theta).unsqueeze(-1)
        R = torch.cat([ct, -st, st, ct], dim=-1)
        return R.view(*theta.shape, 2, 2)

    def construct_lower_triangular_size1(self, params: torch.Tensor) -> torch.Tensor:
        B, ch, k, _ = params.shape
        L = torch.zeros(B, ch, k, 1, 1, device=params.device, dtype=params.dtype)
        L[..., 0, 0] = F.softplus(params[..., 0]) + 1e-1
        # L[..., 0, 0] = torch.exp(params[..., 0]) + 1e-2
        return L

    def construct_lower_triangular_size2(self, params: torch.Tensor) -> torch.Tensor:
        B, ch, k, _ = params.shape
        l11, l21, l22 = torch.split(params, 1, dim=-1)

        l11 = F.softplus(l11) + 1e-1
        l22 = F.softplus(l22) + 1e-1

        # l11 = torch.exp(l11) + 1e-2
        # l22 = torch.exp(l22) + 1e-2

        L = torch.zeros(B, ch, k, 2, 2, device=params.device, dtype=params.dtype)
        L[..., 0, 0] = l11.squeeze(-1)
        L[..., 1, 0] = l21.squeeze(-1)
        L[..., 1, 1] = l22.squeeze(-1)
        return L

    def construct_lower_triangular_size3(self, params: torch.Tensor) -> torch.Tensor:
        B, ch, k, _ = params.shape
        l11, l21, l22, l31, l32, l33 = torch.split(params, 1, dim=-1)

        l11 = F.softplus(l11) + 1e-1
        l22 = F.softplus(l22) + 1e-1
        l33 = F.softplus(l33) + 1e-1

        # l11 = torch.exp(l11) + 1e-2
        # l22 = torch.exp(l22) + 1e-2
        # l33 = torch.exp(l33) + 1e-2

        L = torch.zeros(B, ch, k, 3, 3, device=params.device, dtype=params.dtype)
        L[..., 0, 0] = l11.squeeze(-1)
        L[..., 1, 0] = l21.squeeze(-1)
        L[..., 1, 1] = l22.squeeze(-1)
        L[..., 2, 0] = l31.squeeze(-1)
        L[..., 2, 1] = l32.squeeze(-1)
        L[..., 2, 2] = l33.squeeze(-1)
        return L

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
        R = self.ang_to_rot_mat(theta_xy)
        C_xy = torch.matmul(R, torch.matmul(L_spatial, L_spatial.transpose(-2, -1)))
        C_xy = torch.matmul(C_xy, R.transpose(-2, -1))
        C_xy = 0.5 * (C_xy + C_xy.transpose(-2, -1))
        if ch == 1:
            C_color = (
                torch.matmul(L_color, L_color.transpose(-2, -1)).squeeze(-1).squeeze(-1)
            )
            B_, ch_, k_ = C_xy.shape[:3]
            C_full = torch.zeros(
                B_, ch_, k_, 3, 3, device=C_xy.device, dtype=C_xy.dtype
            )
            C_full[..., :2, :2] = C_xy
            C_full[..., 2, 2] = C_color
        elif ch == 3:
            C_color = torch.matmul(L_color, L_color.transpose(-2, -1))
            B_, ch_, k_ = C_xy.shape[:3]
            C_full = torch.zeros(
                B_, ch_, k_, 5, 5, device=C_xy.device, dtype=C_xy.dtype
            )
            C_full[..., :2, :2] = C_xy
            C_full[..., 2:, 2:] = C_color
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")
        return C_full * self.sharpening_factor

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        p: [B, ch, k * param_per_kernel] reshaped to [B, ch, k, param_per_kernel]
        For both kernel types (GAUSSIAN_CAUCHY and GAUSSIAN), we apply:
         - Spectral normalization on raw spatial (indices 2:5) and color parameters,
         - A differentiable kernel weight selection via Gumbel–Softmax using a learnable temperature.
        """
        B, _, _ = p.shape
        p = p.view(B, ch, k, -1)
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            # Spatial means.
            mu_x = p[..., 0].reshape(B, ch, k, 1)
            mu_y = p[..., 1].reshape(B, ch, k, 1)
            # Apply spectral mapping to spatial covariance parameters.
            raw_L_spatial = p[..., 2:5].reshape(B, ch, k, 3)
            L_spatial_params = self.spatial_mapper(raw_L_spatial.view(-1, 3)).view(
                B, ch, k, 3
            )
            L_spatial = self.construct_lower_triangular(L_spatial_params, s=2)
            theta_xy = (p[..., 5].reshape(B, ch, k) + math.pi) % (2 * math.pi) - math.pi
            # Use Gumbel-softmax for mixture weight selection.
            logits = p[..., 6].reshape(B, ch, k)
            # tau = F.softplus(self.temp) + 1e-8
            tau = F.softplus(self.temp).clamp_min(0.2)

            w = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            alpha = torch.sigmoid(p[..., 7].reshape(B, ch, k))
            c = F.softplus(p[..., 8].reshape(B, ch, k)) + self.min_diag
            if ch == 1:
                raw_L_color = p[..., 9:10].reshape(B, ch, k, 1)
                L_color_params = self.color_mapper_1(raw_L_color.view(-1, 1)).view(
                    B, ch, k, 1
                )
                color_mean = torch.zeros_like(mu_x)
            elif ch == 3:
                raw_L_color = p[..., 9:15].reshape(B, ch, k, 6)
                L_color_params = self.color_mapper_3(raw_L_color.view(-1, 6)).view(
                    B, ch, k, 6
                )
                color_mean = p[..., 15:18].reshape(B, ch, k, 3)
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            color_cov_size = 1 if ch == 1 else 3
            L_color = self.construct_lower_triangular(L_color_params, s=color_cov_size)
            mu_xy = torch.cat([mu_x, mu_y, color_mean], dim=-1)
            cov_matrix = self.cov_mat(L_spatial, theta_xy, L_color, ch)
            return mu_xy, cov_matrix, w, alpha, c
        elif self.kernel_type == KernelType.GAUSSIAN:
            mu_x = p[..., 0].reshape(B, ch, k, 1)
            mu_y = p[..., 1].reshape(B, ch, k, 1)
            raw_L_spatial = p[..., 2:5].reshape(B, ch, k, 3)
            L_spatial_params = self.spatial_mapper(raw_L_spatial.view(-1, 3)).view(
                B, ch, k, 3
            )
            L_spatial = self.construct_lower_triangular(L_spatial_params, s=2)
            theta_xy = (p[..., 5].reshape(B, ch, k) + math.pi) % (2 * math.pi) - math.pi
            logits = p[..., 6].reshape(B, ch, k)

            # tau = F.softplus(self.temp) + 1e-8
            tau = F.softplus(self.temp).clamp_min(0.2)

            w = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            if ch == 1:
                raw_L_color = p[..., 7:8].reshape(B, ch, k, 1)
                L_color_params = self.color_mapper_1(raw_L_color.view(-1, 1)).view(
                    B, ch, k, 1
                )
                color_mean = torch.zeros_like(mu_x)
            elif ch == 3:
                raw_L_color = p[..., 7:13].reshape(B, ch, k, 6)
                L_color_params = self.color_mapper_3(raw_L_color.view(-1, 6)).view(
                    B, ch, k, 6
                )
                color_mean = p[..., 13:16].reshape(B, ch, k, 3)
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            color_cov_size = 1 if ch == 1 else 3
            L_color = self.construct_lower_triangular(L_color_params, s=color_cov_size)
            mu_xy = torch.cat([mu_x, mu_y, color_mean], dim=-1)
            cov_matrix = self.cov_mat(L_spatial, theta_xy, L_color, ch)
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
        e = -0.5 * torch.einsum("bckwhd,bckde,bckwhe->bckwh", d, Sigma_inv, d)
        mx = e.max(dim=2, keepdim=True).values
        e = e - mx
        G_sigma = torch.exp(e)
        norm_x = torch.linalg.norm(d[..., :2], dim=-1)
        Sigma_inv_diag = Sigma_inv[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
        denom = c.unsqueeze(-1).unsqueeze(-1) * Sigma_inv_diag.clamp(min=self.min_diag)
        denom = denom.clamp(min=self.min_denom)
        C_csigma = 1.0 / (1.0 + norm_x**2 / denom)
        combined = (
            alpha.unsqueeze(-1).unsqueeze(-1) * G_sigma
            + (1 - alpha.unsqueeze(-1).unsqueeze(-1)) * C_csigma
        )
        return combined

    def gaussian_kernel(
        self, x: torch.Tensor, mu_spatial: torch.Tensor, Sigma_inv_spatial: torch.Tensor
    ) -> torch.Tensor:
        d = x - mu_spatial  # [B, ch, k, W, H, 2]
        e = -0.5 * torch.einsum("bckwhd,bckde,bckwhe->bckwh", d, Sigma_inv_spatial, d)
        mx = e.max(dim=2, keepdim=True).values
        e = e - mx
        G_sigma = torch.exp(e)
        return G_sigma

    def forward_spatial(self, h: int, w: int, params: torch.Tensor) -> torch.Tensor:
        B, ch, _ = params.shape  # [B, ch, k * param_per_kernel]
        k = self.kernel

        mu, cov, wt, alp, cst = self.extract_parameters(params, k, ch)
        d = cov.shape[-1]
        I = torch.eye(d, device=cov.device).view(1, 1, 1, d, d)
        eig = torch.linalg.eigvalsh(cov)
        m = eig.min(dim=-1).values[..., None, None]
        eps = torch.clamp(F.softplus(-m), min=1e-4) + self.reg_lambda
        # eps = torch.abs(m) + self.reg_lambda

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            cov_reg = cov + eps * I
            L_chol = torch.linalg.cholesky(cov_reg)
        SI = torch.cholesky_inverse(L_chol)

        G = self.grid(h, w, params.device)  # [W, H, 2]
        G_exp = G.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(B, ch, k, 1, 1, 1)
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
            raise ValueError(f"Unsupported number of channels: {ch}")

        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            K_out = self.gaussian_cauchy_kernel(X, mu_full, SI, alp, cst)
        else:
            mu_sp = mu[..., :2].reshape(B, ch, -1, 1, 1, 2)
            SI_sp = SI[..., :2, :2]
            K_out = self.gaussian_kernel(G_exp, mu_sp, SI_sp)

        K_out = K_out * wt.unsqueeze(-1).unsqueeze(-1)
        KS = K_out.sum(dim=2, keepdim=True)
        K_norm = K_out / (KS + 1e-8)
        out = K_norm.sum(dim=2)
        return torch.clamp(out, min=0.0, max=1.0)

    def extract_dynamic(
        self, x: torch.Tensor, cnt: torch.Tensor, p: int
    ) -> torch.Tensor:
        B, C, _ = x.shape
        K = int(cnt.max().item())
        lst = []
        for i in range(B):
            k_i = int(cnt[i].item())
            xi = x[i, :, : k_i * p].view(C, k_i, p)
            if k_i < K:
                pad = torch.zeros(C, K - k_i, p, device=x.device, dtype=x.dtype)
                xi = torch.cat([xi, pad], dim=1)
            lst.append(xi.unsqueeze(0))
        return torch.cat(lst, dim=0)

    def forward_spatial_(
        self, h: int, w: int, params: torch.Tensor, cnt: torch.Tensor
    ) -> torch.Tensor:
        B, ch, L = params.shape
        p = L // (cnt.max().item() if cnt.numel() > 0 else 1)
        x_dyn = self.extract_dynamic(params, cnt, p)
        x_flat = x_dyn.view(B, ch, -1)
        mu, cov, wt, alp, cst = self.extract_parameters(x_flat, x_dyn.shape[2], ch)
        d = cov.shape[-1]
        I = torch.eye(d, device=cov.device).view(1, 1, 1, d, d)
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
    num_chunks: int = 1


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

        self.encoder = Encoder(cfg.EncoderConfig, cfg.phw, d_in=cfg.d_in, d_out=d_out)
        self.decoder = MoE(cfg.DecoderConfig)
        self.params_per_kernel = params_per_kernel

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

    def extract_blocks(
        self, x: torch.Tensor, block_size: int, overlap: int
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:

        B, C, H, W = x.shape
        step = block_size - overlap
        pad = block_size // 2
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        out_h = (xp.shape[2] - block_size) // step + 1
        out_w = (xp.shape[3] - block_size) // step + 1
        blocks = F.unfold(xp, kernel_size=block_size, stride=step)
        blocks = blocks.transpose(1, 2).reshape(
            B, out_h * out_w, C, block_size, block_size
        )
        return blocks, (B, out_h * out_w, C, H, W)

    def reconstruct(
        self,
        blocks: torch.Tensor,
        dims: Tuple[int, int, int, int, int],
        block_size: int,
        overlap: int,
    ) -> torch.Tensor:
        B, _, C, H, W = dims

        device = blocks.device
        step = block_size - overlap
        pad = block_size // 2
        out_h, out_w = H + 2 * pad, W + 2 * pad

        blocks_reshaped = blocks.reshape(B, -1, C * block_size * block_size).transpose(
            1, 2
        )

        recon_padded = F.fold(
            blocks_reshaped,
            output_size=(out_h, out_w),
            kernel_size=block_size,
            stride=step,
        )

        window = torch.hann_window(block_size, periodic=False, device=device)
        window2d = window.unsqueeze(0) * window.unsqueeze(1)
        window2d = window2d.view(1, 1, block_size, block_size)
        ones = torch.ones_like(blocks)
        ones_reshaped = ones.reshape(B, -1, C * block_size * block_size).transpose(1, 2)

        weight_sum = F.fold(
            ones_reshaped,
            output_size=(out_h, out_w),
            kernel_size=block_size,
            stride=step,
        )
        recon_norm = recon_padded / weight_sum.clamp_min(1e-8)
        return recon_norm[:, :, pad : H + pad, pad : W + pad]

    @staticmethod
    def det_split(x: torch.Tensor, num_chunks: int):
        k_eff = min(x.shape[0], num_chunks)
        boundaries = [math.floor(i * x.shape[0] / k_eff) for i in range(k_eff + 1)]
        return [x[boundaries[i] : boundaries[i + 1]] for i in range(k_eff)]

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_p, dims = self.extract_blocks(x, self.phw, self.overlap)

        if x_p.ndim == 5:
            x_p = x_p.reshape(-1, *x_p.shape[2:])

        chunks = self.det_split(x_p, self.cfg.num_chunks)

        res = [
            self.encoder(chunk, self.snet(chunk), self.knet(chunk)) for chunk in chunks
        ]

        gaussians, kinfo, sigma = map(lambda arr: torch.cat(arr, dim=0), zip(*res))

        # B, L, C, H, W, pad = dims

        B, L, C, H, W = dims
        sp = self.phw * self.encoder.scale_factor
        dec_chunks = self.det_split(gaussians, self.cfg.num_chunks)
        dec = torch.cat([self.decoder(sp, sp, bt) for bt in dec_chunks], dim=0)

        rec = self.reconstruct(
            dec,
            (
                B,
                L,
                C,
                H * self.encoder.scale_factor,
                W * self.encoder.scale_factor,
                # pad * self.encoder.scale_factor,
            ),
            sp,
            self.overlap * self.encoder.scale_factor,
        )
        kinfo_avg = kinfo.view(B, L, -1).mean(dim=1)
        sigma_avg = sigma.view(B, L, 1, 1, 1).mean(dim=1)

        return rec, kinfo_avg, sigma_avg


class Autoencoder_(Backbone[AutoencoderConfig]):
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

        self.encoder = Encoder(cfg.EncoderConfig, cfg.phw, d_in=cfg.d_in, d_out=d_out)
        self.decoder = MoE(cfg.DecoderConfig)
        self.params_per_kernel = params_per_kernel

    @staticmethod
    def hann_window(block_size: int, device: torch.device) -> torch.Tensor:
        hann_1d = torch.hann_window(block_size, periodic=False, device=device)
        hann_2d = hann_1d.unsqueeze(0) * hann_1d.unsqueeze(1)
        return hann_2d

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

    def extract_blocks(
        self, x: torch.Tensor, block_size: int, overlap: int
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        B, C, H, W = x.shape
        step = block_size - overlap
        pad = block_size // 2
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        out_h = (xp.shape[2] - block_size) // step + 1
        out_w = (xp.shape[3] - block_size) // step + 1
        s0, s1, s2, s3 = xp.stride()
        shape = (B, C, out_h, out_w, block_size, block_size)
        stride = (s0, s1, step * s2, step * s3, s2, s3)
        blocks = xp.as_strided(shape, stride)
        blocks = blocks.permute(0, 2, 3, 1, 4, 5).reshape(
            B, out_h * out_w, C, block_size, block_size
        )
        L = out_h * out_w
        return blocks, (B, L, C, H, W)

    def reconstruct(
        self,
        blocks: torch.Tensor,
        dims: Tuple[int, int, int, int],
        block_size: int,
        overlap: int,
    ) -> torch.Tensor:
        B, C, H, W = dims
        step = block_size - overlap
        pad = block_size // 2
        device = blocks.device
        out_h, out_w = H + 2 * pad, W + 2 * pad
        window = (
            self.hann_window(block_size, device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )

        w_blocks = blocks * window
        w_blocks = w_blocks.permute(0, 2, 3, 4, 1).reshape(
            B, C * block_size * block_size, -1
        )

        recon_pad = F.fold(
            w_blocks,
            output_size=(out_h, out_w),
            kernel_size=block_size,
            stride=step,
        )

        ones = torch.ones_like(blocks, device=device)
        w_ones = ones * window
        w_ones = w_ones.permute(0, 2, 3, 4, 1).reshape(
            B, C * block_size * block_size, -1
        )

        wsum_pad = F.fold(
            w_ones,
            output_size=(out_h, out_w),
            kernel_size=block_size,
            stride=step,
        )
        recon_pad = recon_pad / wsum_pad.clamp_min(1e-8)
        recon = recon_pad[:, :, pad : H + pad, pad : W + pad]
        return recon

    @staticmethod
    def det_split(x: torch.Tensor, num_chunks: int):
        k_eff = min(x.shape[0], num_chunks)
        boundaries = [math.floor(i * x.shape[0] / k_eff) for i in range(k_eff + 1)]
        return [x[boundaries[i] : boundaries[i + 1]] for i in range(k_eff)]

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_p, dims = self.extract_blocks(x, self.phw, self.overlap)

        if x_p.ndim == 5:
            x_p = x_p.reshape(-1, *x_p.shape[2:])

        chunks = self.det_split(x_p, self.cfg.num_chunks)

        res = [
            self.encoder(chunk, self.snet(chunk), self.knet(chunk)) for chunk in chunks
        ]

        gaussians, kinfo, sigma = map(lambda arr: torch.cat(arr, dim=0), zip(*res))

        # B, L, C, H, W, pad = dims

        B, L, C, H, W = dims
        sp = self.phw * self.encoder.scale_factor
        dec_chunks = self.det_split(gaussians, self.cfg.num_chunks)
        dec = torch.cat([self.decoder(sp, sp, bt) for bt in dec_chunks], dim=0)

        rec = self.reconstruct(
            dec,
            (
                B,
                # L,
                C,
                H * self.encoder.scale_factor,
                W * self.encoder.scale_factor,
                # pad * self.encoder.scale_factor,
            ),
            sp,
            self.overlap * self.encoder.scale_factor,
        )
        kinfo_avg = kinfo.view(B, L, -1).mean(dim=1)
        sigma_avg = sigma.view(B, L, 1, 1, 1).mean(dim=1)

        return rec, kinfo_avg, sigma_avg


# def reconstruct(
#         self,
#         blocks: torch.Tensor,
#         dims: Tuple[int, int, int, int],
#         block_size: int,
#         overlap: int,
#     ) -> torch.Tensor:
#         B, C, H, W = dims
#         step = block_size - overlap
#         pad = block_size // 2
#         device = blocks.device
#         out_h, out_w = H + 2 * pad, W + 2 * pad
#         window = self.hann_window(block_size, device).view(
#             1, 1, 1, block_size, block_size
#         )
#         weighted = blocks * window
#         B, L, C, bs1, bs2 = weighted.shape
#         weighted = weighted.permute(0, 2, 3, 4, 1).reshape(
#             B, C * block_size * block_size, L
#         )
#         rec = F.fold(
#             weighted, output_size=(out_h, out_w), kernel_size=block_size, stride=step
#         )
#         ones = window.expand(B, L, C, block_size, block_size)
#         ones = ones.permute(0, 2, 3, 4, 1).reshape(B, C * block_size * block_size, L)
#         wsum = F.fold(
#             ones, output_size=(out_h, out_w), kernel_size=block_size, stride=step
#         )
#         rec.div_(wsum.clamp_min(1e-8))
#         return rec[:, :, pad : H + pad, pad : W + pad]

# def forward_(
#     self, x: torch.Tensor
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     pchs, dm = self.extract_blocks(x, self.phw, self.overlap)
#     if pchs.ndim == 5:
#         pchs = pchs.reshape(-1, *pchs.shape[2:])
#     chunks = self.det_split(pchs, self.cfg.num_chunks)
#     res = [
#         self.encoder(chunk, self.snet(chunk), self.knet(chunk)) for chunk in chunks
#     ]
#     g, kinfo, sigma, kcnt = map(lambda a: torch.cat(a, dim=0), zip(*res))
#     B_enc, C_enc, L_enc = g.shape
#     B_orig, L_orig, C_orig, H, W, pd = dm
#     sp = self.phw * self.encoder.scale_factor
#     g_r = g.view(B_orig, L_orig, C_enc, L_enc)
#     g_flat = g_r.view(B_enc, C_enc, L_enc)
#     thresholds = kcnt.view(B_enc) * self.params_per_kernel
#     idx = torch.arange(L_enc, device=g.device).unsqueeze(0).expand(B_enc, L_enc)
#     mask = idx < thresholds.unsqueeze(1)
#     mask_exp = mask.unsqueeze(1).expand(B_enc, C_enc, L_enc)
#     g_mask = g_flat * mask_exp
#     splits = self.det_split(g_mask, self.cfg.num_chunks)
#     kcnt_splits = self.det_split(kcnt, self.cfg.num_chunks)
#     dec_list = [
#         self.decoder(sp, sp, latent, cnt)
#         for latent, cnt in zip(splits, kcnt_splits)
#     ]
#     dec = torch.cat(dec_list, dim=0)
#     rec = self.reconstruct(
#         dec,
#         (
#             B_orig,
#             L_orig,
#             C_orig,
#             H * self.encoder.scale_factor,
#             W * self.encoder.scale_factor,
#             pd * self.encoder.scale_factor,
#         ),
#         sp,
#         self.overlap * self.encoder.scale_factor,
#     )
#     kinfo_avg = kinfo.view(B_orig, L_orig, -1).mean(dim=1)
#     sigma_avg = sigma.view(B_orig, L_orig, 1, 1, 1).mean(dim=1)
#     return rec, kinfo_avg, sigma_avg


# @staticmethod
# def hann_window_(
#     block_size: int, C: int, step: int, device: torch.device
# ) -> torch.Tensor:
#     hann_1d = torch.hann_window(block_size, periodic=False, device=device)
#     hann_2d = hann_1d.unsqueeze(1) * hann_1d.unsqueeze(0)
#     window = hann_2d.view(1, 1, block_size * block_size)
#     window = window.repeat(C, 1, 1)
#     window = window.view(1, C * block_size * block_size, 1)
#     window = window * (step / block_size)
#     return window


# def extract_blocks_(
#         self, img_tensor: torch.Tensor, block_size: int, overlap: int
#     ) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
#         B, C, H, W = img_tensor.shape
#         step = block_size - overlap
#         pad = (block_size - step) // 2 + step
#         img_padded = F.pad(img_tensor, (pad, pad, pad, pad), mode="reflect")
#         patches = F.unfold(img_padded, kernel_size=block_size, stride=step)
#         window = self.hann_window(block_size, C, step, img_tensor.device)
#         L = patches.shape[-1]
#         window = window.repeat(1, 1, L)
#         patches = patches * window
#         patches = (
#             patches.view(B, C, block_size, block_size, L)
#             .permute(0, 4, 1, 2, 3)
#             .contiguous()
#         )
#         return patches, (B, L, C, H, W, pad)

# def reconstruct_(
#         self,
#         decoded_patches: torch.Tensor,
#         dims: Tuple[int, int, int, int, int, int],
#         block_size_out: int,
#         overlap_out: int,
#     ) -> torch.Tensor:
#         B, L, C, H, W, pad_out = dims
#         step_out = block_size_out - overlap_out

#         decoded_patches = (
#             decoded_patches.reshape(B, L, C * block_size_out * block_size_out)
#             .permute(0, 2, 1)
#             .contiguous()
#         )

#         recon_padded = F.fold(
#             decoded_patches,
#             output_size=(H + 2 * pad_out, W + 2 * pad_out),
#             kernel_size=block_size_out,
#             stride=step_out,
#         )

#         window = self.hann_window(block_size_out, C, step_out, decoded_patches.device)
#         window_sum = F.fold(
#             torch.ones_like(decoded_patches) * window,
#             output_size=(H + 2 * pad_out, W + 2 * pad_out),
#             kernel_size=block_size_out,
#             stride=step_out,
#         )
#         recon_padded = recon_padded / window_sum.clamp_min(1e-8)
#         recon = recon_padded[:, :, pad_out : pad_out + H, pad_out : pad_out + W]

#         return recon
