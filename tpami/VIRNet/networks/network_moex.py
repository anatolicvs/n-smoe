# type: ignore

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


def detect_sdpa_backend(device: torch.device, dtype: torch.dtype, opt_out_flash: bool = False) -> list:
    """Detect the best available SDPA backend for given device/dtype constraints."""
    available_backends = []

    # FlashAttention requires GPU + fp16/bf16
    if not opt_out_flash and device.type == "cuda" and dtype in [torch.float16, torch.bfloat16]:
        available_backends.append(SDPBackend.FLASH_ATTENTION)

    # Efficient attention is generally available
    available_backends.append(SDPBackend.EFFICIENT_ATTENTION)

    # Math backend is always available as fallback
    available_backends.append(SDPBackend.MATH)

    return available_backends


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
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
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
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class AttentionPool2d_(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5)
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
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
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
        max_tokens: int = 4096,  # Add max_tokens guard
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim or embed_dim
        self.max_tokens = max_tokens
        self.fourier_pos = FourierPosition(in_dim=2, mapping_size=fourier_size, scale=fourier_scale)
        self.linear_pos = nn.Linear(fourier_size, embed_dim)
        self.global_pos = nn.Parameter(torch.empty(embed_dim))
        nn.init.xavier_uniform_(self.global_pos.unsqueeze(0))
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, self.output_dim)

        # Positional embeddings computed fresh each time

    def _get_positional_embeddings(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Compute positional embeddings fresh each time to avoid gradient graph issues."""
        grid_y = torch.linspace(0, 1, H, device=device)
        grid_x = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).view(-1, 2)
        fourier_features = self.fourier_pos(coords)
        pos_spatial = self.linear_pos(fourier_features)
        return pos_spatial

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = input_x.shape
        if C != self.embed_dim:
            raise ValueError(f"Input channel count {C} must equal embed_dim ({self.embed_dim})")

        # Check if we need to pool to reduce tokens
        tokens = H * W
        pool_stride = 1
        if tokens > self.max_tokens:
            pool_stride = int(math.ceil(math.sqrt(tokens / self.max_tokens)))
            input_x = F.avg_pool2d(input_x, kernel_size=pool_stride, stride=pool_stride)
            H, W = input_x.shape[-2:]

        x = input_x.flatten(start_dim=2).permute(2, 0, 1)
        global_token = x.mean(dim=0, keepdim=True)
        x = torch.cat([global_token, x], dim=0)

        # Use cached positional embeddings
        pos_spatial = self._get_positional_embeddings(H, W, input_x.device)
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
    def __init__(self, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dropout_p = dropout

        self.scale_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)

        scale = self.scale_param / math.sqrt(ch)

        # Reshape for SDPA
        q = q.view(bs, self.n_heads, ch, length).transpose(-2, -1)  # [bs, n_heads, length, ch]
        k = k.view(bs, self.n_heads, ch, length).transpose(-2, -1)  # [bs, n_heads, length, ch]
        v = v.view(bs, self.n_heads, ch, length).transpose(-2, -1)  # [bs, n_heads, length, ch]

        dropout_p = self.dropout_p if self.training else 0.0

        with sdpa_kernel(backends):
            a = F.scaled_dot_product_attention(q * scale, k * scale, v, dropout_p=dropout_p)

        # Reshape back to original format
        a = a.transpose(-2, -1).contiguous().view(bs, -1, length)
        return a

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
    freqs_x: torch.Tensor = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y: torch.Tensor = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
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
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
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
        opt_out_flash: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.kv_in_dim: Any | int = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim: int = embedding_dim // downsample_rate
        self.num_heads: int = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p: float = dropout
        self.opt_out_flash = opt_out_flash
        self._cached_backends = None
        self._cached_device = None
        self._cached_dtype = None

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def _get_sdpa_backends(self, device: torch.device, dtype: torch.dtype) -> list:
        """Get cached SDPA backends for the given device/dtype."""
        if self._cached_backends is None or self._cached_device != device or self._cached_dtype != dtype:
            self._cached_backends = detect_sdpa_backend(device, dtype, self.opt_out_flash)
            self._cached_device = device
            self._cached_dtype = dtype
        return self._cached_backends

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p: float = self.dropout_p if self.training else 0.0

        # Use cached backends
        backends = self._get_sdpa_backends(q.device, q.dtype)
        with sdpa_kernel(backends):
            out: torch.Tensor = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention(Attention):
    """
    Rotary Position Embedding (RoPE) Attention.

    Note: Requires head_dim to be a multiple of 4 for axial RoPE to work correctly.
    This is because 2D RoPE uses two pairs of sin/cos functions (one for each spatial dimension).
    """

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(32, 32),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Enforce RoPE head dimension constraint
        head_dim = self.internal_dim // self.num_heads
        assert head_dim % 4 == 0, f"Axial RoPE requires head_dim % 4 == 0. Got head_dim={head_dim}"

        self.compute_cis = partial(compute_axial_cis, dim=head_dim, theta=rope_theta)
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

        # Use cached backends
        backends = self._get_sdpa_backends(q.device, q.dtype)
        with sdpa_kernel(backends):
            out: torch.Tensor = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

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
            self.h_upd = Upsample(cfg.channels, False, cfg.dims, resample_2d=cfg.resample_2d)
            self.x_upd = Upsample(cfg.channels, False, cfg.dims, resample_2d=cfg.resample_2d)
        elif cfg.down:
            self.h_upd = Downsample(cfg.channels, False, cfg.dims, resample_2d=cfg.resample_2d)
            self.x_upd = Downsample(cfg.channels, False, cfg.dims, resample_2d=cfg.resample_2d)
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
            self.skip_connection = conv_nd(cfg.dims, cfg.channels, cfg.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(cfg.dims, cfg.channels, cfg.out_channels, 1)

        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), list(self.parameters()), self.cfg.use_checkpoint)

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
        self.qkv: nn.Conv1d | nn.Conv2d | nn.Conv3d = conv_nd(1, cfg.channels, cfg.channels * 3, 1)

        self.attention_type: str = cfg.attention_type if hasattr(cfg, "attention_type") else "attention"

        self.attention_map = {
            "attention": QKVAttention(cfg.num_heads, dropout=cfg.dropout_rate),
            "cross_attention": RoPEAttention(
                embedding_dim=cfg.channels,
                num_heads=cfg.num_heads,
                rope_theta=cfg.rope_theta,
                rope_k_repeat=True,
                feat_sizes=(phw, phw),
                dropout=cfg.dropout_rate,
            ),
        }

        self.attention = self.attention_map.get(self.attention_type, None)

        self.proj_out: nn.Conv1d | nn.Conv2d | nn.Conv3d = zero_module(
            conv_nd(1, cfg.channels, cfg.channels, 1)  # Force dims=1 for flattened sequences
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), list(self.parameters()), self.cfg.use_checkpoint)

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
        self.interpolation_method = interpolation_methods.get(base_resize_method, "bilinear")
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        if init_weights is not None:
            for i in range(num_layers):
                self.weights.append(nn.Parameter(torch.tensor(init_weights[2 * i], dtype=dtype)))
                self.biases.append(nn.Parameter(torch.tensor(init_weights[2 * i + 1], dtype=dtype)))
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
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, 1) * gaussian_kernel.view(1, 1, 1, kernel_size)
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
        net = F.interpolate(x, size=target_size, mode=self.interpolation_method, align_corners=False)
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
            [nn.Sequential(conv_nd(cfg.dims, d_in + extra_chn, cfg.model_channels, 3, padding=1))]
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
                if cfg.attention_resolutions is not None and ds in cfg.attention_resolutions:
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
                        else Downsample(ch, cfg.conv_resample, dims=cfg.dims, out_channels=out_ch)
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

    def get_eps(self, t: torch.Tensor, factor: float = 1e-3, min_eps: float = 1e-8) -> torch.Tensor:
        return torch.maximum(torch.tensor(min_eps, device=t.device, dtype=t.dtype), factor * t.mean())

    def forward(
        self,
        x: torch.Tensor,
        sigma_est: Optional[torch.Tensor] = None,
        kinfo_est: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        sigma = F.softplus(sigma_est) + self.get_eps(sigma_est)

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
                    tmp_list.append(F.interpolate(s_sqrt, scale_factor=self.scale_factor, mode="nearest"))
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

        gaussians = self.out(h)
        gaussians = rearrange(gaussians, "b (c latent) -> b c latent", c=self.d_in, latent=self.latent)

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
    kernel_type: Optional[Enum] = None
    activation: str = "GELU"
    min_diag: float = 1e-4
    max_diag: float = 1e2
    min_denom: float = 1e-4
    initial_temp: float = 0.5
    tau_min: float = 0.1
    reg_lambda: float = 1e-3  # Increased for better numerical stability
    grid_cache: Optional[torch.Tensor] = None
    scale_factor: int = 1


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
        self.max_diag = cfg.max_diag
        self.min_denom = cfg.min_denom
        self.tau_min = cfg.tau_min
        self.reg_lambda_param = nn.Parameter(torch.tensor(cfg.reg_lambda))
        self.log_temp = nn.Parameter(torch.log(torch.tensor(cfg.initial_temp)), requires_grad=True)

        if hasattr(nn, cfg.activation):
            self.activation = getattr(nn, cfg.activation)()
        else:
            self.activation = nn.GELU()
        self.spatial_mapper = spectral_norm(nn.Linear(3, 3))
        self.color_mapper_1 = spectral_norm(nn.Linear(1, 1))
        self.color_mapper_3 = spectral_norm(nn.Linear(6, 6))

    def grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        if self.cfg.grid_cache is not None:
            return self.cfg.grid_cache.to(device)
        yy = torch.linspace(0.0, 1.0, height, device=device)
        xx = torch.linspace(0.0, 1.0, width, device=device)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((xx, yy), dim=-1).float()  # (H,W,2)

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        ct = torch.cos(theta).unsqueeze(-1)
        st = torch.sin(theta).unsqueeze(-1)
        R = torch.cat([ct, -st, st, ct], dim=-1)
        return R.view(*theta.shape, 2, 2)

    def get_eps(self, t: torch.Tensor, factor: float = 1e-5, min_eps: float = 1e-8) -> torch.Tensor:
        return torch.maximum(torch.tensor(min_eps, device=t.device, dtype=t.dtype), factor * t.mean())

    def construct_lower_triangular(self, params: torch.Tensor, s: int) -> torch.Tensor:
        eps = self.get_eps(params, factor=1e-6, min_eps=1e-6)
        B, ch, k, _ = params.shape
        if s == 2:
            L11 = torch.clamp(
                F.softplus(params[..., 0]) + self.min_diag + eps,
                min=self.min_diag,
                max=self.max_diag,
            )
            L22 = torch.clamp(
                F.softplus(params[..., 2]) + self.min_diag + eps,
                min=self.min_diag,
                max=self.max_diag,
            )
            L21 = torch.sqrt(L11 * L22) * (2 * torch.sigmoid(params[..., 1]) - 1)
            L = torch.zeros(B, ch, k, 2, 2, device=params.device, dtype=params.dtype)
            L[..., 0, 0] = L11.squeeze(-1)
            L[..., 1, 1] = L22.squeeze(-1)
            L[..., 1, 0] = L21.squeeze(-1)
            return L
        elif s == 1:
            L = torch.zeros(B, ch, k, 1, 1, device=params.device, dtype=params.dtype)
            L[..., 0, 0] = torch.clamp(
                F.softplus(params[..., 0]) + self.min_diag + eps,
                min=self.min_diag,
                max=self.max_diag,
            )
            return L
        elif s == 3:
            L11 = torch.clamp(
                F.softplus(params[..., 0]) + self.min_diag + eps,
                min=self.min_diag,
                max=self.max_diag,
            )
            L22 = torch.clamp(
                F.softplus(params[..., 2]) + self.min_diag + eps,
                min=self.min_diag,
                max=self.max_diag,
            )
            L33 = torch.clamp(
                F.softplus(params[..., 5]) + self.min_diag + eps,
                min=self.min_diag,
                max=self.max_diag,
            )
            L21 = torch.sqrt(L11 * L22) * (2 * torch.sigmoid(params[..., 1]) - 1)
            L31 = torch.sqrt(L11 * L33) * (2 * torch.sigmoid(params[..., 3]) - 1)
            L32 = torch.sqrt(L22 * L33) * (2 * torch.sigmoid(params[..., 4]) - 1)
            L = torch.zeros(B, ch, k, 3, 3, device=params.device, dtype=params.dtype)
            L[..., 0, 0] = L11.squeeze(-1)
            L[..., 1, 0] = L21.squeeze(-1)
            L[..., 1, 1] = L22.squeeze(-1)
            L[..., 2, 0] = L31.squeeze(-1)
            L[..., 2, 1] = L32.squeeze(-1)
            L[..., 2, 2] = L33.squeeze(-1)
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
        R = self.ang_to_rot_mat(theta_xy)
        C_xy = torch.matmul(R, torch.matmul(L_spatial, L_spatial.transpose(-2, -1)))
        C_xy = torch.matmul(C_xy, R.transpose(-2, -1))
        C_xy = (C_xy + C_xy.transpose(-1, -2)) / 2
        if ch == 1:
            C_color = torch.matmul(L_color, L_color.transpose(-2, -1)).squeeze(-1).squeeze(-1)
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

    def extract_parameters(
        self, p: torch.Tensor, k: int, ch: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, _ = p.shape
        p = p.view(B, ch, k, -1)
        p = torch.nan_to_num(p, nan=0.0, posinf=10.0, neginf=-10.0)
        p = torch.clamp(p, -10.0, 10.0)
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            mu_x = p[..., 0].reshape(B, ch, k, 1)
            mu_y = p[..., 1].reshape(B, ch, k, 1)
            raw_L_spatial = p[..., 2:5].reshape(B, ch, k, 3)
            L_spatial_params = self.spatial_mapper(raw_L_spatial.view(-1, 3)).view(B, ch, k, 3)
            L_spatial = self.construct_lower_triangular(L_spatial_params, s=2)
            theta_xy = (p[..., 5].reshape(B, ch, k) + math.pi) % (2 * math.pi) - math.pi
            logits = p[..., 6].reshape(B, ch, k)
            tau = F.softplus(self.log_temp).clamp_min(self.tau_min)
            w = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            alpha = torch.clamp(torch.sigmoid(p[..., 7].reshape(B, ch, k)) + 1e-6, min=1e-4, max=0.9999)
            c = torch.clamp(
                F.softplus(p[..., 8].reshape(B, ch, k)) + self.min_diag + 1e-6,
                min=1e-3,
                max=1e4,
            )
            if ch == 1:
                raw_L_color = p[..., 9:10].reshape(B, ch, k, 1)
                L_color_params = self.color_mapper_1(raw_L_color.view(-1, 1)).view(B, ch, k, 1)
                color_mean = torch.zeros_like(mu_x)
            elif ch == 3:
                raw_L_color = p[..., 9:15].reshape(B, ch, k, 6)
                L_color_params = self.color_mapper_3(raw_L_color.view(-1, 6)).view(B, ch, k, 6)
                color_mean = p[..., 15:18].reshape(B, ch, k, 3)
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            L_color = self.construct_lower_triangular(L_color_params, s=1 if ch == 1 else 3)
            mu_xy = torch.cat([mu_x, mu_y, color_mean], dim=-1)
            cov_matrix = self.cov_mat(L_spatial, theta_xy, L_color, ch)
            return mu_xy, cov_matrix, w, alpha, c
        elif self.kernel_type == KernelType.GAUSSIAN:
            mu_x = p[..., 0].reshape(B, ch, k, 1)
            mu_y = p[..., 1].reshape(B, ch, k, 1)
            raw_L_spatial = p[..., 2:5].reshape(B, ch, k, 3)
            L_spatial_params = self.spatial_mapper(raw_L_spatial.view(-1, 3)).view(B, ch, k, 3)
            L_spatial = self.construct_lower_triangular(L_spatial_params, s=2)
            theta_xy = (p[..., 5].reshape(B, ch, k) + math.pi) % (2 * math.pi) - math.pi
            logits = p[..., 6].reshape(B, ch, k)
            tau = F.softplus(self.log_temp).clamp_min(self.tau_min)
            w = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            if ch == 1:
                raw_L_color = p[..., 7:8].reshape(B, ch, k, 1)
                L_color_params = self.color_mapper_1(raw_L_color.view(-1, 1)).view(B, ch, k, 1)
                color_mean = torch.zeros_like(mu_x)
            elif ch == 3:
                raw_L_color = p[..., 7:13].reshape(B, ch, k, 6)
                L_color_params = self.color_mapper_3(raw_L_color.view(-1, 6)).view(B, ch, k, 6)
                color_mean = p[..., 13:16].reshape(B, ch, k, 3)
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            L_color = self.construct_lower_triangular(L_color_params, s=1 if ch == 1 else 3)
            mu_xy = torch.cat([mu_x, mu_y, color_mean], dim=-1)
            cov_matrix = self.cov_mat(L_spatial, theta_xy, L_color, ch)
            return mu_xy, cov_matrix, w, None, None
        else:
            raise NotImplementedError(f"Kernel type {self.kernel_type} not implemented.")

    def gaussian_cauchy_kernel(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        L_chol: torch.Tensor,
        alpha: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        B, ch, k, h, w, d = x.shape

        d_vec = x - mu  # [B, ch, k, h, w, d]

        # Reshape for batch matrix operations
        d_flat = d_vec.reshape(B, ch, k, h * w, d)  # [B, ch, k, h*w, d]
        d_flat_expanded = d_flat.unsqueeze(-1)  # [B, ch, k, h*w, d, 1]

        # Expand L_chol to match spatial dimensions
        L_chol_expanded = L_chol.unsqueeze(3).expand(-1, -1, -1, h * w, -1, -1)  # [B, ch, k, h*w, d, d]

        # Use Cholesky solve: ||L^{-1} d||^2
        y_flat = torch.linalg.solve_triangular(L_chol_expanded, d_flat_expanded, upper=False).squeeze(-1)
        y = y_flat.reshape(B, ch, k, h, w, d)  # [B, ch, k, h, w, d]

        e = -0.5 * (y * y).sum(dim=-1)  # [B, ch, k, h, w]

        mx = e.max(dim=2, keepdim=True).values
        e = e - mx
        G_sigma = torch.exp(e)
        norm_x = torch.linalg.norm(d_vec[..., :2], dim=-1)

        # Get diagonal elements from spatial block of Cholesky factor for Cauchy component
        L_sp = L_chol[..., :2, :2]  # Extract spatial block
        L_diag = torch.diagonal(L_sp, dim1=-2, dim2=-1)  # Get spatial diagonals [B, ch, k, 2]
        L_diag_norm = torch.norm(L_diag, dim=-1)  # Combine both spatial diagonals
        L_diag_inv_sq = 1.0 / (L_diag_norm.clamp(min=self.min_diag) ** 2)
        L_diag_inv_sq = L_diag_inv_sq.unsqueeze(-1).unsqueeze(-1)

        c_exp = c.unsqueeze(-1).unsqueeze(-1)
        denom = c_exp * L_diag_inv_sq
        denom = denom.clamp(min=self.min_denom)
        C_csigma = 1.0 / (1.0 + (norm_x**2 / denom))
        combined = (alpha.unsqueeze(-1).unsqueeze(-1) * G_sigma) + ((1 - alpha.unsqueeze(-1).unsqueeze(-1)) * C_csigma)
        return combined

    def gaussian_kernel(self, x: torch.Tensor, mu_spatial: torch.Tensor, L_chol_spatial: torch.Tensor) -> torch.Tensor:
        d = x - mu_spatial
        # Use Cholesky solve instead of matrix inverse: ||L^{-1} d||^2
        y = torch.linalg.solve_triangular(L_chol_spatial, d.unsqueeze(-1), upper=False).squeeze(-1)
        e = -0.5 * (y * y).sum(dim=-1)

        mx = e.max(dim=2, keepdim=True).values
        e = e - mx
        return torch.exp(e)

    def cholesky_solve_quadratic(self, cov: torch.Tensor, reg_lambda: torch.Tensor) -> torch.Tensor:
        """Robust Cholesky factorization avoiding eigh in the hot path.

        Uses torch.linalg.cholesky_ex with jitter retry to handle ill-conditioned inputs.
        Only falls back to eigenvalue clamping as a last resort.

        Returns the Cholesky factor L such that cov = L @ L^T.
        """
        # Symmetrize input
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        B, ch, k, d, _ = cov.shape
        eye = torch.eye(d, device=cov.device, dtype=cov.dtype)

        # Base regularization
        lam = F.softplus(self.reg_lambda_param) + 1e-6
        A = cov + lam * eye

        # Try Cholesky with escalating jitter
        jitter = 1e-6
        for _ in range(6):
            L, info = torch.linalg.cholesky_ex(A + jitter * eye)  # returns (L, info)
            if (info == 0).all():  # success
                return L
            jitter *= 10

        # Fallback (slow path): clamp eigenvalues once, then Cholesky
        A_flat = A.reshape(-1, d, d)
        S, U = torch.linalg.eigh(A_flat)  # safe here; we do it rarely
        S = torch.clamp(S, min=1e-6)
        A_spd_flat = U @ torch.diag_embed(S) @ U.transpose(-1, -2)
        A_spd = A_spd_flat.reshape(B, ch, k, d, d)
        return torch.linalg.cholesky(A_spd)

    def svd_cov_inv(self, cov: torch.Tensor, reg_lambda: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
        B, ch, k, d, _ = cov.shape
        m = torch.linalg.eigvalsh(cov).min(dim=-1, keepdim=True).values.unsqueeze(-1)
        eps = torch.clamp(F.softplus(-m), min=reg_lambda.item()) + reg_lambda
        I = torch.eye(d, device=cov.device, dtype=cov.dtype).view(1, 1, 1, d, d).expand(B, ch, k, d, d)
        cov_reg = cov + eps * I
        U, S, Vh = torch.linalg.svd(cov_reg)
        S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))
        return Vh.transpose(-2, -1) @ torch.diag_embed(S_inv) @ U.transpose(-2, -1)

    def forward_spatial(self, h: int, w: int, params: torch.Tensor) -> torch.Tensor:
        B, ch, L = params.shape
        param_count = (
            12
            if self.kernel_type == KernelType.GAUSSIAN_CAUCHY and ch == 1
            else (18 if self.kernel_type == KernelType.GAUSSIAN_CAUCHY else (10 if ch == 1 else 17))
        )
        k = L // param_count
        mu, cov, wt, alp, cst = self.extract_parameters(params, k, ch)
        L_chol = self.cholesky_solve_quadratic(cov, self.reg_lambda_param)
        G = self.grid(h, w, params.device)

        # Assertion for grid correctness
        assert G.shape == (h, w, 2), f"Expected grid shape ({h}, {w}, 2), got {G.shape}"

        G_exp = G.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(B, ch, k, 1, 1, 1)
        mu_full = mu.unsqueeze(3).unsqueeze(4)
        if ch == 1:
            CZ = torch.zeros_like(mu[..., -1:]).unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, h, w, -1)
            X = torch.cat([G_exp, CZ], dim=-1)
        elif ch == 3:
            # For RGB, we need to create a 5D tensor: [x, y, r, g, b]
            # G_exp is [B, ch, k, h, w, 2] (x, y coordinates)
            # We need to add 3 more dimensions for RGB values
            # Create zeros for RGB values at each spatial location
            RGB_zeros = torch.zeros(B, ch, k, h, w, 3, device=G_exp.device, dtype=G_exp.dtype)
            X = torch.cat([G_exp, RGB_zeros], dim=-1)
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            K_out = self.gaussian_cauchy_kernel(X, mu_full, L_chol, alp, cst)
        else:
            mu_sp = mu[..., :2].reshape(B, ch, k, 1, 1, 2)
            L_chol_sp = L_chol[..., :2, :2]
            K_out = self.gaussian_kernel(G_exp, mu_sp, L_chol_sp)
        K_out = K_out * wt.unsqueeze(-1).unsqueeze(-1)

        KS = K_out.sum(dim=2, keepdim=True)
        eps_val = self.get_eps(KS)
        K_norm = K_out / (KS + eps_val)
        out = K_norm.sum(dim=2)
        return out

    def extract_dynamic(self, x: torch.Tensor, cnt: torch.Tensor, p: int) -> torch.Tensor:
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

    def forward_spatial_(self, h: int, w: int, params: torch.Tensor, cnt: torch.Tensor) -> torch.Tensor:
        B, ch, L = params.shape
        param_count = (
            12
            if self.kernel_type == KernelType.GAUSSIAN_CAUCHY and ch == 1
            else (18 if self.kernel_type == KernelType.GAUSSIAN_CAUCHY else (10 if ch == 1 else 17))
        )
        k = L // param_count
        x_dyn = self.extract_dynamic(params, cnt, p=param_count)
        x_flat = x_dyn.view(B, ch, -1)
        mu, cov, wt, alp, cst = self.extract_parameters(x_flat, x_dyn.shape[2], ch)
        d = cov.shape[-1]
        I = torch.eye(d, device=cov.device, dtype=cov.dtype).view(1, 1, 1, d, d)
        eig = torch.linalg.eigvalsh(cov)
        m = eig.min(dim=-1).values[..., None, None]
        eps = F.softplus(-m) + 1e-8
        cov_reg = cov + (1e-6 + eps) * I
        L_chol = torch.linalg.cholesky(cov_reg)

        G = self.grid(h, w, params.device)
        G_exp = G.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(B, ch, x_dyn.shape[2], 1, 1, 1)
        mu_full = mu.unsqueeze(3).unsqueeze(4)
        if ch == 1:
            CZ = torch.zeros_like(mu[..., -1:]).unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, h, w, -1)
            X = torch.cat([G_exp, CZ], dim=-1)
        elif ch == 3:
            CM = mu[..., -3:].unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, h, w, -1)
            X = torch.cat([G_exp, CM], dim=-1)
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            K_out = self.gaussian_cauchy_kernel(X, mu_full, L_chol, alp, cst)
        else:
            mu_sp = mu[..., :2].reshape(B, ch, x_dyn.shape[2], 1, 1, 2)
            L_chol_sp = L_chol[..., :2, :2]
            K_out = self.gaussian_kernel(G_exp, mu_sp, L_chol_sp)
        K_out = K_out * wt.unsqueeze(-1).unsqueeze(-1)

        KS = K_out.sum(dim=2, keepdim=True)
        K_norm = K_out / (KS + 1e-8)
        out = K_norm.sum(dim=2)
        # Remove internal clamp - let the final output stage handle range control
        return out

    def forward(self, h: int, w: int, params: torch.Tensor, cnt: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cnt is None:
            return self.forward_spatial(h, w, params)
        else:
            return self.forward_spatial_(h, w, params, cnt)

    @property
    def d_out(self) -> int:
        return self.kernel

    @property
    def scale_factor(self) -> int:
        return self.cfg.grid_cache.shape[0] if self.cfg.grid_cache is not None else self.cfg.kernel


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
        d_out, params_per_kernel = self.num_params(cfg.DecoderConfig.kernel_type, cfg.d_in, cfg.DecoderConfig.kernel)
        self.snet = DnCNN(
            in_channels=cfg.d_in,
            out_channels=cfg.EncoderConfig.sigma_chn,
            dep=cfg.dep_S,
            noise_avg=cfg.EncoderConfig.noise_avg,
        )
        self.knet = KernelNet(in_nc=cfg.d_in, out_chn=cfg.EncoderConfig.kernel_chn, num_blocks=cfg.dep_K)
        self.encoder = Encoder(cfg.EncoderConfig, cfg.phw, d_in=cfg.d_in, d_out=d_out)
        self.decoder = MoE(cfg.DecoderConfig)
        self.params_per_kernel = params_per_kernel

    def get_eps(self, t: torch.Tensor, factor: float = 1e-3, min_eps: float = 1e-8) -> torch.Tensor:
        return torch.maximum(torch.tensor(min_eps, device=t.device, dtype=t.dtype), factor * t.mean())

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
        blocks = blocks.transpose(1, 2).reshape(B, out_h * out_w, C, block_size, block_size)
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
        window = torch.hann_window(block_size, periodic=False, device=device)
        window2d = window.unsqueeze(0) * window.unsqueeze(1)
        window2d = window2d.view(1, 1, block_size, block_size)
        blocks_weighted = blocks * window2d
        blocks_reshaped = blocks_weighted.reshape(B, -1, C * block_size * block_size).transpose(1, 2)
        recon_padded = F.fold(
            blocks_reshaped,
            output_size=(out_h, out_w),
            kernel_size=block_size,
            stride=step,
        )
        ones = torch.ones_like(blocks)
        ones_weighted = ones * window2d
        ones_reshaped = ones_weighted.reshape(B, -1, C * block_size * block_size).transpose(1, 2)
        weight_sum = F.fold(
            ones_reshaped,
            output_size=(out_h, out_w),
            kernel_size=block_size,
            stride=step,
        )
        eps_ws = self.get_eps(weight_sum)
        recon_norm = recon_padded / weight_sum.clamp_min(eps_ws)
        return recon_norm[:, :, pad : H + pad, pad : W + pad]

    @staticmethod
    def det_split(x: torch.Tensor, num_chunks: int):
        k_eff = min(x.shape[0], num_chunks)
        boundaries = [math.floor(i * x.shape[0] / k_eff) for i in range(k_eff + 1)]
        return [x[boundaries[i] : boundaries[i + 1]] for i in range(k_eff)]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_p, dims = self.extract_blocks(x, self.phw, self.overlap)
        if x_p.ndim == 5:
            x_p = x_p.reshape(-1, *x_p.shape[2:])
        chunks = self.det_split(x_p, self.cfg.num_chunks)

        res = [self.encoder(chunk, self.snet(chunk), self.knet(chunk)) for chunk in chunks]

        gaussians, kinfo, sigma = map(lambda arr: torch.cat(arr, dim=0), zip(*res))

        kinfo = torch.cat([F.softplus(kinfo[:, :2]), kinfo[:, 2:]], dim=1)

        B, L, C, H, W = dims
        sp = self.phw * self.encoder.scale_factor
        dec_chunks = self.det_split(gaussians, self.cfg.num_chunks)

       
        dec_results = []
        for bt in dec_chunks:
            chunk_result = self.decoder(sp, sp, bt)
            dec_results.append(chunk_result)
        dec = torch.cat(dec_results, dim=0)
        rec = self.reconstruct(
            dec,
            (B, L, C, H * self.encoder.scale_factor, W * self.encoder.scale_factor),
            sp,
            self.overlap * self.encoder.scale_factor,
        )
        kinfo_avg = kinfo.view(B, L, -1).mean(dim=1)
        sigma_avg = sigma.view(B, L, 1, 1, 1).mean(dim=1)
        return rec, kinfo_avg, sigma_avg
