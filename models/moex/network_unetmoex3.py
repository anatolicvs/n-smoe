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
from utils_n.nn import GroupNorm32, avg_pool_nd, checkpoint, conv_nd, zero_module


backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

OLD_GPU = True
USE_FLASH_ATTN = False
MATH_KERNEL_ON = True

T = TypeVar("T")


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
        base_resize_method="bilinear",
        kernel_size=5,
        stddev=1.0,
        num_layers=2,
        avg_pool=False,
        init_weights=None,
        dtype=torch.float32,
    ):
        super(MullerResizer, self).__init__()
        self.d_in: int = d_in
        self.kernel_size: int = kernel_size
        self.stddev: float = stddev
        self.num_layers: int = num_layers
        self.avg_pool: bool = avg_pool
        self.dtype: torch.dtype = dtype

        interpolation_methods: dict[str, str] = {
            "bilinear": "bilinear",
            "nearest": "nearest",
            "bicubic": "bicubic",
        }
        self.interpolation_method: str = interpolation_methods.get(
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

    def create_gaussian_kernel(self, kernel_size, stddev) -> torch.Tensor:
        t = torch.arange(kernel_size, dtype=self.dtype) - (kernel_size - 1) / 2
        gaussian_kernel: torch.Tensor = torch.exp(-t.pow(2) / (2 * stddev**2))
        gaussian_kernel /= gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(
            1, 1, kernel_size, 1
        ) * gaussian_kernel.view(1, 1, 1, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.d_in, 1, 1, 1)
        return gaussian_kernel

    def _apply_gaussian_blur(self, input) -> torch.Tensor:
        padding: int = self.kernel_size // 2
        x: torch.Tensor = F.pad(
            input, (padding, padding, padding, padding), mode="reflect"
        )

        gaussian_kernel: torch.Tensor = self.gaussian_kernel.to(x.device)
        return F.conv2d(x, gaussian_kernel, groups=self.d_in)

    def forward(self, input_tensor, target_size):
        x = input_tensor.to(dtype=self.dtype)
        if self.avg_pool:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        net = F.interpolate(
            x, size=target_size, mode=self.interpolation_method, align_corners=False
        )

        for weight, bias in zip(self.weights, self.biases):
            blurred: torch.Tensor = self._apply_gaussian_blur(x)
            residual = blurred - x
            resized_residual = F.interpolate(
                residual,
                size=target_size,
                mode=self.interpolation_method,
                align_corners=False,
            )
            net = net + torch.tanh(weight * resized_residual + bias)
            x: torch.Tensor = blurred

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

        self.input_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    conv_nd(cfg.dims, self.d_in, cfg.model_channels, 3, padding=1)
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
                int(((self.phw * (cfg.scale_factor / 2)) ** 2) // (ds)),
                ch,
                cfg.num_heads,
                int(self.d_in * self.latent),
            ),
        )

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
        target_size: Tuple[int, int] = (target_h, target_w)
        x_resized = self.resizer(x, target_size)
        return x_resized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._interpolate(x, self.cfg.scale_factor)

        h: torch.Tensor = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)

        h = self.middle_block(h)

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
    mu: Optional[torch.Tensor] = None
    sigma: Optional[torch.Tensor] = None
    w: Optional[torch.Tensor] = None
    theta: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None


class MoE(Backbone[MoEConfig]):
    def __init__(self, cfg: MoEConfig) -> None:
        super(MoE, self).__init__(cfg)
        self.kernel: int = cfg.kernel
        self.alpha: float = cfg.sharpening_factor

    def grid(self, height, width) -> torch.Tensor:
        xx: torch.Tensor = torch.linspace(0.0, 1.0, width)
        yy: torch.Tensor = torch.linspace(0.0, 1.0, height)
        grid_x, grid_y = torch.meshgrid(xx, yy, indexing="ij")
        grid: torch.Tensor = torch.stack((grid_x, grid_y), -1).float()
        return grid

    def cov_mat_2d(self, scale: torch.Tensor, theta: torch.Tensor, epsilon=1e-4):
        R: torch.Tensor = self.ang_to_rot_mat(theta)
        S: torch.Tensor = torch.diag_embed(scale) + epsilon * torch.eye(
            scale.size(-1), device=scale.device
        )

        RS: torch.Tensor = R @ S
        Sigma: torch.Tensor = RS @ RS.transpose(-2, -1)
        return Sigma

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        cos_theta: torch.Tensor = torch.cos(theta).unsqueeze(-1)
        sin_theta: torch.Tensor = torch.sin(theta).unsqueeze(-1)
        R: torch.Tensor = torch.cat(
            [cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1
        )
        return R.view(*theta.shape, 2, 2)

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Gaussians:
        # v2 - models after Oct 6 uses this version
        mu_x: torch.Tensor = p[:, :, :k].reshape(-1, ch, k, 1)
        mu_y: torch.Tensor = p[:, :, k : 2 * k].reshape(-1, ch, k, 1)
        mu: torch.Tensor = torch.cat((mu_x, mu_y), dim=-1).view(-1, ch, k, 2)

        scale_idx: int = 3 * k
        scale: torch.Tensor = p[:, :, scale_idx : scale_idx + 2 * k].reshape(
            -1, p.shape[1], k, 2
        )

        scale = F.softplus(scale) + 1e-8  # Ensure positive scales

        rot_idx: int = scale_idx + 2 * k
        theta: torch.Tensor = p[:, :, rot_idx : rot_idx + k].reshape(-1, p.shape[1], k)
        theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi  # Constrain to [-π, π]

        cov_matrix: torch.Tensor = self.cov_mat_2d(scale, theta)
        cov_matrix = torch.mul(cov_matrix, self.alpha)

        w: torch.Tensor = p[:, :, 2 * k : 3 * k].reshape(-1, ch, k)

        return Gaussians(mu, cov_matrix, w, theta, scale)

    def extract_parameters_v1(self, p: torch.Tensor, k: int, ch: int) -> Gaussians:
        # v2
        mu_x: torch.Tensor = p[:, :, :k].reshape(-1, ch, k, 1)
        mu_y: torch.Tensor = p[:, :, k : 2 * k].reshape(-1, ch, k, 1)
        mu: torch.Tensor = torch.cat((mu_x, mu_y), dim=-1).view(-1, ch, k, 2)

        scale_idx: int = 3 * k
        scale: torch.Tensor = p[:, :, scale_idx : scale_idx + 2 * k].reshape(
            -1, p.shape[1], k, 2
        )

        scale = F.softplus(scale) + 1e-6  # Ensure positive scales

        rot_idx: int = scale_idx + 2 * k
        theta: torch.Tensor = p[:, :, rot_idx : rot_idx + k].reshape(-1, p.shape[1], k)
        theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi  # Constrain to [-π, π]

        w: torch.Tensor = p[:, :, 2 * k : 3 * k].reshape(-1, ch, k)

        return Gaussians(mu, None, w, theta, scale)

    def extract_parameters_v0(self, p: torch.Tensor, k: int, ch: int) -> Gaussians:
        # v0, models before Oct 6 uses this version
        mu_x: torch.Tensor = p[:, :, :k].reshape(-1, ch, k, 1)
        mu_y: torch.Tensor = p[:, :, k : 2 * k].reshape(-1, ch, k, 1)
        mu: torch.Tensor = torch.cat((mu_x, mu_y), dim=-1).view(-1, ch, k, 2)

        scale_idx: int = 3 * k
        scale: torch.Tensor = p[:, :, scale_idx : scale_idx + 2 * k].reshape(
            -1, p.shape[1], k, 2
        )

        rot_idx: int = scale_idx + 2 * k
        theta: torch.Tensor = p[:, :, rot_idx : rot_idx + k].reshape(-1, p.shape[1], k)

        cov_matrix: torch.Tensor = self.cov_mat_2d(scale, theta)
        cov_matrix = torch.mul(cov_matrix, self.alpha)

        w: torch.Tensor = p[:, :, 2 * k : 3 * k].reshape(-1, ch, k)

        return Gaussians(mu, cov_matrix, w, theta, scale)

    def forward_v0(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, _ = params.shape
        gauss: Gaussians = self.extract_parameters(params, self.kernel, num_channels)

        grid: torch.Tensor = self.grid(height, width).to(params.device)
        grid_expanded: torch.Tensor = (
            grid.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, num_channels, self.kernel, height, width, 2)
        )

        mu_expanded: torch.Tensor = (
            gauss.mu.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, height, width, -1)
        )
        x_sub_mu: torch.Tensor = grid_expanded - mu_expanded

        Sigma_expanded: torch.Tensor = (
            gauss.sigma.unsqueeze(3)
            .unsqueeze(4)
            .expand(-1, -1, -1, height, width, -1, -1)
        )

        x_sub_mu_t: torch.Tensor = x_sub_mu.unsqueeze(-1)

        exp_terms: torch.Tensor = -0.5 * torch.einsum(
            "bnkhwli,bnkhwlj,bnkhwmi->bnkhw",
            x_sub_mu_t,
            Sigma_expanded,
            x_sub_mu_t,
        ).squeeze(-1)

        e: torch.Tensor = torch.exp(exp_terms)

        max_e: torch.Tensor = torch.max(e, dim=2, keepdim=True)[0]
        log_sum_exp_e: torch.Tensor = max_e + torch.log(
            torch.sum(torch.exp(e - max_e), dim=2, keepdim=True)
        )
        e_norm: torch.Tensor = torch.exp(e - log_sum_exp_e)

        y_hat: torch.Tensor = torch.sum(
            e_norm * gauss.w.unsqueeze(-1).unsqueeze(-1), dim=2
        )
        y_hat = torch.clamp(y_hat, min=0, max=1)

        return y_hat

    def forward_v1(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        # Sigma_inv added.
        batch_size, num_channels, _ = params.shape
        gauss: Gaussians = self.extract_parameters(params, self.kernel, num_channels)

        grid: torch.Tensor = self.grid(height, width).to(params.device)
        grid_expanded: torch.Tensor = (
            grid.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, num_channels, self.kernel, height, width, 2)
        )

        mu_expanded: torch.Tensor = (
            gauss.mu.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, height, width, -1)
        )
        x_sub_mu: torch.Tensor = grid_expanded - mu_expanded

        S_inv = torch.diag_embed(1 / gauss.scale)  # [B, N, K, 2, 2]

        R = self.ang_to_rot_mat(gauss.theta)  # [B, N, K, 2, 2]
        R_T = R.transpose(-2, -1)

        Sigma_inv = R @ S_inv @ R_T  # [B, N, K, 2, 2]
        Sigma_inv_expanded = (
            Sigma_inv.unsqueeze(3)
            .unsqueeze(4)
            .expand(-1, -1, -1, height, width, -1, -1)
        )

        x_sub_mu_t: torch.Tensor = x_sub_mu.unsqueeze(-1)

        exp_terms: torch.Tensor = -0.5 * torch.einsum(
            "bnkhwli,bnkhwlj,bnkhwmi->bnkhw",
            x_sub_mu_t,
            Sigma_inv_expanded,  # Sigma_expanded,
            x_sub_mu_t,
        ).squeeze(-1)

        e: torch.Tensor = torch.exp(exp_terms)

        max_e: torch.Tensor = torch.max(e, dim=2, keepdim=True)[0]
        log_sum_exp_e: torch.Tensor = max_e + torch.log(
            torch.sum(torch.exp(e - max_e), dim=2, keepdim=True)
        )
        e_norm: torch.Tensor = torch.exp(e - log_sum_exp_e)

        y_hat: torch.Tensor = torch.sum(
            e_norm * gauss.w.unsqueeze(-1).unsqueeze(-1), dim=2
        )
        y_hat = torch.clamp(y_hat, min=0, max=1)

        return y_hat

    def forward(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        # v2 - models after Oct 6 uses this version 
        batch_size, num_channels, _ = params.shape
        gauss = self.extract_parameters(
            params, self.kernel, num_channels
        )  # Gaussians(mu, sigma, w, scale, theta)

        grid = self.grid(height, width).to(params.device)  # [H, W, 2]
        grid_expanded = (
            grid.unsqueeze(0).unsqueeze(0).unsqueeze(2)
        )  # [1, 1, 1, H, W, 2]
        grid_expanded = grid_expanded.expand(
            batch_size, num_channels, self.kernel, height, width, 2
        )  # [B, N, K, H, W, 2]

        mu_expanded = gauss.mu.unsqueeze(3).unsqueeze(4)  # [B, N, K, 1, 1, 2]
        mu_expanded = mu_expanded.expand(
            -1, -1, -1, height, width, -1
        )  # [B, N, K, H, W, 2]

        x_sub_mu = grid_expanded - mu_expanded  # [B, N, K, H, W, 2]
        x_sub_mu_t = x_sub_mu.unsqueeze(-1)  # [B, N, K, H, W, 2, 1]

        Sigma = gauss.sigma  # [B, N, K, 2, 2]
        try:
            Sigma_inv = torch.linalg.inv(Sigma)  # [B, N, K, 2, 2]
        except RuntimeError as e:
            epsilon = 1e-6
            Sigma += epsilon * torch.eye(2, device=Sigma.device).unsqueeze(0).unsqueeze(
                0
            ).unsqueeze(0)
            Sigma_inv = torch.linalg.inv(Sigma)

        Sigma_inv_expanded = Sigma_inv.unsqueeze(3).unsqueeze(
            4
        )  # [B, N, K, 1, 1, 2, 2]
        Sigma_inv_expanded = Sigma_inv_expanded.expand(
            -1, -1, -1, height, width, -1, -1
        )  # [B, N, K, H, W, 2, 2]

        exp_terms = -0.5 * torch.einsum(
            "bnkhwli,bnkhwlj,bnkhwmi->bnkhw",
            x_sub_mu_t,  # [B, N, K, H, W, 2, 1]
            Sigma_inv_expanded,  # [B, N, K, H, W, 2, 2]
            x_sub_mu_t,  # [B, N, K, H, W, 2, 1]
        ).squeeze(
            -1
        )  # [B, N, K, H, W]

        max_exp_terms = torch.max(
            exp_terms, dim=2, keepdim=True
        ).values  # [B, N, 1, H, W]
        exp_terms = exp_terms - max_exp_terms  # [B, N, K, H, W]
        e = torch.exp(exp_terms)  # [B, N, K, H, W]

        sum_e = torch.sum(e, dim=2, keepdim=True)  # [B, N, 1, H, W]
        e_norm = e / (sum_e + 1e-8)  # [B, N, K, H, W]

        w = F.softmax(gauss.w, dim=2).unsqueeze(-1).unsqueeze(-1)  # [B, N, K, 1, 1]
        y_hat = torch.sum(e_norm * w, dim=2)  # [B, N, H, W]

        y_hat = torch.clamp(y_hat, min=0.0, max=1.0)  # [B, N, H, W]

        return y_hat

    def forward_v3(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        #v3 - models before Oct 6 uses this version
        batch_size, num_channels, _ = params.shape
        gauss = self.extract_parameters(
            params, self.kernel, num_channels
        )  # Gaussians(mu, sigma, w, scale, theta)

        grid = self.grid(height, width).to(params.device)  # [H, W, 2]
        grid_expanded = (
            grid.unsqueeze(0).unsqueeze(0).unsqueeze(2)
        )  # [1, 1, 1, H, W, 2]
        grid_expanded = grid_expanded.expand(
            batch_size, num_channels, self.kernel, height, width, 2
        )  # [B, N, K, H, W, 2]

        mu_expanded = gauss.mu.unsqueeze(3).unsqueeze(4)  # [B, N, K, 1, 1, 2]
        mu_expanded = mu_expanded.expand(
            -1, -1, -1, height, width, -1
        )  # [B, N, K, H, W, 2]

        x_sub_mu = grid_expanded - mu_expanded  # [B, N, K, H, W, 2]
        x_sub_mu_t = x_sub_mu.unsqueeze(-1)  # [B, N, K, H, W, 2, 1]

        Sigma = gauss.sigma  # [B, N, K, 2, 2]

        try:
            L = torch.linalg.cholesky(Sigma)  # [B, N, K, 2, 2]
            Sigma_inv = torch.cholesky_inverse(L)  # [B, N, K, 2, 2]
        except RuntimeError as e:
            epsilon = 1e-6
            Sigma += epsilon * torch.eye(2, device=Sigma.device).unsqueeze(0).unsqueeze(
                0
            ).unsqueeze(0)
            Sigma_inv = torch.linalg.inv(Sigma)

        Sigma_inv_expanded = Sigma_inv.unsqueeze(3).unsqueeze(
            4
        )  # [B, N, K, 1, 1, 2, 2]
        Sigma_inv_expanded = Sigma_inv_expanded.expand(
            -1, -1, -1, height, width, -1, -1
        )  # [B, N, K, H, W, 2, 2]

        exp_terms = -0.5 * torch.einsum(
            "bnkhwli,bnkhwlj,bnkhwmi->bnkhw",
            x_sub_mu_t,  # [B, N, K, H, W, 2, 1]
            Sigma_inv_expanded,  # [B, N, K, H, W, 2, 2]
            x_sub_mu_t,  # [B, N, K, H, W, 2, 1]
        ).squeeze(
            -1
        )  # [B, N, K, H, W]

        max_exp_terms = torch.max(
            exp_terms, dim=2, keepdim=True
        ).values  # [B, N, 1, H, W]
        exp_terms = exp_terms - max_exp_terms  # [B, N, K, H, W]
        e = torch.exp(exp_terms)  # [B, N, K, H, W]

        sum_e = torch.sum(e, dim=2, keepdim=True)  # [B, N, 1, H, W]
        e_norm = e / (sum_e + 1e-8)  # [B, N, K, H, W]

        w = F.softmax(gauss.w, dim=2).unsqueeze(-1).unsqueeze(-1)  # [B, N, K, 1, 1]
        y_hat = torch.sum(e_norm * w, dim=2)  # [B, N, H, W]

        y_hat = torch.clamp(y_hat, min=0.0, max=1.0)  # [B, N, H, W]

        return y_hat


class MoE_v1(nn.Module):
    def __init__(self, cfg: MoEConfig) -> None:
        super(MoE, self).__init__()
        self.kernel = cfg.kernel
        self.alpha = cfg.sharpening_factor

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        cos_theta = torch.cos(theta).unsqueeze(-1)
        sin_theta = torch.sin(theta).unsqueeze(-1)
        R = torch.cat([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1)
        return R.view(*theta.shape, 2, 2)

    def cov_mat_2d(self, scale: torch.Tensor, theta: torch.Tensor, epsilon=1e-6):
        R = self.ang_to_rot_mat(theta)
        S = torch.diag_embed(scale + epsilon)
        Sigma = R @ S @ S.transpose(-2, -1) @ R.transpose(-2, -1)
        return Sigma

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Gaussians:
        B = p.size(0)
        mu_x = p[:, :, :k]
        mu_y = p[:, :, k : 2 * k]
        mu = torch.stack((mu_x, mu_y), dim=-1)  # [B, C, K, 2]

        w = p[:, :, 2 * k : 3 * k]  # [B, C, K]

        scale_idx = 3 * k
        scale = p[:, :, scale_idx : scale_idx + 2 * k].reshape(B, ch, k, 2)
        # scale = F.softplus(scale) + 1e-6
        scale = scale / self.alpha**0.5  # Apply sharpening factor to scales

        rot_idx = scale_idx + 2 * k
        theta = p[:, :, rot_idx : rot_idx + k]
        theta = theta % (2 * torch.pi)

        sigma = self.cov_mat_2d(scale, theta)
        return Gaussians(mu=mu, sigma=sigma, w=w)

    def grid(self, height: int, width: int) -> torch.Tensor:
        xx = torch.linspace(0.0, 1.0, width)
        yy = torch.linspace(0.0, 1.0, height)
        grid_x, grid_y = torch.meshgrid(xx, yy, indexing="ij")
        grid = torch.stack((grid_x, grid_y), -1).float()  # [H, W, 2]
        return grid

    def forward(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        B, C, _ = params.shape
        K = self.kernel

        gauss = self.extract_parameters(params, K, C)

        grid = self.grid(height, width).to(params.device)  # [H, W, 2]

        grid_expanded = (
            grid.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(B, C, K, height, width, 2)
        )  # [B, C, K, H, W, 2]

        mu_expanded = (
            gauss.mu.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, height, width, -1)
        )  # [B, C, K, H, W, 2]

        x_sub_mu = grid_expanded - mu_expanded  # [B, C, K, H, W, 2]
        x_sub_mu_flat = x_sub_mu.view(
            B * C * K, height * width, 2, 1
        )  # [BCK, HW, 2, 1]

        sigma = gauss.sigma.view(B * C * K, 2, 2)  # [BCK, 2, 2]

        try:
            L = torch.linalg.cholesky(sigma)  # [BCK, 2, 2]
        except RuntimeError:
            epsilon = 1e-6
            sigma += epsilon * torch.eye(2, device=sigma.device).unsqueeze(0)
            L = torch.linalg.cholesky(sigma)

        x_sub_mu_flat_squeezed = x_sub_mu_flat.squeeze(-1)  # [BCK, HW, 2]

        x_sub_mu_flat_t = x_sub_mu_flat_squeezed.transpose(0, 1)  # [HW, BCK, 2]

        L_expanded = L.unsqueeze(0).expand(
            height * width, -1, -1, -1
        )  # [HW, BCK, 2, 2]

        y = torch.cholesky_solve(
            x_sub_mu_flat_t.unsqueeze(-1), L_expanded, upper=False
        )  # [HW, BCK, 2, 1]

        y = y.squeeze(-1).transpose(0, 1)  # [BCK, HW, 2]

        exp_terms_flat = -0.5 * torch.sum(
            x_sub_mu_flat_squeezed * y, dim=-1
        )  # [BCK, HW]

        exp_terms = exp_terms_flat.view(B, C, K, height, width)  # [B, C, K, H, W]

        e = torch.exp(exp_terms)  # [B, C, K, H, W]

        w = F.softmax(gauss.w, dim=2).unsqueeze(-1).unsqueeze(-1)  # [B, C, K, 1, 1]
        weighted_e = e * w  # [B, C, K, H, W]

        y_hat = weighted_e.sum(dim=2)  # [B, C, H, W]

        y_hat = y_hat / (y_hat.amax(dim=[-1, -2], keepdim=True) + 1e-8)
        y_hat = torch.clamp(y_hat, min=0.0, max=1.0)

        return y_hat  # [B, C, H, W]


@dataclass
class AutoencoderConfig:
    EncoderConfig: EncoderConfig
    DecoderConfig: MoEConfig
    d_in: int
    d_out: int
    phw: int = 32
    overlap: int = 24


class Autoencoder(Backbone[AutoencoderConfig]):
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__(cfg)
        self.phw: int = cfg.phw
        self.overlap: int = cfg.overlap
        self.encoder = Encoder(
            cfg.EncoderConfig, cfg.phw, d_in=cfg.d_in, d_out=cfg.d_out
        )
        self.decoder = MoE(cfg.DecoderConfig)
        self.scale_factor: int = 1

    @staticmethod
    def reconstruct(
        blocks: torch.Tensor,
        original_dims: Tuple[int, int, int, int],
        block_size: int,
        overlap: int,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = original_dims
        step: int = block_size - overlap
        device: torch.device = blocks.device

        recon_images: torch.Tensor = torch.zeros(
            batch_size, num_channels, height, width
        ).to(device)
        count_matrix: torch.Tensor = torch.zeros(
            batch_size, num_channels, height, width
        ).to(device)

        num_blocks_per_row: int = (width - block_size) // step + 1
        num_blocks_per_column: int = (height - block_size) // step + 1
        num_blocks_per_image: int = num_blocks_per_row * num_blocks_per_column

        for b in range(batch_size):
            idx_start: int = b * num_blocks_per_image
            current_blocks: torch.Tensor = blocks[
                idx_start : idx_start + num_blocks_per_image
            ]
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
            device: int = torch.cuda.current_device()
            torch.cuda.set_device(device)
            tot_mem = torch.cuda.get_device_properties(device).total_memory
            used_mem: int = torch.cuda.memory_reserved(device)
            free_mem = tot_mem - used_mem

            thresholds: list[float] = [0.7, 0.5, 0.3, 0.1]
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

        es: int = x.element_size()
        ml = self.mem_lim()
        bm: int = x.shape[1:].numel() * es
        mx_bs = ml // bm
        n: int = max(1, min(x.shape[0] // 1024, mx_bs))
        cs: int = (x.shape[0] + n - 1) // n

        b = torch.split(x, cs)

        enc: torch.Tensor = torch.cat([self.encoder(bt) for bt in b], dim=0)

        B, C, H, W = s
        sp: int = self.phw * self.encoder.scale_factor

        dec: torch.Tensor = torch.cat(
            [self.decoder(sp, sp, bt) for bt in torch.split(enc, cs)], dim=0
        )

        y: torch.Tensor = self.reconstruct(
            dec,
            (B, C, H * self.encoder.scale_factor, W * self.encoder.scale_factor),
            sp,
            self.overlap * self.encoder.scale_factor,
        )
        return y
