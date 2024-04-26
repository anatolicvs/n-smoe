
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.fft

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def checkpoint(func, inputs, params, flag=True):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()

def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
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
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
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
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])

def fft2d(input, gamma=0.1):
    temp = input.permute(0, 2, 3, 1)
    fft = torch.fft.fft2(torch.complex(temp, torch.zeros_like(temp)))
    absfft = torch.pow(torch.abs(fft) + 1e-8, gamma)
    output = absfft.permute(0, 3, 1, 2)
    return output

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
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor.

        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
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

class Encoder(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        scale_factor=2,
        pool="spatial_v2",
        num_layers=8,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.scale_factor = scale_factor

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [nn.Sequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = nn.Sequential(
            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

        self.resizer = MullerResizer(
            base_resize_method='bicubic', kernel_size=5, stddev=1.0, num_layers=num_layers,
            dtype=torch.float32
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
    
    def process_component(self, component):
        h = component.type(self.dtype)
        results = []
        for module in self.input_blocks:
            h = module(h)
            if self.pool.startswith("spatial"):
                results.append(h.type(component.dtype).mean(dim=(2, 3)))
        return 
    
    def combine_components(self, ll_outputs, lh_outputs, hl_outputs, hh_outputs):
        combined = []
        for i in range(len(ll_outputs)):
            combined_layer = torch.cat([ll_outputs[i], lh_outputs[i], hl_outputs[i], hh_outputs[i]], dim=1)
            combined.append(combined_layer)
        return combined
    
    def _interpolate(self, x, scale_factor):
        B, C, H, W = x.size() 

        target_h = int(H * scale_factor)  
        target_w = int(W * scale_factor)  
        target_size = (target_h, target_w)  

        x_resized = self.resizer(x, target_size)  

        return x_resized  
    
    def forward(self, x):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        results = []
        x = self._interpolate(x, self.scale_factor)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = torch.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)

class MoE(nn.Module):
    def __init__(self, in_channels=3, num_mixtures=4, kernel=4):
        super(MoE, self).__init__()
        self.ch = in_channels
        self.kernel = kernel
        self.num_mixtures = num_mixtures

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

        raw_Σ = params[:, :, 3 * self.kernel : 3 * self.kernel + self.kernel * 2 * 2]
        raw_Σ = raw_Σ.reshape(-1, self.kernel, 2, 2)

        Σ_lower_tri = torch.tril(raw_Σ)
        Σ = Σ_lower_tri @ Σ_lower_tri.transpose(-2, -1)

        raw_w = params[:, :, 2 * self.kernel : 3 * self.kernel].reshape(-1, self.kernel)
        w = F.softmax(raw_w, dim=1)

        grid = self.grid(height, width).to(params.device)
        μ = μ.unsqueeze(dim=2)
        grid_expand = grid.unsqueeze(0).unsqueeze(0)
        x = grid_expand.expand(μ.shape[0], μ.shape[1], -1, -1)
        x_sub_μ = (x.float() - μ.float()).unsqueeze(-1)

        e = torch.exp(-0.5 * torch.einsum("abcli,ablm,abcnj->abc", x_sub_μ, Σ, x_sub_μ))

        g = torch.sum(e, dim=1, keepdim=True)
        g_max = torch.clamp(g, min=1e-8)
        e_norm = e / g_max

        y_hat = torch.sum(e_norm * w.unsqueeze(-1), dim=1)
        y_hat = torch.clamp(y_hat, min=0, max=1)

        y_hat = y_hat.view(-1, self.ch, height, width)
        return y_hat


# class MoE(nn.Module):
#     def __init__(
#         self,
#         in_channels=3,  # Default to 3 for RGB; set to 4 for RGBA or 1 for grayscale
#         num_mixtures=4,
#         kernel=4,
#         sharpening_factor=1,
#         clip_value=5,
#     ):
#         super(MoE, self).__init__()

#         self.ch = in_channels
#         self.kernel = kernel
#         self.num_mixtures = num_mixtures
#         self.clip_value = clip_value

#         self.α = sharpening_factor

#     def grid(self, height, width):
#         xx = torch.linspace(0.0, 1.0, width)
#         yy = torch.linspace(0.0, 1.0, height)
#         grid_x, grid_y = torch.meshgrid(xx, yy, indexing="ij")
#         grid = torch.stack((grid_x, grid_y), 2).float()
#         return grid.reshape(height * width, 2)

#     @staticmethod
#     def _soft_clipping(x, beta=10):
#         return 1 / (1 + torch.exp(-beta * (x - 0.5)))

#     def forward(self, height, width, params):
        
#         μ_x = params[:, :, : self.kernel].reshape(-1, self.kernel, 1)
#         μ_y = params[:, :, self.kernel : 2 * self.kernel].reshape(-1, self.kernel, 1)
#         μ = torch.cat((μ_x, μ_y), 2).view(-1, self.kernel, 2)
#         Σ = params[
#             :, :, 3 * self.kernel : 3 * self.kernel + self.kernel * 2 * 2
#         ].reshape(-1, self.kernel, 2, 2)
#         w = params[:, :, 2 * self.kernel : 3 * self.kernel].reshape(-1, self.kernel)

#         Σ = torch.tril(Σ)
#         Σ = torch.mul(Σ, self.α)

#         grid = self.grid(height, width).to(params.device)
#         μ = μ.unsqueeze(dim=2)
#         grid_expand_dim = torch.unsqueeze(torch.unsqueeze(grid, dim=0), dim=0)
#         x = torch.tile(grid_expand_dim, (μ.shape[0], μ.shape[1], 1, 1))
#         x_sub_μ = torch.unsqueeze(x.float() - μ.float(), dim=-1)

#         e = torch.exp(
#             torch.negative(
#                 0.5 * torch.einsum("abcli,ablm,abnm,abcnj->abc", x_sub_μ, Σ, Σ, x_sub_μ)
#             )
#         )

#         g = torch.sum(e, dim=1, keepdim=True)
#         g_max = torch.max(torch.tensor(10e-8), g)
#         e_norm = torch.divide(e, g_max)

#         y_hat = torch.sum(e_norm * torch.unsqueeze(w, dim=-1), dim=1)
#         y_hat = torch.clamp(y_hat, min=0, max=1)

        
#         y_hat = y_hat.view(-1, self.ch, height, width)

#         return y_hat


# class MoE(nn.Module):
#     def __init__(self, in_channels=3, num_mixtures=4, sharpening_factor=1):
#         super().__init__()
#         self.ch = in_channels
#         self.num_mixtures = num_mixtures
#         self.α = sharpening_factor
        
#     def grid(self, height, width):
        
#         xx = torch.linspace(0, 1, width, device='cuda')
#         yy = torch.linspace(0, 1, height, device='cuda')
#         grid_x, grid_y = torch.meshgrid(xx, yy, indexing='xy')
#         grid = torch.stack((grid_x, grid_y), dim=-1)
#         return grid.view(-1, 2) 

#     def forward(self, height, width, params):
#         batch_size = params.shape[0]
#         param_per_mixture = 2 + 2*2 + 1  
#         outputs = []
#         grid = self.grid(height, width)
#         for c in range(self.ch):  
            
#             channel_params = params[:, c, :].view(batch_size, self.num_mixtures, param_per_mixture)
#             means = channel_params[:, :, :2]
#             raw_covariances = channel_params[:, :, 2:6].reshape(batch_size, self.num_mixtures, 2, 2)
#             weights = F.softmax(channel_params[:, :, 6:], dim=1)

#             L = torch.tril(raw_covariances)
#             L = L * self.α
#             covariances = L @ L.transpose(-2, -1)

#             diff = grid.unsqueeze(0).unsqueeze(1) - means.unsqueeze(2)
#             mahalanobis_dist = torch.einsum('bnij,bnjk,bnik->bni', diff, covariances.inverse(), diff)
#             exponent = torch.exp(-0.5 * mahalanobis_dist)

#             weighted_sum = torch.sum(weights * exponent, dim=1)
#             output = weighted_sum.view(batch_size, 1, height, width)
#             outputs.append(output)

#         final_output = torch.cat(outputs, dim=1)
#         return final_output.clamp(0, 1)

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
        sigma = max(self.stddev, 0.1)  # Ensure sigma is not too small
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
            net += torch.tanh(scaled_residual.clamp(min=-3, max=3))  # Clamping to prevent extreme values
            inputs = blurred
        return net

class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels:int=3,
        latent_dim:int=28,
        num_mixtures:int=4,
        sharpening_factor:int=1,
        stride:int=16,
        phw:int=64,
        dropout:int=0.1,
        image_size:int=64,
        model_channels:int=32,
        num_res_blocks:int=64,
        attention_resolutions:list=[16, 8, 4],
        channel_mult:tuple=(1, 2, 4, 8),
        num_heads:int=4,
        num_head_channels:int=-1,
        use_checkpoint:bool=True,
        pool:str="spatial_v2",
        scale_factor:int=1,
        num_layers:int=8
    ):
        super().__init__()

        self.phw = phw
        self.stride = stride
        self.latent_dim = latent_dim
        self.scale_factor = scale_factor

        self.encoder = Encoder(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=latent_dim*in_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            pool=pool,
            use_checkpoint=use_checkpoint,
            scale_factor=scale_factor,
            num_layers=num_layers
        )

        self.decoder = MoE(
            in_channels=in_channels,
            kernel=num_mixtures,
            num_mixtures=num_mixtures,
            # sharpening_factor=sharpening_factor
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