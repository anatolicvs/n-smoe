import torch
import torch.nn as nn
import math
from einops import rearrange
import numpy as np

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

def normalization(channels):
    groups = min(32, channels)
    return nn.GroupNorm(groups, channels)

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

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

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.0):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x):
#         return self.net(x)


# class PreNorm(nn.Module):
#     def __init__(self, dim, fn, norm_type='ln'):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn

#     def forward(self, x, *args, **kwargs):
#         x = self.norm(x)
#         return self.fn(x, *args, **kwargs)


# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(
#                 nn.ModuleList(
#                     [
#                         PreNorm(
#                             dim,
#                             nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True),
#                             norm_type='ln'
#                         ),
#                         PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout), norm_type='ln'),
#                     ]
#                 )
#             )

#     def forward(self, x, **kwargs):
#         for attn, ff in self.layers:
#             attn_out, _ = attn(x, x, x)
#             x = x + attn_out
#             x = x + ff(x, **kwargs)
#         return x


# class Encoder(nn.Module):
#     def __init__(self, in_chans, latent_dim, dim, depth, heads, mlp_dim, dropout=0.0):
#         super().__init__()
#         self.in_chans = in_chans
#         self.latent_dim = latent_dim
#         self.dim = dim

#         self.to_patch_embedding = nn.Conv2d(in_chans, dim, kernel_size=4, stride=4)
#         self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
#         self.fc = nn.Linear(dim, latent_dim)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.to_patch_embedding(x)
#         x = rearrange(x, "b c h w -> b (h w) c")
#         x = self.transformer(x)
#         x = self.fc(x)
#         x = rearrange(x, "b n c -> b c n")  # [Batch, Latent Dim, Color]
        
        
#         num_patches = (H // 4) * (W // 4)
#         x = rearrange(x, "b c n -> b n c")
#         x = torch.mean(x, dim=1, keepdim=True)  # Global average pooling to [Batch, 1, Latent Dim]
#         x = x.expand(B, self.in_chans, self.latent_dim)  # Expand to [Batch, Color, Latent Dim]
#         return x


# if __name__ == "__main__":
#     dummy_inputs = [torch.randn(1, 3, 128, 128), torch.randn(1, 3, 256, 256), torch.randn(1, 3, 512, 512)]

#     model = Encoder(
#         in_chans=3,
#         latent_dim=10,
#         dim=64,
#         depth=6,
#         heads=8,
#         mlp_dim=128,
#         dropout=0.1
#     )

#     for dummy_input in dummy_inputs:
#         output = model(dummy_input)
#         print(f"Input shape: {dummy_input.shape} -> Output shape: {output.shape}")

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
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
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

class Encoder(nn.Module):
    def __init__(self, in_chans, latent_dim, embed_dim, depth, heads, mlp_dim, dropout=0.0, patch_size=4):
        super().__init__()
        self.in_chans = in_chans
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.transformer = Transformer(embed_dim, depth, heads, mlp_dim, dropout)
    
        self.fc = nn.Sequential(
                nn.Linear(embed_dim, in_chans*latent_dim),
                nn.ReLU())
    
    def _interpolate(self, x, scale_factor):
        B, C, H, W = x.size() 

        target_h = int(H * scale_factor)  
        target_w = int(W * scale_factor)  
        target_size = (target_h, target_w)  

        x_resized = self.resizer(x, target_size)  

        return x_resized

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.fc(x)
        x = rearrange(x, "b n c -> b c n")  # [Batch, Latent Dim, Color]

        x = torch.mean(x, dim=2, keepdim=True)
        x = rearrange(x, 'b (c latent) 1 -> b c latent', c=self.in_chans, latent=self.latent_dim)
        return x

if __name__ == "__main__":
    dummy_inputs = [torch.randn(1, 1, 128, 128), torch.randn(1, 1, 256, 256), torch.randn(1, 1, 512, 512)]

    model = Encoder(
        in_chans=1,
        latent_dim=78,
        embed_dim=64,
        depth=6,
        heads=8,
        mlp_dim=128,
        dropout=0.1,
        patch_size=4
    )

    for dummy_input in dummy_inputs:
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
