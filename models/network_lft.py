import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class LFT(nn.Module):
    def __init__(self, args):
        super(LFT, self).__init__()
        channels = args['channels']  
        self.channels = channels
        self.angRes = args['angRes']
        self.factor = args['scale_factor']
        layer_num = 4

        self.pos_encoding = PositionEncoding(temperature=10000)
        self.MHSA_params = {'num_heads': 8, 'dropout': 0.}

        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.altblock = self.make_layer(layer_num=layer_num)

        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels*self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.resizer = MullerResizer(
            base_resize_method='bicubic', kernel_size=5, stddev=1.0, num_layers=8,
            dtype=torch.float32
        )

    
    def make_layer(self, layer_num):
        layers = []
        for i in range(layer_num):
            layers.append(AltFilter(self.angRes, self.channels, self.MHSA_params))
        return nn.Sequential(*layers)

    def _interpolate(self, x, angRes, scale_factor):
        [B, _, H, W] = x.size()
        h = H // angRes
        w = W // angRes
        x_upscale = x.view(B, 1, angRes, h, angRes, w)
        x_upscale = x_upscale.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * angRes ** 2, 1, h, w)
        
        target_h = h * scale_factor
        target_w = w * scale_factor
        target_size = (target_h, target_w)

        x_upscale = self.resizer(x_upscale,target_size)
        x_upscale = x_upscale.view(B, angRes, angRes, 1, h * scale_factor, w * scale_factor)
        x_upscale = x_upscale.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, H * scale_factor, W * scale_factor)
        # [B, 1, A*h*S, A*w*S]
        return x_upscale


    def forward(self, lr):
        # Bicubic
        lr_upscale = self._interpolate(lr, self.angRes, scale_factor=self.factor)
        # [B(atch), 1, A(ngRes)*h(eight)*S(cale), A(ngRes)*w(idth)*S(cale)]

        # reshape for LFT
        lr = rearrange(lr, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=self.angRes, a2=self.angRes)
        # [B, C(hannels), A^2, h, w]
        for m in self.modules():
            m.h = lr.size(-2)
            m.w = lr.size(-1)

      
        buffer = self.conv_init0(lr)
        buffer = self.conv_init(buffer) + buffer  # [B, C, A^2, h, w]
     
        spa_position = self.pos_encoding(buffer, dim=[3, 4], token_dim=self.channels)
        ang_position = self.pos_encoding(buffer, dim=[2], token_dim=self.channels)
        for m in self.modules():
            m.spa_position = spa_position
            m.ang_position = ang_position
        
        buffer = self.altblock(buffer) + buffer
        
        buffer = rearrange(buffer, 'b c (a1 a2) h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        buffer = self.upsampling(buffer)
        out = buffer + lr_upscale

        return out


class PositionEncoding(nn.Module):
    def __init__(self, temperature):
        super(PositionEncoding, self).__init__()
        self.temperature = temperature

    def forward(self, x, dim: list, token_dim):
        self.token_dim = token_dim
        assert len(x.size()) == 5, 'the object of position encoding requires 5-dim tensor! '
        grid_dim = torch.linspace(0, self.token_dim - 1, self.token_dim, dtype=torch.float32)
        grid_dim = 2 * (grid_dim // 2) / self.token_dim
        grid_dim = self.temperature ** grid_dim
        position = None
        for index in range(len(dim)):
            pos_size = [1, 1, 1, 1, 1, self.token_dim]
            length = x.size(dim[index])
            pos_size[dim[index]] = length

            pos_dim = (torch.linspace(0, length - 1, length, dtype=torch.float32).view(-1, 1) / grid_dim).to(x.device)
            pos_dim = torch.cat([pos_dim[:, 0::2].sin(), pos_dim[:, 1::2].cos()], dim=1)
            pos_dim = pos_dim.view(pos_size)

            if position is None:
                position = pos_dim
            else:
                position = position + pos_dim
            pass

        position = rearrange(position, 'b 1 a h w dim -> b dim a h w')

        return position / len(dim)


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
        return F.interpolate(inputs, size=target_size, mode=self.base_resize_method)

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

class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attnscale=True):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)
        
        self.proj_drop = nn.Dropout(proj_drop)

        self.attnscale = attnscale
        if attnscale:
            self.lamb = nn.Parameter(torch.zeros(num_heads), requires_grad=True)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale , qkv[1], qkv[2] 
    
        attn = (q @ k.transpose(-2, -1))

        # pre-softmax communication
        attn = self.proj_l(attn.permute(0,2,3,1)).permute(0,3,1,2)
                
        attn = attn.softmax(dim=-1)
  
        # post-softmax communication
        attn = self.proj_w(attn.permute(0,2,3,1)).permute(0,3,1,2)

        ### AttnScale
        if self.attnscale:
            attn_d = torch.ones(attn.shape[-2:], device=attn.device) / N    # [l, l]
            attn_d = attn_d[None, None, ...]                                # [B, N, l, l]
            attn_h = attn - attn_d                                          # [B, N, l, l]
            attn_h = attn_h * (1.+self.lamb[None, :, None, None])           # [B, N, l, l]
            attn = attn_d + attn_h                                          # [B, N, l, l]

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpaTrans(nn.Module):
    def __init__(self, channels, angRes, MHSA_params):
        super(SpaTrans, self).__init__()
        self.angRes = angRes
        self.kernel_field = 3
        self.kernel_search = 5
        self.spa_dim = channels * 2
        self.MLP = nn.Linear(channels * self.kernel_field ** 2, self.spa_dim, bias=False)

        self.norm = nn.LayerNorm(self.spa_dim)

        # self.attention = nn.MultiheadAttention(self.spa_dim,
        #                                        MHSA_params['num_heads'],
        #                                        MHSA_params['dropout'],
        #                                        bias=False)
        self.attention = Attention_talking_head(self.spa_dim, num_heads=MHSA_params['num_heads'], attn_drop=MHSA_params['dropout'], proj_drop=MHSA_params['dropout'], attnscale=True)

        # nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        # self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.spa_dim),
            nn.Linear(self.spa_dim, self.spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.spa_dim*2, self.spa_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )
        self.linear = nn.Sequential(
            nn.Conv3d(self.spa_dim, channels, kernel_size=(1, 1, 1), padding=(0, 0, 0), dilation=1, bias=False),
        )

    @staticmethod
    def gen_mask(h:int, w:int, k:int):
        atten_mask = torch.zeros([h, w, h, w])
        k_left = k//2
        k_right = k - k_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[max(0, i-k_left):min(h,i+k_right), max(0, j-k_left):min(h,j+k_right)] = 1
                atten_mask[i, j, :, :] = temp

        atten_mask = rearrange(atten_mask, 'a b c d -> (a b) (c d)')
        atten_mask = atten_mask.float().masked_fill(atten_mask == 0, float('-inf')).\
            masked_fill(atten_mask == 1, float(0.0))

        return atten_mask

    def SAI2Token(self, buffer):
        buffer = rearrange(buffer, 'b c a h w -> (b a) c h w')
        # local feature embedding
        spa_token = F.unfold(buffer, kernel_size=self.kernel_field, padding=self.kernel_field//2).permute(2, 0, 1)
        spa_token = self.MLP(spa_token)
        return spa_token

    def Token2SAI(self, buffer_token_spa):
        buffer = rearrange(buffer_token_spa, '(h w) (b a) c -> b c a h w', h=self.h, w=self.w, a=self.angRes**2)
        buffer = self.linear(buffer)
        return buffer

    def forward(self, buffer):
        atten_mask = self.gen_mask(self.h, self.w, self.kernel_search).to(buffer.device)

        spa_token = self.SAI2Token(buffer)
        spa_PE = self.SAI2Token(self.spa_position)
        spa_token_norm = self.norm(spa_token + spa_PE)

        # spa_token = self.attention(query=spa_token_norm,
        #                            key=spa_token_norm,
        #                            value=spa_token,
        #                            need_weights=False,
        #                            attn_mask=atten_mask)[0] + spa_token

        spa_token = self.attention(spa_token_norm) + spa_token

        spa_token = self.feed_forward(spa_token) + spa_token
        buffer = self.Token2SAI(spa_token)

        return buffer


class AngTrans(nn.Module):
    def __init__(self, channels, angRes, MHSA_params):
        super(AngTrans, self).__init__()
        self.angRes = angRes
        self.ang_dim = channels
        self.norm = nn.LayerNorm(self.ang_dim)

        # self.attention = nn.MultiheadAttention(self.ang_dim,
        #                                        MHSA_params['num_heads'],
        #                                        MHSA_params['dropout'],
        #                                        bias=False)
        self.attention = Attention_talking_head(self.ang_dim, num_heads=MHSA_params['num_heads'], attn_drop=MHSA_params['dropout'], proj_drop=MHSA_params['dropout'], attnscale=True)

        # nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        # self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.ang_dim),
            nn.Linear(self.ang_dim, self.ang_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.ang_dim * 2, self.ang_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )

    @staticmethod
    def SAI2Token(buffer):
        buffer_token = rearrange(buffer, 'b c a h w -> a (b h w) c')
        return buffer_token

    def Token2SAI(self, buffer_token):
        buffer = rearrange(buffer_token, '(a) (b h w) (c) -> b c a h w', a=self.angRes ** 2, h=self.h, w=self.w)
        return buffer

    def forward(self, buffer):
        ang_token = self.SAI2Token(buffer)
        ang_PE = self.SAI2Token(self.ang_position)
        ang_token_norm = self.norm(ang_token + ang_PE)

        # ang_token = self.attention(query=ang_token_norm,
        #                            key=ang_token_norm,
        #                            value=ang_token,
        #                            need_weights=False)[0] + ang_token

        ang_token = self.attention(ang_token_norm) + ang_token

        ang_token = self.feed_forward(ang_token) + ang_token
        buffer = self.Token2SAI(ang_token)

        return buffer


class AltFilter(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.spa_trans = SpaTrans(channels, angRes, MHSA_params)
        self.ang_trans = AngTrans(channels, angRes, MHSA_params)

    def forward(self, buffer):
        buffer = self.ang_trans(buffer)
        buffer = self.spa_trans(buffer)

        return buffer