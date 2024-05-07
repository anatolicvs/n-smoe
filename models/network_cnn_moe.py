import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class InceptionConfig:
    out_1x1: int
    red_3x3: int
    out_3x3: int
    red_5x5: int
    out_5x5: int
    out_1x1pool: int

class FeedForward(nn.Module):
    def __init__(self, in_features, out_features, layers=4, dropout_rate=0.0):
        super(FeedForward, self).__init__()

        intermediate_features = [in_features * (2 ** i) for i in range(1, layers - 1)]
        layer_sizes = [in_features] + intermediate_features + [out_features]

        sequence = []
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            sequence.append(layer)
            sequence.append(nn.LayerNorm(layer_sizes[i + 1]))
            sequence.append(nn.ReLU())
            sequence.append(nn.Dropout(dropout_rate))

        self.layers = nn.Sequential(*sequence)

    def forward(self, x):
        return self.layers(x)
    
class DenseResidualBlock(nn.Module):
    def __init__(self, num_channels, growth_rate, num_layers, group_size=32):
        super(DenseResidualBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_channels = num_channels + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.GroupNorm(1, layer_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(layer_channels, growth_rate, kernel_size=3, padding=1, groups=group_size, bias=False)
                )
            )
            for layer in self.layers:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(outputs, 1))
            outputs.append(out)
        return torch.cat(outputs, 1)

class DenseAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(DenseAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )
        for m in self.attention:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        attention = self.attention(x)
        return torch.cat([x, attention], dim=1)

class CBAM(nn.Module):
    def __init__(self, C, r=16, k=7, p1=1, p2=2):
        super(CBAM, self).__init__()
        self.avgP = nn.AdaptiveAvgPool2d(p1)
        self.maxP = nn.AdaptiveMaxPool2d(p1)
        self.fc = nn.Sequential(
            nn.Conv2d(C, C // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // r, C, 1, bias=False)
        )
        self.sig = nn.Sigmoid()
        self.avgP_small = nn.AdaptiveAvgPool2d(p2)
        self.maxP_small = nn.AdaptiveMaxPool2d(p2)
        self.conv = nn.Conv2d(C, C, kernel_size=p2, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avgF = self.fc(self.avgP(x))
        maxF = self.fc(self.maxP(x))
        chnlAtt = avgF + maxF

        avgF_small = self.conv(self.avgP_small(x))
        maxF_small = self.conv(self.maxP_small(x))
        spatlAtt = self.sig(avgF_small + maxF_small)
        spatlAtt = F.interpolate(spatlAtt, x.shape[2:], mode='bilinear', align_corners=False)

        return self.sig(chnlAtt) * spatlAtt * x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, config: InceptionConfig):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, config.out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, config.red_3x3, kernel_size=1),
            nn.GroupNorm(1, config.red_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.red_3x3, config.out_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, config.red_5x5, kernel_size=1),
            nn.GroupNorm(1, config.red_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.red_5x5, config.out_5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, config.out_1x1pool, kernel_size=1)
        )
        for m in [self.branch1, self.branch2, self.branch3, self.branch4]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        outputs = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        return torch.cat(outputs, 1)

class Network(nn.Module):
    def __init__(self, n_chan, chan_embed=64, inception_config=None, growth_rate=32, num_layers=1,num_res_blocks=3, latent_dim=28, dropout=0.05, avg_pool=False, scale_factor=1.0):
        super(Network, self).__init__()
        self.latent_dim = latent_dim
        self.n_chan = n_chan
        self.scale_factor = scale_factor

        if inception_config is None:
            inception_config = InceptionConfig(
                out_1x1=chan_embed // 4,
                red_3x3=chan_embed // 8,
                out_3x3=chan_embed // 4,
                red_5x5=chan_embed // 16,
                out_5x5=chan_embed // 8,
                out_1x1pool=chan_embed // 4
            )

        self.conv1 = nn.Conv2d(n_chan, chan_embed, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.conv1a = nn.Conv2d(chan_embed, chan_embed, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.conv1a.weight, mode='fan_out', nonlinearity='relu')

        self.inception1 = InceptionModule(chan_embed, inception_config)
        inception_output_channels = sum([inception_config.out_1x1, inception_config.out_3x3, inception_config.out_5x5, inception_config.out_1x1pool])

        self.dense_attention1 = DenseAttention(inception_output_channels)
        attention_output_channels = inception_output_channels * 2

        self.conv2 = nn.Conv2d(attention_output_channels, chan_embed * 2, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.cbam = CBAM(chan_embed * 2)

        self.res_blocks = nn.ModuleList([
            DenseResidualBlock(chan_embed * 2 + growth_rate * num_layers * i, growth_rate, num_layers)
            for i in range(num_res_blocks)
        ])

        final_block_output_channels = chan_embed * 2 + growth_rate * num_layers * num_res_blocks
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        flattened_size = final_block_output_channels * 6 * 6

        self.ff = FeedForward(
            in_features=flattened_size,
            out_features=n_chan * latent_dim,
            layers=num_layers,
            dropout_rate=dropout
        )

        self.resizer = MullerResizer(
            base_resize_method='bilinear', kernel_size=5, stddev=1.0, num_layers=num_layers,
            dtype=torch.float32, avg_pool=avg_pool
        )
    
    def _interpolate(self, x, scale_factor):
        B, C, H, W = x.size() 

        target_h = int(H * scale_factor)  
        target_w = int(W * scale_factor)  
        target_size = (target_h, target_w)  

        x_resized = self.resizer(x, target_size)  

        return x_resized
    
    def forward(self, x):

        x = self._interpolate(x, self.scale_factor)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1a(x))
        x = self.inception1(x)
        x = self.dense_attention1(x)
        x = F.relu(self.conv2(x))
        x = self.cbam(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.ff(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_chan, self.latent_dim)
        return x

def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)
    
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)
    
    return coordinates.reshape(-1, 2), stacked_indices

class MoE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_mixtures=4,
        kernel=4,
        sharpening_factor=1,
        clip_value=5,
    ):
        super(MoE, self).__init__()

        self.ch = in_channels
        self.kernel = kernel
        self.num_mixtures = num_mixtures
        self.clip_value = clip_value
        self.α = sharpening_factor

    def forward(self, height, width, params):
        
        grid, _ = sample_image_grid((height, width), device=params.device)
        
        μ_x = params[:, :, : self.kernel].reshape(-1, self.kernel, 1)
        μ_y = params[:, :, self.kernel : 2 * self.kernel].reshape(-1, self.kernel, 1)
        μ = torch.cat((μ_x, μ_y), 2).view(-1, self.kernel, 2)
        Σ = params[:, :, 3 * self.kernel : 3 * self.kernel + self.kernel * 2 * 2].reshape(-1, self.kernel, 2, 2)
        w = params[:, :, 2 * self.kernel : 3 * self.kernel].reshape(-1, self.kernel)

        Σ = torch.tril(Σ)
        Σ = torch.mul(Σ, self.α)

        μ = μ.unsqueeze(dim=2)
        grid_expand_dim = torch.unsqueeze(torch.unsqueeze(grid, dim=0), dim=0)
        x = torch.tile(grid_expand_dim, (μ.shape[0], μ.shape[1], 1, 1))
        x_sub_μ = torch.unsqueeze(x.float() - μ.float(), dim=-1)

        e = torch.exp(
            -0.5 * torch.einsum("abcli,ablm,abnm,abcnj->abc", x_sub_μ, Σ, Σ, x_sub_μ)
        )

        g = torch.sum(e, dim=1, keepdim=True)
        g_max = torch.max(torch.tensor(10e-8), g)
        e_norm = torch.divide(e, g_max)

        y_hat = torch.sum(e_norm * torch.unsqueeze(w, dim=-1), dim=1)
        y_hat = torch.clamp(y_hat, min=0, max=1)
        
        y_hat = y_hat.view(-1, self.ch, height, width)

        return y_hat

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
        num_mixtures:int=4,  # Tune this parameter based on task complexity
        kernel:int=4,        # Tune this for desired patch size
        sharpening_factor:int=1,
        stride:int=16,       # Set stride such that it covers the image efficiently
        phw:int=64,          # Set patch height and width
        dropout:int=0.0,
        chan_embed:int=64,
        growth_rate:int=32,
        layers:int=4,
        num_res_blocks:int=3
    ):
        super().__init__()

        self.phw = phw
        self.stride = stride
        self.latent_dim = latent_dim

        self.encoder = Network(
            n_chan=in_channels,
            chan_embed=chan_embed,
            growth_rate=growth_rate,
            num_layers=layers,
            num_res_blocks=num_res_blocks,
            latent_dim=latent_dim,
            dropout=dropout
        )

        self.decoder = MoE(
            in_channels=in_channels,
            num_mixtures=num_mixtures,
            kernel=kernel,
            sharpening_factor=sharpening_factor,
            dropout=dropout,
            num_heads=4,  
        )

    @staticmethod
    def reconstruct(patches, shape, khw, stride):
        B, C, H, W = shape

        patches = patches.view(B, -1, C * khw * khw).transpose(1, 2)

        y_hat = F.fold(
            patches, output_size=(H, W), kernel_size=(khw, khw), stride=stride
        )

        one_patch = torch.ones((1, khw * khw, patches.shape[-1]), device=patches.device)
        overlap_count = F.fold(
            one_patch, output_size=(H, W), kernel_size=(khw, khw), stride=stride
        )

        y_hat /= overlap_count.clamp(min=1)

        return y_hat

    def forward(self, x, shape):
        if len(x.shape) == 5:
            x = x.view(-1, *x.size()[2:])

        encoded = self.encoder(x)
        _, C, H, W = shape
        row, col = (W - self.phw) // self.stride + 1, (H - self.phw) // self.stride + 1

        params = encoded.view(-1, row, col, C, self.latent_dim)
        params = params.view(-1, *params.size()[3:])
        decoded = self.decoder(self.phw, self.phw, params)

        y_hat = self.reconstruct(decoded, shape, self.phw, self.stride)

        return y_hat
