import torch
import torch.nn as nn
import torch.nn.functional as F

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x1, x2): # x1 (bs,out_ch,w1,h1) x2 (bs,in_ch,w2,h2)
        x2 = self.up_scale(x2) # (bs,out_ch,2*w2,2*h2)
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # (bs,out_ch,w1,h1)
        x = torch.cat([x2, x1], dim=1) # (bs,2*out_ch,w1,h1)
        return x

class up_layer(nn.Module):
    def __init__(self, in_ch, out_ch): # !! 2*out_ch = in_ch !!
        super(up_layer, self).__init__()
        self.up = up(in_ch, out_ch)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2): # x1 (bs,out_ch,w1,h1) x2 (bs,in_ch,w2,h2)
        a = self.up(x1, x2) # (bs,2*out_ch,w1,h1)
        x = self.conv(a) # (bs,out_ch,w1,h1) because 2*out_ch = in_ch
        return x

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_layer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x

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
            # net += torch.tanh(scaled_residual.clamp(min=-3, max=3))  # Old. Clamping to prevent extreme values
            net += F.relu(scaled_residual.clamp(min=0, max=1))
            inputs = blurred
        return net

class Encoder(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 latent_dim:int=28, 
                 size:int=16,
                 num_layers:int=2,
                 avg_pool:bool=False,
                 scale_factor:int=1
                 ):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.scale_factor = scale_factor
        
        self.conv1 = double_conv(in_channels, 64)
        self.down1 = down_layer(64, 128)
        self.down2 = down_layer(128, 256)
        self.down3 = down_layer(256, 512)
        self.down4 = down_layer(512, 1024)
        
        self.up1 = up_layer(1024, 512)
        self.up2 = up_layer(512, 256)
        self.up3 = up_layer(256, 128)
        self.up4 = up_layer(128, 64)
        self.last_conv = nn.Conv2d(64, latent_dim, 1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
                nn.Linear((latent_dim*(size*scale_factor)**2), in_channels * latent_dim),
                nn.BatchNorm1d(in_channels * latent_dim),
                nn.ReLU())
        
        self.resizer = MullerResizer(
            base_resize_method='bicubic', kernel_size=5, stddev=1.0, num_layers=num_layers,
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
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1_up = self.up1(x4, x5)
        x2_up = self.up2(x3, x1_up)
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up) 
        x_last = self.last_conv(x4_up)

        xf = self.flatten(x_last)
        x = self.fc1(xf)
        # x = x.view(-1, self.depth, self.latent_dim)
        return x

class MoE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_mixtures=4,
        kernel=4,
        sharpening_factor=1,
        clip_value=5
    ):
        super(MoE, self).__init__()

        self.ch = in_channels
        self.kernel = kernel
        self.num_mixtures = num_mixtures
        self.clip_value = clip_value
        self.α = sharpening_factor

    @staticmethod
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
    
    def forward(self, height, width, params):
        
        grid, _ = self.sample_image_grid((height, width), device=params.device)
        
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

class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels:int=1,
        latent_dim:int=28,
        num_mixtures:int=4,
        kernel:int=4,
        sharpening_factor:int=1,
        scale_factor:int=1,
        stride:int=16,
        phw:int=64,
        num_layers:int=2,
        avg_pool:bool=False,
        pre_trained:str=None
    ):
        super().__init__()

        self.phw = phw
        self.stride = stride
        self.latent_dim = latent_dim
        self.scale_factor = scale_factor
        
        self.encoder= Encoder(in_channels=in_channels,latent_dim=latent_dim, size=phw, num_layers=num_layers, avg_pool=avg_pool, scale_factor=scale_factor)

        if pre_trained:
            model = torch.load(pre_trained)
            state_dict = {
                k.replace("encoder.", ""): v for k, v in model["state_dict"].items()
            }
            self.encoder.load_state_dict(state_dict, strict=False)

        self.decoder = MoE(
            in_channels=in_channels,
            num_mixtures=num_mixtures,
            kernel=kernel,
            sharpening_factor=sharpening_factor
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

        count = count.clamp(min=1)
        out /= count
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
