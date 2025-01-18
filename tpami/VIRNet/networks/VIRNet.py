#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:35:06

from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DnCNN import DnCNN
from .AttResUNet import AttResUNet
from .KNet import KernelNet as KNet

log_max = log(1e2)
log_min = log(1e-10)


class VIRAttResUNet(nn.Module):
    """
    For Denoising task with UNet denoiser.
    """

    def __init__(
        self,
        im_chn,
        sigma_chn=3,
        n_feat=[64, 128, 192],
        dep_S=5,
        n_resblocks=2,
        noise_cond=True,
        extra_mode="Input",
        noise_avg=False,
    ):
        super(VIRAttResUNet, self).__init__()
        self.SNet = DnCNN(im_chn, sigma_chn, dep=dep_S, noise_avg=noise_avg)

        self.noise_cond = noise_cond
        extra_chn = sigma_chn if noise_cond else 0
        self.RNet = AttResUNet(
            im_chn,
            extra_chn=extra_chn,
            out_chn=im_chn,
            n_feat=n_feat,
            n_resblocks=n_resblocks,
            extra_mode=extra_mode,
        )

    def forward(self, x):
        sigma = torch.exp(torch.clamp(self.SNet(x), min=log_min, max=log_max))
        extra_maps = sigma.sqrt() if self.noise_cond else None
        mu = self.RNet(x, extra_maps)
        return mu, sigma


class VIRAttResUNetSR(nn.Module):
    """
    For Denoising task with UNet denoiser.
    """

    def __init__(
        self,
        im_chn,
        sigma_chn=1,
        kernel_chn=3,
        n_feat=[64, 128, 192],
        dep_S=5,
        dep_K=8,
        noise_cond=True,
        kernel_cond=True,
        n_resblocks=1,
        extra_mode="Down",
        noise_avg=True,
    ):
        super(VIRAttResUNetSR, self).__init__()
        self.noise_cond = noise_cond
        self.noise_avg = noise_avg
        self.kernel_cond = kernel_cond

        extra_chn = 0
        if self.kernel_cond:
            extra_chn += kernel_chn
        if self.noise_cond:
            extra_chn += sigma_chn
        self.SNet = DnCNN(im_chn, sigma_chn, dep=dep_S, noise_avg=noise_avg)
        self.KNet = KNet(im_chn, kernel_chn, num_blocks=dep_K)
        self.RNet = AttResUNet(
            im_chn,
            extra_chn=extra_chn,
            out_chn=im_chn,
            n_feat=n_feat,
            n_resblocks=n_resblocks,
            extra_mode=extra_mode,
        )

    def forward(self, x, sf):
        sigma = torch.exp(
            torch.clamp(self.SNet(x), min=log_min, max=log_max)
        )  # N x [] x 1 x 1
        kinfo_est = self.KNet(x)  # N x [] x 1 x 1
        x_up = F.interpolate(x, scale_factor=sf, mode="nearest")
        h_up, w_up = x_up.shape[-2:]
        if not self.noise_cond and not self.kernel_cond:
            extra_maps = None
        else:
            extra_temp = []
            if self.kernel_cond:
                extra_temp.append(kinfo_est.repeat(1, 1, h_up, w_up))
            if self.noise_cond:
                if self.noise_avg:
                    extra_temp.append(sigma.sqrt().repeat(1, 1, h_up, w_up))
                else:
                    extra_temp.append(
                        F.interpolate(sigma.sqrt(), scale_factor=sf, mode="nearest")
                    )
            extra_maps = torch.cat(extra_temp, 1)  # n x [] x h x w
        mu = self.RNet(x_up, extra_maps)
        return mu, kinfo_est.squeeze(-1).squeeze(-1), sigma


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VIRAttResUNetSR(
        im_chn=3,  # Number of image channels (3 for RGB)
        sigma_chn=1,  # Number of noise channels
        kernel_chn=3,  # Number of kernel channels
        n_feat=[64, 128, 192],  # Feature sizes for different layers
        dep_S=5,  # Depth for SNet (noise estimator)
        dep_K=8,  # Depth for KNet (kernel estimator)
        noise_cond=True,  # Use noise conditioning
        kernel_cond=True,  # Use kernel conditioning
        n_resblocks=1,  # Number of residual blocks in RNet
        extra_mode="both",  # Mode for handling extra information
        noise_avg=True,  # Use averaging for noise estimates
    ).to(device)

    model.eval()

    batch_size = 1
    channels = 3
    height = 256
    width = 256

    random_image = torch.rand((batch_size, channels, height, width)).to(device)
    scale_factor = 2

    with torch.no_grad():
        mu, kinfo_est, sigma = model(random_image, sf=scale_factor)

    mu_cpu = mu.squeeze(0).cpu()  # Shape: [3, H*sf, W*sf]
    kinfo_est_cpu = kinfo_est.squeeze(0).cpu().numpy()  # Shape: [3]
    sigma_cpu = sigma.squeeze(0).cpu().numpy()  # Shape: [1]

    print("Estimated Kernel Info (kinfo_est):", kinfo_est_cpu)
    print("Estimated Noise Level (sigma):", sigma_cpu)
