# type: ignore
import csv
import argparse
import datetime
import json
import logging
import math
import os.path
import random
from typing import Any, List

from matplotlib import figure
from sklearn import metrics
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from torch.utils.data import DataLoader

import models.basicblock as B
from data.select_dataset import define_Dataset
from models.network_dpsr import MSRResNet_prior as dpsr
from models.network_rrdb import RRDB as rrdb
from models.network_unetmoex1 import (
    Autoencoder,
    AutoencoderConfig,
    EncoderConfig,
    MoEConfig,
)
from models.select_model import define_Model
from utils_n import utils_image as util
from utils_n import utils_logger
from utils_n import utils_option as option
from utils_n.utils_dist import get_dist_info, init_dist
import piq

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


"""
# --------------------------------------------
# SR network withsr/BasicSR/basicsr Residual in Residual Dense Block (RRDB)
# "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
# --------------------------------------------
"""


class RRDB(nn.Module):
    """
    gc: number of growth channels
    nb: number of RRDB
    """

    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nc=64,
        nb=23,
        gc=32,
        upscale=4,
        act_mode="L",
        upsample_mode="upconv",
    ):
        super(RRDB, self).__init__()
        assert (
            "R" in act_mode or "L" in act_mode
        ), "Examples of activation function: R, L, BR, BL, IR, IL"

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode="C")

        m_body = [B.RRDB(nc, gc=32, mode="C" + act_mode) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode="C"))

        if upsample_mode == "upconv":
            upsample_block = B.upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )

        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode="3" + act_mode)
        else:
            m_uper = [
                upsample_block(nc, nc, mode="2" + act_mode) for _ in range(n_upscale)
            ]

        H_conv0 = B.conv(nc, nc, mode="C" + act_mode)
        H_conv1 = B.conv(nc, out_nc, mode="C")
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(
            m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail
        )

    def forward(self, x):
        x = self.model(x)
        return x


""""
# --------------------------------------------
# modified SRResNet
#   -- MSRResNet_prior (for DPSR)
# --------------------------------------------
References:
@inproceedings{zhang2019deep,
  title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1671--1681},
  year={2019}
}
@inproceedings{wang2018esrgan,
  title={Esrgan: Enhanced super-resolution generative adversarial networks},
  author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Change Loy, Chen},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={0--0},
  year={2018}
}
@inproceedings{ledig2017photo,
  title={Photo-realistic single image super-resolution using a generative adversarial network},
  author={Ledig, Christian and Theis, Lucas and Husz{\'a}r, Ferenc and Caballero, Jose and Cunningham, Andrew and Acosta, Alejandro and Aitken, Andrew and Tejani, Alykhan and Totz, Johannes and Wang, Zehan and others},
  booktitle={IEEE conference on computer vision and pattern recognition},
  pages={4681--4690},
  year={2017}
}
# --------------------------------------------
"""

# --------------------------------------------
# MSRResNet super-resolver prior for DPSR
# https://github.com/cszn/DPSR
# https://github.com/cszn/DPSR/blob/master/models/network_srresnet.py
# --------------------------------------------


class MSRResNet_prior(nn.Module):
    def __init__(
        self,
        in_nc=4,
        out_nc=3,
        nc=96,
        nb=16,
        upscale=4,
        act_mode="R",
        upsample_mode="upconv",
    ):
        super(MSRResNet_prior, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode="C")

        m_body = [B.ResBlock(nc, nc, mode="C" + act_mode + "C") for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode="C"))

        if upsample_mode == "upconv":
            upsample_block = B.upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode="3" + act_mode)
        else:
            m_uper = [
                upsample_block(nc, nc, mode="2" + act_mode) for _ in range(n_upscale)
            ]

        H_conv0 = B.conv(nc, nc, mode="C" + act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode="C")
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(
            m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SRResNet(nn.Module):
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nc=64,
        nb=16,
        upscale=4,
        act_mode="R",
        upsample_mode="upconv",
    ):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode="C")

        m_body = [B.ResBlock(nc, nc, mode="C" + act_mode + "C") for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode="C"))

        if upsample_mode == "upconv":
            upsample_block = B.upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode="3" + act_mode)
        else:
            m_uper = [
                upsample_block(nc, nc, mode="2" + act_mode) for _ in range(n_upscale)
            ]

        H_conv0 = B.conv(nc, nc, mode="C" + act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode="C")
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(
            m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail
        )

    def forward(self, x):
        x = self.model(x)
        return x


def convert_to_3_channel(images):
    return [
        np.repeat(img[:, :, None], 3, axis=-1) if len(img.shape) == 2 else img
        for img in images
    ]


def visualize_with_segmentation(
    images: List[np.ndarray],
    titles: List[str],
    mask_generator: SAM2AutomaticMaskGenerator,
    cmap: str = "gray",
    save_path: str = None,
    visualize: bool = False,
    backend: str = "TkAgg",
):
    import matplotlib

    matplotlib.use(backend)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import cv2

    def show_anns(anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        img = np.ones(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
                4,
            )
        )
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann["segmentation"]
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            if borders:
                contours, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                contours = [
                    cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                    for contour in contours
                ]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
        ax.imshow(img)

    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(
        3,
        len(images) + 1,
        height_ratios=[2, 2, 0.5],  # Adjusting the height ratios for better space usage
        width_ratios=[2, 2, 1, 1, 1, 1, 1],
        hspace=0.01,  # Reducing space between rows
        wspace=0.01,  # Reducing space between columns
    )

    annotated_mask = mask_generator.generate(np.repeat(images[0][:, :, :], 3, axis=-1))
    ax_annotated = fig.add_subplot(gs[0:2, 0])
    ax_annotated.imshow(images[0], cmap=cmap)
    show_anns(annotated_mask)
    ax_annotated.axis("off")
    ax_annotated.set_title("Annotated Segmentation", fontsize=12, weight="bold")

    ax_img_hr = fig.add_subplot(gs[0:2, 1])
    ax_img_hr.imshow(images[0], cmap=cmap)
    ax_img_hr.axis("off")
    ax_img_hr.set_title(titles[0], fontsize=12, weight="bold")

    for i in range(1, len(images)):
        ax_img = fig.add_subplot(gs[0, i + 1])
        ax_img.imshow(images[i], cmap=cmap)
        ax_img.axis("off")

        mask = mask_generator.generate(np.repeat(images[i][:, :, None], 3, axis=-1))
        ax_seg = fig.add_subplot(gs[1, i + 1])
        ax_seg.imshow(images[i], cmap=cmap)
        show_anns(mask)
        ax_seg.axis("off")

        if len(titles[i]) > 20:
            title_lines = titles[i].split(" ", 1)
            display_title = "\n".join(title_lines)

        else:
            display_title = titles[i]

        ax_title = fig.add_subplot(gs[2, i + 1])
        ax_title.text(
            0.5,
            0.5,
            display_title,
            fontsize=10,
            weight="bold",
            va="center",
            ha="center",
        )
        ax_title.axis("off")

    plt.tight_layout(pad=0.05)
    plt.subplots_adjust(
        wspace=0.02, hspace=0.02
    )  # Further reduce space between figures

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0)
    if visualize:
        plt.show()


def visualize_data_pair(images, titles):
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from numpy.fft import fft2, fftshift
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    num_images = len(images)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = GridSpec(5, num_images + 1, figure=fig, height_ratios=[3, 0.5, 0.5, 1, 2])

    axes_colors = ["darkslategray", "olive", "steelblue", "darkred", "slategray"]
    reference_title = "Ground Truth Crop"
    reference_index = titles.index(reference_title) if reference_title in titles else -1
    reference_image = (
        images[reference_index].squeeze() if reference_index != -1 else None
    )

    for i, (img, title) in enumerate(zip(images, titles)):
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(img, cmap="gray")
        ax_img.axis("on")
        for spine in ax_img.spines.values():  # Apply color to each spine
            spine.set_color(axes_colors[i % len(axes_colors)])

        if title in ["N-SMoE", "DPSR"] and reference_image is not None:
            current_psnr = psnr(reference_image, img, data_range=img.max() - img.min())
            current_ssim = ssim(reference_image, img, data_range=img.max() - img.min())
            title += f"\nPSNR: {current_psnr:.2f} dB, SSIM: {current_ssim:.4f}"

        ax_img.set_title(
            title, fontsize=12, family="Times New Roman", fontweight="bold"
        )

        freq = fftshift(fft2(img))
        freq_magnitude = np.log(np.abs(freq) + 1)

        ax_x_spectrum = fig.add_subplot(gs[1, i])
        ax_x_spectrum.plot(np.sum(freq_magnitude, axis=0), color="blue")
        ax_x_spectrum.set_title(
            "X-Spectrum", fontsize=12, family="Times New Roman", fontweight="bold"
        )
        ax_x_spectrum.set_xlabel(
            "Frequency (pixels)", fontsize=11, family="Times New Roman"
        )
        ax_x_spectrum.set_yticklabels([])
        ax_x_spectrum.tick_params(axis="both", which="major", labelsize=10)
        ax_x_spectrum.grid(True)

        ax_y_spectrum = fig.add_subplot(gs[2, i])
        ax_y_spectrum.plot(np.sum(freq_magnitude, axis=1), color="blue")
        ax_y_spectrum.set_title(
            "Y-Spectrum", fontsize=12, family="Times New Roman", fontweight="bold"
        )
        ax_y_spectrum.set_xlabel(
            "Frequency (pixels)", fontsize=11, family="Times New Roman"
        )
        ax_y_spectrum.set_yticklabels([])
        ax_y_spectrum.tick_params(axis="both", which="major", labelsize=10)
        ax_y_spectrum.grid(True)

        ax_2d_spectrum = fig.add_subplot(gs[3, i])
        ax_2d_spectrum.imshow(freq_magnitude, cmap="gray")
        ax_2d_spectrum.set_title(
            "2D Spectrum", fontsize=12, family="Times New Roman", fontweight="bold"
        )
        ax_2d_spectrum.axis("on")

    nsmoe_idx = titles.index("N-SMoE") if "N-SMoE" in titles else -1
    dpsr_idx = titles.index("DPSR") if "DPSR" in titles else -1

    if nsmoe_idx != -1 and dpsr_idx != -1:
        rec_image = images[nsmoe_idx].squeeze()
        dpsr_image = images[dpsr_idx].squeeze()
        error_map = np.abs(rec_image - dpsr_image)

        ax_error_map = fig.add_subplot(gs[0, dpsr_idx + 1])
        ax_error_map.imshow(error_map, cmap="viridis")
        ax_error_map.set_title(
            "Error Map (N-SMoE - DPSR)",
            fontsize=12,
            family="Times New Roman",
            fontweight="bold",
        )
        ax_error_map.axis("off")

    plt.show()


def visualize_data(
    images: List[np.ndarray],
    titles: List[str],
    cmap: str = "gray",
    save_path: str = None,
    visualize: bool = True,
    backend: str = "TkAgg",
) -> None:
    import matplotlib

    matplotlib.use(backend)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from numpy.fft import fft2, fftshift
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    num_images = len(images)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = GridSpec(5, num_images, figure=fig, height_ratios=[3, 0.5, 0.5, 1, 2])

    axes_colors = ["darkslategray", "olive", "steelblue", "darkred", "slategray"]
    reference_title = "Ground Truth Crop"
    low_res_title = "Noisy Low Resolution Crop"
    reference_index = titles.index(reference_title)
    reference_image = images[reference_index].squeeze()

    psnr_values = {}
    ssim_values = {}

    for i, (img, title) in enumerate(zip(images, titles)):
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(img, cmap=cmap)
        ax_img.axis("on")
        for spine in ax_img.spines.values():
            spine.set_color(axes_colors[i % len(axes_colors)])

        if title != reference_title and title != low_res_title:
            current_psnr = psnr(reference_image, img, data_range=img.max() - img.min())
            current_ssim = ssim(
                reference_image, img, channel_axis=-1, data_range=img.max() - img.min()
            )
            psnr_values[title] = current_psnr
            ssim_values[title] = current_ssim
            title += f"\nPSNR: {current_psnr:.2f} dB, SSIM: {current_ssim:.4f}"

        ax_img.set_title(
            title, fontsize=12, family="Times New Roman", fontweight="bold"
        )

        freq = fftshift(fft2(img))
        freq_magnitude = np.log(np.abs(freq) + 1)

        ax_x_spectrum = fig.add_subplot(gs[1, i])
        ax_x_spectrum.plot(np.sum(freq_magnitude, axis=0), color="blue")
        ax_x_spectrum.set_title(
            "X-Spectrum", fontsize=12, family="Times New Roman", fontweight="bold"
        )
        ax_x_spectrum.set_xlabel(
            "Frequency (pixels)", fontsize=11, family="Times New Roman"
        )
        ax_x_spectrum.set_yticklabels([])
        ax_x_spectrum.tick_params(axis="both", which="major", labelsize=10)
        ax_x_spectrum.grid(True)

        ax_y_spectrum = fig.add_subplot(gs[2, i])
        ax_y_spectrum.plot(np.sum(freq_magnitude, axis=1), color="blue")
        ax_y_spectrum.set_title(
            "Y-Spectrum", fontsize=12, family="Times New Roman", fontweight="bold"
        )
        ax_y_spectrum.set_xlabel(
            "Frequency (pixels)", fontsize=11, family="Times New Roman"
        )
        ax_y_spectrum.set_yticklabels([])
        ax_y_spectrum.tick_params(axis="both", which="major", labelsize=10)
        ax_y_spectrum.grid(True)

        ax_2d_spectrum = fig.add_subplot(gs[3, i])
        ax_2d_spectrum.imshow(freq_magnitude, cmap=cmap)
        ax_2d_spectrum.set_title(
            "2D Spectrum", fontsize=12, family="Times New Roman", fontweight="bold"
        )
        ax_2d_spectrum.axis("on")

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0)
    if visualize:
        plt.show()


def main(json_path="options/testing/test_tmi_local.json"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt", type=str, default=json_path, help="Path to option JSON file."
    )
    parser.add_argument("--launcher", default="pytorch", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist", default=False)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--backend", default="TkAgg")

    args = parser.parse_args()
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt["dist"] = args.dist
    opt["visualize"] = args.visualize
    opt["backend"] = args.backend

    if opt["dist"]:
        init_dist("pytorch")
    opt["rank"], opt["world_size"] = get_dist_info()

    border = opt["scale"]

    opt = option.dict_to_nonedict(opt)

    if opt["rank"] == 0:
        util.mkdirs(
            (path for key, path in opt["path"].items() if "pretrained" not in key)
        )

    if isinstance(opt, dict) and opt.get("rank") == 0:
        logger_name = "train"
        utils_logger.logger_info(
            logger_name, os.path.join(opt["path"]["log"], logger_name + ".log")
        )
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    seed = random.randint(1, 10000)
    print("Random seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for phase, dataset_opt in opt["datasets"].items():
        if phase == "test":
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=16,
                drop_last=False,
                pin_memory=True,
                collate_fn=util.custom_collate,
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_moex1 = """
    {
        "netG": {
            "net_type": "unet_moex1",
            "kernel": 16,
            "sharpening_factor": 1,
            "model_channels": 64,
            "num_res_blocks": 8,
            "attention_resolutions": [16,8,4],
            "dropout": 0.2,
            "num_groups": 8,
            "num_heads": 32,
            "num_head_channels": 32,
            "use_new_attention_order": true,
            "use_checkpoint": true,
            "resblock_updown": false,
            "channel_mult": [1,2,4,8],
            "resample_2d": false,
            "pool": "attention",
            "activation": "GELU",
            "resizer_num_layers": 2,
            "resizer_avg_pool": false,
            "scale": 2,
            "n_channels": 1
        }
    }
    """

    netG_moex1 = json.loads(json_moex1)["netG"]

    z = 2 * netG_moex1["kernel"] + 4 * netG_moex1["kernel"] + netG_moex1["kernel"]

    encoder_cfg = EncoderConfig(
        model_channels=netG_moex1["model_channels"],
        num_res_blocks=netG_moex1["num_res_blocks"],
        attention_resolutions=netG_moex1["attention_resolutions"],
        dropout=netG_moex1["dropout"],
        num_groups=netG_moex1["num_groups"],
        scale_factor=netG_moex1["scale"],
        num_heads=netG_moex1["num_heads"],
        num_head_channels=netG_moex1["num_head_channels"],
        use_new_attention_order=netG_moex1["use_new_attention_order"],
        use_checkpoint=netG_moex1["use_checkpoint"],
        resblock_updown=netG_moex1["resblock_updown"],
        channel_mult=netG_moex1["channel_mult"],
        resample_2d=netG_moex1["resample_2d"],
        pool=netG_moex1["pool"],
        activation=netG_moex1["activation"],
    )

    decoder_cfg = MoEConfig(
        kernel=netG_moex1["kernel"],
        sharpening_factor=netG_moex1["sharpening_factor"],
    )

    autoenocer_cfg = AutoencoderConfig(
        EncoderConfig=encoder_cfg,
        DecoderConfig=decoder_cfg,
        d_in=netG_moex1["n_channels"],
        d_out=z,
        phw=opt["phw"],
        overlap=opt["overlap"],
    )

    model_moex1 = Autoencoder(cfg=autoenocer_cfg)

    model_moex1.load_state_dict(
        torch.load(opt["pretrained_models"]["moex1"], weights_only=True), strict=True
    )
    model_moex1.eval()
    for k, v in model_moex1.named_parameters():
        v.requires_grad = False
    model_moex1 = model_moex1.to(device)

    json_dpsr = """
        {
        "netG": {
            "net_type": "dpsr",
            "in_nc": 1,
            "out_nc": 1,
            "nc": 96,
            "nb": 16,
            "gc": 32,
            "ng": 2,
            "reduction": 16,
            "act_mode": "R",
            "upsample_mode": "pixelshuffle",
            "downsample_mode": "strideconv",
            "init_type": "orthogonal",
            "init_bn_type": "uniform",
            "init_gain": 0.2,
            "scale": 2,
            "n_channels": 1,
            "ang_res": 5,
            "phw": 16,
            "overlap": 10
            }
         }
        """

    netG_dpsr = json.loads(json_dpsr)["netG"]

    model_dpsr = dpsr(
        in_nc=netG_dpsr["in_nc"],
        out_nc=netG_dpsr["out_nc"],
        nc=netG_dpsr["nc"],
        nb=netG_dpsr["nb"],
        upscale=netG_dpsr["scale"],
        act_mode=netG_dpsr["act_mode"],
        upsample_mode=netG_dpsr["upsample_mode"],
    )

    model_dpsr.load_state_dict(
        torch.load(opt["pretrained_models"]["dpsr"], weights_only=True), strict=True
    )
    model_dpsr.eval()
    for k, v in model_dpsr.named_parameters():
        v.requires_grad = False
    model_dpsr = model_dpsr.to(device)

    json_rrdb = """
    {
        "netG": {
            "net_type": "rrdb",
            "in_nc": 1,
            "out_nc": 1,
            "nc": 64,
            "nb": 23,
            "gc": 32,
            "ng": 2,
            "reduction": 16,
            "act_mode": "R",
            "upsample_mode": "upconv",
            "downsample_mode": "strideconv",
            "init_type": "orthogonal",
            "init_bn_type": "uniform",
            "init_gain": 0.2,
            "scale": 2,
            "n_channels": 1,
            "ang_res": 5
        }
       }
    """
    netG_rrdb = json.loads(json_rrdb)["netG"]

    model_esrgan = rrdb(
        in_nc=netG_rrdb["in_nc"],
        out_nc=netG_rrdb["out_nc"],
        nc=netG_rrdb["nc"],
        nb=netG_rrdb["nb"],
        gc=netG_rrdb["gc"],
        upscale=netG_rrdb["scale"],
        act_mode=netG_rrdb["act_mode"],
        upsample_mode=netG_rrdb["upsample_mode"],
    )

    model_esrgan.load_state_dict(
        torch.load(opt["pretrained_models"]["esrgan"], weights_only=True), strict=True
    )
    model_esrgan.eval()
    for k, v in model_esrgan.named_parameters():
        v.requires_grad = False
    model_esrgan = model_esrgan.to(device)

    # titles = [
    #     "High Resolution",
    #     "Low Resolution Crop",
    #     "High Resolution Crop",
    #     "N-SMoE",
    #     "DPSR",
    # ]

    model_cfg = "sam2_hiera_l.yaml"

    sam2 = build_sam2(
        model_cfg,
        opt["pretrained_models"]["sam2"],
        device="cuda",
        apply_postprocessing=True,
    )

    # mask_generator = SAM2AutomaticMaskGenerator(
    #     model=sam2,
    #     points_per_side=256,  # Very high density for the finest details
    #     points_per_batch=128,  # More points per batch for thorough segmentation
    #     pred_iou_thresh=0.7,  # Balanced IoU threshold for quality masks
    #     stability_score_thresh=0.95,  # High stability score threshold for the most stable masks
    #     stability_score_offset=1.0,
    #     mask_threshold=0.0,
    #     box_nms_thresh=0.7,
    #     crop_n_layers=4,  # More layers for multi-level cropping
    #     crop_nms_thresh=0.7,
    #     crop_overlap_ratio=512 / 1500,
    #     crop_n_points_downscale_factor=2,  # Adjusted for better point distribution
    #     min_mask_region_area=20,  # Small region processing to remove artifacts
    #     output_mode="binary_mask",
    #     use_m2m=True,  # Enable M2M refinement
    #     multimask_output=True,
    # )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        # points_per_side=64,
        # points_per_batch=128,
        # pred_iou_thresh=0.7,
        # stability_score_thresh=0.92,
        # stability_score_offset=0.7,
        # crop_n_layers=1,
        # box_nms_thresh=0.7,
        # crop_n_points_downscale_factor=2,
        # min_mask_region_area=25.0,
        # use_m2m=True,
    )

    avg_psnr = 0.0
    idx = 0

    psnr_moex_list: list[torch.float] = []
    psnr_dpsr_list: list[torch.float] = []
    psnr_esrgan_list = []

    ssim_moex_list: list[torch.float] = []
    ssim_dpsr_list: list[torch.float] = []
    ssim_esrgan_list: list[torch.float] = []

    lpips_moex_list: list[torch.float] = []
    lpips_dpsr_list: list[torch.float] = []
    lpips_esrgan_list: list[torch.float] = []

    dists_moex_list: list[torch.float] = []
    dists_dpsr_list: list[torch.float] = []
    dists_esrgan_list: list[torch.float] = []

    dataset_name = opt["datasets"]["test"]["name"]
    degrdation = opt["datasets"]["test"]["degradation_type"]
    H_img_size = opt["datasets"]["test"]["H_size"]
    scale: str = f'x{opt["scale"]}'

    timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    methods: List[str] = ["DPSR", "ESRGAN", "N-SMoE"]
    csv_dir = os.path.join(opt["path"]["root"], dataset_name)
    util.mkdir(csv_dir)
    fmetric_name = os.path.join(
        csv_dir,
        degrdation
        + "_"
        + dataset_name
        + "_"
        + timestamp.replace(" ", "_").replace(":", "-"),
    )
    for test_data in test_loader:
        if test_data is None:
            continue

        idx += 1
        image_name_ext = os.path.basename(test_data["L_path"][0])
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt["path"]["images"], img_name)
        util.mkdir(img_dir)

        fname = os.path.join(
            img_dir,
            f"{img_name}_{degrdation}_{dataset_name}_{timestamp.replace(' ', '_').replace(':', '-')}",
        )
        figure_path = f"{fname}.pdf"
        seg_figure_path = os.path.join(
            img_dir,
            f"seg-{img_name}_{degrdation}_{dataset_name}_{timestamp.replace(' ', '_').replace(':', '-')}.pdf",
        )
        with torch.no_grad():
            E_img_moex1 = model_moex1(
                test_data["L_p"].to(device), test_data["L"].size()
            )

            E_img_dpsr = model_dpsr(test_data["L"].to(device))
            E_img_esrgan = model_esrgan(test_data["L"].to(device))

        # gt_img = (test_data["H"].mul(255.0).clamp(0, 255).to(torch.uint8)).to(device)
        # E_img_moex_t = E_img_moex1.mul(255.0).clamp(0, 255).to(torch.uint8)
        # E_img_dpsr_t = E_img_dpsr.mul(255.0).clamp(0, 255).to(torch.uint8)
        # E_img_esrgan_t = E_img_esrgan.mul(255.0).clamp(0, 255).to(torch.uint8)

        gt_img = (test_data["H"].clamp(0, 1).to(torch.float)).to(device)
        E_img_moex_t = E_img_moex1.clamp(0, 1).to(torch.float)
        E_img_dpsr_t = E_img_dpsr.clamp(0, 1).to(torch.float)
        E_img_esrgan_t = E_img_esrgan.clamp(0, 1).to(torch.float)

        psnr_moex1 = piq.psnr(E_img_moex_t, gt_img, data_range=1).float()
        psnr_dpsr = piq.psnr(E_img_dpsr_t, gt_img, data_range=1).float()
        psnr_esrgan = piq.psnr(E_img_esrgan_t, gt_img, data_range=1).float()

        ssim_moex1 = piq.ssim(E_img_moex_t, gt_img, data_range=1, reduction="mean")
        ssim_dpsr = piq.ssim(E_img_dpsr_t, gt_img, data_range=1, reduction="mean")
        ssim_esrgan = piq.ssim(E_img_esrgan_t, gt_img, data_range=1, reduction="mean")

        lpips_moex1 = piq.LPIPS()(E_img_moex_t, gt_img).item()
        lpips_dpsr = piq.LPIPS()(E_img_dpsr_t, gt_img).item()
        lpips_esrgan = piq.LPIPS()(E_img_esrgan_t, gt_img).item()

        dists_moex1 = piq.DISTS()(E_img_moex_t, gt_img).item()
        dists_dpsr = piq.DISTS()(E_img_dpsr_t, gt_img).item()
        dists_esrgan = piq.DISTS()(E_img_esrgan_t, gt_img).item()

        brisque_moex1 = piq.brisque(E_img_moex_t, data_range=1.0, reduction="none")
        brisque_dpsr = piq.brisque(E_img_dpsr_t, data_range=1.0, reduction="none")
        brisque_esrgan = piq.brisque(E_img_esrgan_t, data_range=1.0, reduction="none")

        print(
            f"PSNR N-SMoE: {psnr_moex1}, PSNR DPSR: {psnr_dpsr}, PSNR ESRGAN: {psnr_esrgan}"
        )

        print(
            f"SSIM N-SMoE: {ssim_moex1}, SSIM DPSR: {ssim_dpsr}, SSIM ESRGAN: {ssim_esrgan}"
        )

        print(
            f"LPIPS N-SMoE: {lpips_moex1}, LPIPS DPSR: {lpips_dpsr}, LPIPS ESRGAN: {lpips_esrgan}"
        )

        print(
            f"DISTS N-SMoE: {dists_moex1}, DISTS DPSR: {dists_dpsr}, DISTS ESRGAN: {dists_esrgan}"
        )

        print(
            f"Brisque N-SMoE: {brisque_moex1}, Brisque DPSR: {brisque_dpsr}, Brisque ESRGAN: {brisque_esrgan}"
        )

        psnr_moex_list.append(psnr_moex1)
        psnr_dpsr_list.append(psnr_dpsr)
        psnr_esrgan_list.append(psnr_esrgan)

        ssim_moex_list.append(ssim_moex1)
        ssim_dpsr_list.append(ssim_dpsr)
        ssim_esrgan_list.append(ssim_esrgan)

        lpips_moex_list.append(lpips_moex1)
        lpips_dpsr_list.append(lpips_dpsr)
        lpips_esrgan_list.append(lpips_esrgan)

        dists_moex_list.append(dists_moex1)
        dists_dpsr_list.append(dists_dpsr)
        dists_esrgan_list.append(dists_esrgan)

        E_img_moex1 = util.tensor2uint(E_img_moex1)
        E_img_dpsr = util._tensor2uint(E_img_dpsr)
        E_img_esrgan = util._tensor2uint(E_img_esrgan)

        L_crop_img = util.tensor2uint(test_data["L"])
        H_crop_img = util.tensor2uint(test_data["H"])

        img_H = util.imread_uint(test_data["H_path"][0], n_channels=1)
        img_H = util.modcrop(img_H, border)

        images: dict[str, Any] = {
            "H_img": img_H,
            "H_img_size": H_img_size,
            "L_crop_img": L_crop_img,
            "H_crop_img": H_crop_img,
            "E_SMoE_img": E_img_moex1,
            "E_DPSR_img": E_img_dpsr,
            "E_ESRGAN_img": E_img_esrgan,
            "Degradation_Model": degrdation,
            "scale": scale,
        }

        scipy.io.savemat(f"{fname}.mat", images)

        # visualize_data([L_crop_img, H_crop_img, E_img_moex1], titles)

        titles: list[str] = [
            "High Resolution",
            "Noisy Low Resolution Crop",
            "Ground Truth Crop",
            "N-SMoE",
            "DPSR",
            "ESRGAN",
        ]

        visualize_with_segmentation(
            [img_H, L_crop_img, H_crop_img, E_img_moex1, E_img_dpsr, E_img_esrgan],
            titles,
            mask_generator,
            cmap="gray",
            save_path=seg_figure_path,
            visualize=opt["visualize"],
            backend=opt["backend"],
        )

        visualize_data(
            [L_crop_img, H_crop_img, E_img_moex1, E_img_dpsr, E_img_esrgan],
            titles[1:],
            cmap="gray",
            save_path=figure_path,
            visualize=opt["visualize"],
            backend=opt["backend"],
        )

        current_psnr = util.calculate_psnr(E_img_moex1, H_crop_img, border=border)
        logger.info(
            "{:->4d}--> {:>10s} | {:<4.2f}dB".format(idx, image_name_ext, current_psnr)
        )

        avg_psnr += current_psnr

    avg_psnr_moex = torch.tensor(psnr_moex_list).mean().float()
    avg_psnr_dpsr = torch.tensor(psnr_dpsr_list).mean().float()
    avg_psnr_esrgan = torch.tensor(psnr_esrgan_list).mean().float()

    avg_ssim_moex = torch.tensor(ssim_moex_list).mean().float()
    avg_ssim_dpsr = torch.tensor(ssim_dpsr_list).mean().float()
    avg_ssim_esrgan = torch.tensor(ssim_esrgan_list).mean().float()

    avg_lpips_moex = torch.tensor(lpips_moex_list).mean().float()
    avg_lpips_dpsr = torch.tensor(lpips_dpsr_list).mean().float()
    avg_lpips_esrgan = torch.tensor(lpips_esrgan_list).mean().float()

    avg_dists_moex = torch.tensor(dists_moex_list).mean().float()
    avg_dists_dpsr = torch.tensor(dists_dpsr_list).mean().float()
    avg_dists_esrgan = torch.tensor(dists_esrgan_list).mean().float()

    print(f"Average PSNR N-SMoE: {avg_psnr_moex}")
    print(f"Average PSNR DPSR: {avg_psnr_dpsr}")
    print(f"Average PSNR ESRGAN: {avg_psnr_esrgan}")

    print(f"Average SSIM N-SMoE: {avg_ssim_moex}")
    print(f"Average SSIM DPSR: {avg_ssim_dpsr}")
    print(f"Average SSIM ESRGAN: {avg_ssim_esrgan}")

    print(f"Average LPIPS N-SMoE: {avg_lpips_moex}")
    print(f"Average LPIPS DPSR: {avg_lpips_dpsr}")
    print(f"Average LPIPS ESRGAN: {avg_lpips_esrgan}")

    print(f"Average DISTS N-SMoE: {avg_dists_moex}")
    print(f"Average DISTS DPSR: {avg_dists_dpsr}")
    print(f"Average DISTS ESRGAN: {avg_dists_esrgan}")

    psnr_values: List[torch.Tensor] = [
        avg_psnr_dpsr,
        avg_psnr_esrgan,
        avg_psnr_moex,
    ]

    ssim_values: List[torch.Tensor] = [
        avg_ssim_dpsr,
        avg_ssim_esrgan,
        avg_ssim_moex,
    ]

    lpips_values: List[torch.Tensor] = [
        avg_lpips_dpsr,
        avg_lpips_esrgan,
        avg_lpips_moex,
    ]

    dists_values: List[torch.Tensor] = [
        avg_dists_dpsr,
        avg_dists_esrgan,
        avg_dists_moex,
    ]

    diff_psnr_values: List[torch.Tensor] = [
        psnr_values[-1] - psnr for psnr in psnr_values[:-1]
    ]
    diff_ssim_values: List[torch.Tensor] = [
        ssim_values[-1] - ssim for ssim in ssim_values[:-1]
    ]

    with open((fmetric_name + "_metrics.csv"), "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(
            [
                "Dataset",
                "Degradation",
                "Scale",
                "Image_Size",
                "Method",
                "PSNR",
                "SSIM",
                "LPIPS",
                "DISTS",
                "Diff_PSNR",
                "Diff_SSIM",
            ]
        )

        for i, method in enumerate(methods):
            csvwriter.writerow(
                [
                    dataset_name,
                    degrdation,
                    scale,
                    H_img_size,
                    method,
                    psnr_values[i].item(),
                    ssim_values[i].item(),
                    lpips_values[i].item(),
                    dists_values[i].item(),
                    diff_psnr_values[i].item() if i < len(diff_psnr_values) else "N/A",
                    diff_ssim_values[i].item() if i < len(diff_ssim_values) else "N/A",
                ]
            )
        print(f"Results saved to CSV file: {fmetric_name}_metrics.csv")


if __name__ == "__main__":
    main()
