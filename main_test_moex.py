import argparse
import json
import logging
import math
import os.path
import random

import datetime
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from torch.utils.data import DataLoader

import models.basicblock as B
from data.select_dataset import define_Dataset
from models.network_dpsr import MSRResNet_prior as dpsr
from models.select_model import define_Model
from utils_n import utils_image as util
from utils_n import utils_logger
from utils_n import utils_option as option
from utils_n.utils_dist import get_dist_info, init_dist
import scipy.io


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


def visualize_with_segmentation(images, titles, mask_generator):
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

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
                import cv2

                contours, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                # Try to smooth contours
                contours = [
                    cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                    for contour in contours
                ]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

        ax.imshow(img)

    # masks = mask_generator.generate(convert_to_3_channel(images[1:]))

    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(
        3, 5, height_ratios=[2, 2, 1], width_ratios=[2, 1, 1, 1, 1], hspace=0, wspace=0
    )

    ax_img = fig.add_subplot(gs[0:2, 0])
    ax_img.imshow(images[0], cmap="gray")
    ax_img.axis("off")
    ax_img.set_title(titles[0], fontsize=15, weight="bold")

    for i in range(1, len(images)):

        if len(images[i].shape) == 2:
            img_3c = np.repeat(images[i][:, :, None], 3, axis=-1)
        else:
            img_3c = images[i]

        mask = mask_generator.generate(img_3c)

        ax_crop = fig.add_subplot(gs[0, i])
        ax_crop.imshow(images[i], cmap="gray")
        ax_crop.axis("off")

        ax_seg = fig.add_subplot(gs[1, i])
        ax_seg.imshow(images[i])
        show_anns(mask)
        ax_seg.axis("off")

        ax_title = fig.add_subplot(gs[2, i])
        ax_title.text(
            0.5, 0.2, titles[i], fontsize=12, weight="bold", va="center", ha="center"
        )
        ax_title.axis("off")

    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def visualize_data(images, titles):
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


def main(json_path="/home/ozkan/works/n-smoe/options/train_unet_moex1_psnr_local.json"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt", type=str, default=json_path, help="Path to option JSON file."
    )
    parser.add_argument("--launcher", default="pytorch", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist", default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt["dist"] = parser.parse_args().dist

    if opt["dist"]:
        init_dist("pytorch")
    opt["rank"], opt["world_size"] = get_dist_info()

    init_path_G = "/mnt/e/Weights/superresolution/unet_unet_moex1_sr_plain_v5_x2_mri_rgb_act_gelu/models/25000_G.pth"
    init_iter_G = 25000

    opt["path"]["pretrained_netG"] = init_path_G

    current_step = init_iter_G

    border = opt["scale"]

    opt = option.dict_to_nonedict(opt)

    if isinstance(opt, dict) and opt.get("rank") == 0:
        logger_name = "train"
        utils_logger.logger_info(
            logger_name, os.path.join(opt["path"]["log"], logger_name + ".log")
        )
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    seed = opt["train"]["manual_seed"]
    if seed is None:
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

    model = define_Model(opt)
    model.load()
    if opt["rank"] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    directory = "/home/ozkan/works/diff-smoe/zoo/"
    esrgan = os.path.join(directory, "ESRGAN.pth")

    dpsr_state_path = "/home/ozkan/works/n-smoe/superresolution/dpsr/models/10000_G.pth"

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
        torch.load(dpsr_state_path, weights_only=True), strict=True
    )
    model_dpsr.eval()
    for k, v in model_dpsr.named_parameters():
        v.requires_grad = False
    model_dpsr = model_dpsr.to(model.device)

    model_esrgan = RRDB(
        in_nc=3,
        out_nc=3,
        nc=64,
        nb=23,
        gc=32,
        upscale=opt["scale"],
        act_mode="L",
        upsample_mode="upconv",
    )
    model_esrgan.load_state_dict(
        torch.load(esrgan, weights_only=True), strict=False
    )  # strict=False
    model_esrgan.eval()
    for k, v in model_esrgan.named_parameters():
        v.requires_grad = False
    model_esrgan = model_esrgan.to(model.device)

    avg_psnr = 0.0
    idx = 0

    # titles = [
    #     "High Resolution",
    #     "Low Resolution Crop",
    #     "High Resolution Crop",
    #     "N-SMoE",
    #     "DPSR",
    # ]

    titles = ["Noisy Low Resolution Crop", "Ground Truth Crop", "N-SMoE", "DPSR"]

    # sam2_checkpoint = "/home/ozkan/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    # model_cfg = "sam2_hiera_l.yaml"

    # sam2 = build_sam2(
    #     model_cfg, sam2_checkpoint, device="cuda", apply_postprocessing=True
    # )

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

    # mask_generator = SAM2AutomaticMaskGenerator(
    #     model=sam2,
    #     points_per_side=64,
    #     points_per_batch=128,
    #     pred_iou_thresh=0.7,
    #     stability_score_thresh=0.92,
    #     stability_score_offset=0.7,
    #     crop_n_layers=1,
    #     box_nms_thresh=0.7,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=25.0,
    #     use_m2m=True,
    # )

    for test_data in test_loader:
        if test_data is None:
            continue

        idx += 1
        image_name_ext = os.path.basename(test_data["L_path"][0])
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt["path"]["images"], img_name)
        util.mkdir(img_dir)

        with torch.no_grad():
            E_img_dpsr = model_dpsr(test_data["L"].to(model.device))
            model.feed_data(test_data)
            model.test()

        E_img_dpsr = util._tensor2uint(E_img_dpsr)
        visuals = model.current_visuals()
        L_crop_img = util.tensor2uint(visuals["L"])
        E_crop_img = util.tensor2uint(visuals["E"])
        H_crop_img = util.tensor2uint(visuals["H"])

        img_H = util.imread_uint(test_data["H_path"][0], n_channels=1)
        img_H = util.modcrop(img_H, border)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        degradation_model = "bicubic downsampling + blur"
        images: dict[str, Any] = {
            "H_img": img_H,
            "L_crop_img": L_crop_img,
            "H_crop_img": H_crop_img,
            "E_SMoE_img": E_crop_img,
            "E_DPSR_img": E_img_dpsr,
            "Degradation_Model": degradation_model,
        }

        filename = f'/mnt/e/Medical/sr_results_for_{"dpsr"}_{timestamp.replace(" ", "_").replace(":", "-")}.mat'
        scipy.io.savemat(filename, images)

        visualize_data([L_crop_img, H_crop_img, E_crop_img, E_img_dpsr], titles)

        # visualize_with_segmentation(
        #     [img_H, L_crop_img, H_crop_img, E_crop_img, E_img_dpsr],
        #     titles,
        #     mask_generator,
        # )

        save_img_path = os.path.join(
            img_dir, "{:s}_{:d}.png".format(img_name, current_step)
        )
        util.imsave(E_crop_img, save_img_path)

        current_psnr = util.calculate_psnr(E_crop_img, H_crop_img, border=border)

        logger.info(
            "{:->4d}--> {:>10s} | {:<4.2f}dB".format(idx, image_name_ext, current_psnr)
        )

        avg_psnr += current_psnr

    avg_psnr = avg_psnr / idx

    logger.info("<Average PSNR : {:<.2f}dB\n".format(avg_psnr))


if __name__ == "__main__":
    main()
