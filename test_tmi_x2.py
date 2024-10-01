# type: ignore
import csv
import datetime
import json
import logging
import os.path
import random
from typing import List, Dict

import click
import numpy as np
import torch
import torch.nn.functional as F
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from torch.utils.data import DataLoader

from data.select_dataset import define_Dataset
from dnnlib import EasyDict
from utils_n import utils_image as util
from utils_n import utils_logger
from utils_n import utils_option as option
from utils_n.calc_metrics import calc_metrics
from utils_n.gen_latex_table import gen_latex_table
from utils_n.utils_dist import get_dist_info, init_dist
from utils_n.vis import (
    visualize_data,
    visualize_sharpening_results,
    visualize_with_error_map,
    visualize_with_segmentation,
)
from models.model_factory import load_model, ModelConfig


# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class BFloat16SAM2(torch.nn.Module):
    def __init__(self, sam2_model):
        super().__init__()
        self.sam2_model = sam2_model

    def forward(self, *args, **kwargs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.sam2_model(*args, **kwargs)


class BFloat16SAM2AutomaticMaskGenerator:
    def __init__(self, mask_generator):
        self.mask_generator = mask_generator

    def __call__(self, *args, **kwargs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.mask_generator(*args, **kwargs)


def default_resizer(inputs, target_size):
    return F.interpolate(
        inputs, size=target_size, mode="bilinear", align_corners=False, antialias=True
    )


@click.command()
@click.option(
    "--opt",
    type=str,
    default="options/testing/test_tmi_local.json",
    help="Path to option JSON file.",
)
@click.option("--launcher", default="pytorch", help="job launcher")
@click.option("--local_rank", type=int, default=0)
@click.option("--dist", is_flag=True, default=False)
@click.option("--visualize", is_flag=True, default=True)
@click.option("--backend", default="TkAgg")
def main(**kwargs):

    args = EasyDict(kwargs)
    opt = option.parse(args.opt, is_train=True)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = opt.get("task", "sr_x2")

    if task == "sr_x2":
        from models.network_dpsr import MSRResNet_prior as dpsr
        from models.network_rrdb import RRDB as rrdb
        from models.network_unetmoex1 import Autoencoder as ae1
        from models.network_unetmoex1 import AutoencoderConfig as ae1_cfg
        from models.network_unetmoex1 import EncoderConfig as enc1_cfg
        from models.network_unetmoex1 import MoEConfig as moe1_cfg
        from models.network_unetmoex3 import Autoencoder as ae2
        from models.network_unetmoex3 import AutoencoderConfig as ae2_cfg
        from models.network_unetmoex3 import EncoderConfig as enc2_cfg
        from models.network_unetmoex3 import MoEConfig as moe2_cfg

        from models.moex.network_unetmoex3 import Autoencoder as ae23
        from models.moex.network_unetmoex3 import AutoencoderConfig as ae3_cfg
        from models.moex.network_unetmoex3 import EncoderConfig as enc3_cfg
        from models.network_swinir import SwinIR as swinir

        json_moex3_32_rev = """
        {       
            "netG": {
                "net_type": "unet_moex3_rev",
                "kernel": 32,
                "sharpening_factor": 1.0,
                "model_channels": 64,
                "num_res_blocks": 12,
                "attention_resolutions": [64,32,16],
                "dropout": 0.0,
                "num_groups": 16,
                "num_heads": 16,
                "use_new_attention_order": true,
                "use_checkpoint": true,
                "use_fp16": false,
                "resblock_updown": true,
                "channel_mult": [1,2,4,8,16],
                "conv_resample": true,
                "resample_2d": false,
                "attention_type": "cross_attention",
                "activation": "GELU",
                "rope_theta": 960000.0,
                "resizer_num_layers": 3,
                "resizer_avg_pool": true,
                "init_type": "orthogonal",
                "init_bn_type": "constant",
                "init_gain": 1.0,
                "scale": 2,
                "n_channels": 1,
                "ang_res": 5,
                "phw": 16,
                "overlap": 14
            }
            }
        """

        json_moex3 = """
        {
        "netG": {
            "net_type": "unet_moex3",
            "kernel": 16,
            "sharpening_factor": 1,
            "model_channels": 72,
            "num_res_blocks": 8,
            "attention_resolutions": [16,8,4],
            "dropout": 0.1,
            "num_groups": 36,
            "num_heads": 36,
            "use_new_attention_order": true,
            "use_checkpoint": true,
            "use_fp16": false,
            "resblock_updown": true,
            "channel_mult": [2,4,8],
            "conv_resample": true,
            "resample_2d": false,
            "attention_type": "cross_attention",
            "activation": "GELU",
            "rope_theta": 960000.0,
            "resizer_num_layers": 8,
            "resizer_avg_pool": false,
            "init_type": "default",
            "scale": 2,
            "n_channels": 1
            }
        }
        """

        json_moex3_32 = """
        {
        "netG": {
                "net_type": "unet_moex3",
                "kernel": 32,
                "sharpening_factor": 1,
                "model_channels": 72,
                "num_res_blocks": 8,
                "attention_resolutions": [16,8,4],
                "dropout": 0.1,
                "num_groups": 36,
                "num_heads": 36,
                "use_new_attention_order": true,
                "use_checkpoint": true,
                "use_fp16": false,
                "resblock_updown": true,
                "channel_mult": [2,4,8],
                "conv_resample": true,
                "resample_2d": false,
                "attention_type": "cross_attention",
                "activation": "GELU",
                "rope_theta": 960000.0,
                "resizer_num_layers": 8,
                "resizer_avg_pool": false,
                "init_type": "default",
                "scale": 2,
                "n_channels": 1
            }
        }
        """

        json_moex1 = """
        {
            "netG": {
                "net_type": "unet_moex1",
                "kernel": 16,
                "sharpening_factor": 1.1,
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

        encoder_cfg1 = enc1_cfg(
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
        moex1_conf = ModelConfig(
            encoder_config=encoder_cfg1,
            moe_cfg_class=moe1_cfg,
            ae_cfg_class=ae1_cfg,
            ae_class=ae1,
            model_params={
                "kernel": netG_moex1["kernel"],
                "sharpening_factor": netG_moex1["sharpening_factor"],
                "n_channels": netG_moex1["n_channels"],
                "z": int(
                    2 * netG_moex1["kernel"]
                    + 4 * netG_moex1["kernel"]
                    + netG_moex1["kernel"]
                ),
            },
            opt=opt,
        )

        model_moex1 = load_model(
            moex1_conf,
            sharpening_factor=netG_moex1["sharpening_factor"],
            weights_path=opt["pretrained_models"]["moex1_x2"],
            device=device,
        )

        netG_moex3 = json.loads(json_moex3)["netG"]

        encoder_cfg3 = enc2_cfg(
            model_channels=netG_moex3["model_channels"],  # 32,
            num_res_blocks=netG_moex3["num_res_blocks"],  # 4,
            attention_resolutions=netG_moex3["attention_resolutions"],  # [16, 8],
            dropout=netG_moex3["dropout"],  # 0.2,
            channel_mult=netG_moex3["channel_mult"],  # (2, 4, 8),
            conv_resample=netG_moex3["conv_resample"],  # False,
            dims=2,
            use_checkpoint=netG_moex3["use_checkpoint"],  # True,
            use_fp16=netG_moex3["use_fp16"],  # False,
            num_heads=netG_moex3["num_heads"],  # 4,
            # num_head_channels=netG_moex3["num_head_channels"],  # 8,
            resblock_updown=netG_moex3["resblock_updown"],  # False,
            num_groups=netG_moex3["num_groups"],  # 32,
            resample_2d=netG_moex3["resample_2d"],  # True,
            scale_factor=netG_moex3["scale"],
            resizer_num_layers=netG_moex3["resizer_num_layers"],  # 4,
            resizer_avg_pool=netG_moex3["resizer_avg_pool"],  # False,
            activation=netG_moex3["activation"],
            rope_theta=netG_moex3["rope_theta"],  # 10000.0,
            attention_type=netG_moex3[
                "attention_type"
            ],  # "cross_attention",  # "attention" or "cross_attention"
        )

        moex3_conf = ModelConfig(
            encoder_config=encoder_cfg3,
            moe_cfg_class=moe2_cfg,
            ae_cfg_class=ae2_cfg,
            ae_class=ae2,
            model_params={
                "kernel": netG_moex3["kernel"],
                "sharpening_factor": netG_moex3["sharpening_factor"],
                "n_channels": netG_moex3["n_channels"],
                "z": int(
                    2 * netG_moex3["kernel"]
                    + 4 * netG_moex3["kernel"]
                    + netG_moex3["kernel"]
                ),
            },
            opt=opt,
        )

        model_moex3 = load_model(
            moex3_conf,
            sharpening_factor=netG_moex3["sharpening_factor"],
            weights_path=opt["pretrained_models"]["moex3_x2"],
            device=device,
        )

        netG_moex3_32 = json.loads(json_moex3_32)["netG"]

        encoder_cfg3_32 = enc2_cfg(
            model_channels=netG_moex3_32["model_channels"],  # 32,
            num_res_blocks=netG_moex3_32["num_res_blocks"],  # 4,
            attention_resolutions=netG_moex3_32["attention_resolutions"],  # [16, 8],
            dropout=netG_moex3_32["dropout"],  # 0.2,
            channel_mult=netG_moex3_32["channel_mult"],  # (2, 4, 8),
            conv_resample=netG_moex3_32["conv_resample"],  # False,
            dims=2,
            use_checkpoint=netG_moex3_32["use_checkpoint"],  # True,
            use_fp16=netG_moex3_32["use_fp16"],  # False,
            num_heads=netG_moex3_32["num_heads"],  # 4,
            # num_head_channels=netG_moex3_32["num_head_channels"],  # 8,
            resblock_updown=netG_moex3_32["resblock_updown"],  # False,
            num_groups=netG_moex3_32["num_groups"],  # 32,
            resample_2d=netG_moex3_32["resample_2d"],  # True,
            scale_factor=netG_moex3_32["scale"],
            resizer_num_layers=netG_moex3_32["resizer_num_layers"],  # 4,
            resizer_avg_pool=netG_moex3_32["resizer_avg_pool"],  # False,
            activation=netG_moex3_32["activation"],
            rope_theta=netG_moex3_32["rope_theta"],  # 10000.0,
            attention_type=netG_moex3_32[
                "attention_type"
            ],  # "cross_attention",  # "attention" or "cross_attention"
        )

        moex3_conf_32 = ModelConfig(
            encoder_config=encoder_cfg3_32,
            moe_cfg_class=moe2_cfg,
            ae_cfg_class=ae2_cfg,
            ae_class=ae2,
            model_params={
                "kernel": netG_moex3_32["kernel"],
                "sharpening_factor": netG_moex3_32["sharpening_factor"],
                "n_channels": netG_moex3_32["n_channels"],
                "z": int(
                    2 * netG_moex3_32["kernel"]
                    + 4 * netG_moex3_32["kernel"]
                    + netG_moex3_32["kernel"]
                ),
            },
            opt=opt,
        )

        model_moex3_32 = load_model(
            moex3_conf_32,
            sharpening_factor=netG_moex3_32["sharpening_factor"],
            weights_path=opt["pretrained_models"]["moex3_x2_32"],
            device=device,
        )

        netG_moex3_32_rev = json.loads(json_moex3_32_rev)["netG"]

        encoder_cfg3_32_rev = enc3_cfg(
            model_channels=netG_moex3_32_rev["model_channels"],  # 32,
            num_res_blocks=netG_moex3_32_rev["num_res_blocks"],  # 4,
            attention_resolutions=netG_moex3_32_rev[
                "attention_resolutions"
            ],  # [16, 8],
            dropout=netG_moex3_32_rev["dropout"],  # 0.2,
            channel_mult=netG_moex3_32_rev["channel_mult"],  # (2, 4, 8),
            conv_resample=netG_moex3_32_rev["conv_resample"],  # False,
            dims=2,
            use_checkpoint=netG_moex3_32_rev["use_checkpoint"],  # True,
            use_fp16=netG_moex3_32_rev["use_fp16"],  # False,
            num_heads=netG_moex3_32_rev["num_heads"],  # 4,
            # num_head_channels=netG_moex3_32_rev["num_head_channels"],  # 8,
            resblock_updown=netG_moex3_32_rev["resblock_updown"],  # False,
            num_groups=netG_moex3_32_rev["num_groups"],  # 32,
            resample_2d=netG_moex3_32_rev["resample_2d"],  # True,
            scale_factor=netG_moex3_32_rev["scale"],
            resizer_num_layers=netG_moex3_32_rev["resizer_num_layers"],  # 4,
            resizer_avg_pool=netG_moex3_32_rev["resizer_avg_pool"],  # False,
            activation=netG_moex3_32_rev["activation"],
            rope_theta=netG_moex3_32_rev["rope_theta"],  # 10000.0,
            attention_type=netG_moex3_32_rev[
                "attention_type"
            ],  # "cross_attention",  # "attention" or "cross_attention"
        )

        moex3_conf_32_rev_conf = ModelConfig(
            encoder_config=encoder_cfg3_32_rev,
            moe_cfg_class=moe2_cfg,
            ae_cfg_class=ae3_cfg,
            ae_class=ae23,
            model_params={
                "kernel": netG_moex3_32_rev["kernel"],
                "sharpening_factor": netG_moex3_32_rev["sharpening_factor"],
                "n_channels": netG_moex3_32_rev["n_channels"],
                "z": int(
                    2 * netG_moex3_32_rev["kernel"]
                    + 4 * netG_moex3_32_rev["kernel"]
                    + netG_moex3_32_rev["kernel"]
                ),
            },
            opt=opt,
        )

        model_moex3_32_rev = load_model(
            moex3_conf_32_rev_conf,
            sharpening_factor=netG_moex3_32_rev["sharpening_factor"],
            weights_path=opt["pretrained_models"]["moex3_x2_32_rev"],
            device=device,
        )

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
            torch.load(opt["pretrained_models"]["dpsr_x2"], weights_only=True),
            strict=True,
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
            torch.load(opt["pretrained_models"]["esrgan_x2"], weights_only=True),
            strict=True,
        )
        model_esrgan.eval()
        for k, v in model_esrgan.named_parameters():
            v.requires_grad = False
        model_esrgan = model_esrgan.to(device)

        json_swinir = """
        {
            "netG": {
                "net_type": "swinir",
                "upscale": 2,
                "in_chans": 1,
                "img_size": 48,
                "window_size": 8,
                "img_range": 1.0,
                "depths": [
                    6,
                    6,
                    6,
                    6,
                    6,
                    6
                ],
                "embed_dim": 180,
                "num_heads": [
                    6,
                    6,
                    6,
                    6,
                    6,
                    6
                ],
                "mlp_ratio": 2,
                "upsampler": "pixelshuffle",
                "resi_connection": "1conv",
                "init_type": "default",
                "scale": 2,
                "n_channels": 1,
                "ang_res": 5
            }
        }
        """

        # netG_swinir = json.loads(json_swinir)["netG"]

        # model_swinir = swinir(
        #     upscale=netG_swinir["upscale"],
        #     in_chans=netG_swinir["in_chans"],
        #     img_size=netG_swinir["img_size"],
        #     window_size=netG_swinir["window_size"],
        #     img_range=netG_swinir["img_range"],
        #     depths=netG_swinir["depths"],
        #     embed_dim=netG_swinir["embed_dim"],
        #     num_heads=netG_swinir["num_heads"],
        #     mlp_ratio=netG_swinir["mlp_ratio"],
        #     upsampler=netG_swinir["upsampler"],
        #     resi_connection=netG_swinir["resi_connection"],
        # )
        # model_swinir.eval()
        # for k, v in model_swinir.named_parameters():
        #     v.requires_grad = False
        # model_swinir = model_swinir.to(device)

        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        #     if torch.cuda.get_device_properties(0).major >= 8:
        #         torch.backends.cuda.matmul.allow_tf32 = True
        #         torch.backends.cudnn.allow_tf32 = True

        model_cfg = "sam2_hiera_l.yaml"
        sam2 = build_sam2(
            model_cfg,
            opt["pretrained_models"]["sam2"],
            device="cuda",
            apply_postprocessing=True,
        )

        sam2 = BFloat16SAM2(sam2)

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
            # points_per_side=128,
            # points_per_batch=128,
            # pred_iou_thresh=0.7,
            # stability_score_thresh=0.92,
            # stability_score_offset=0.7,
            # crop_n_layers=4,
            # crop_overlap_ratio=512 / 1500,
            # box_nms_thresh=0.7,
            # crop_n_points_downscale_factor=2,
            # min_mask_region_area=25.0,
            # use_m2m=True,
        )
        mask_generator = BFloat16SAM2AutomaticMaskGenerator(mask_generator)

        models: Dict[str, Any] = {
            "N-SMoE": model_moex1,  # k = 16 | attn=attn
            "N-SMoE-II": model_moex3,  # k = 16 | attn=RoPE
            "N-SMoE-III": model_moex3_32,  # model_moex3_32,  # k = 32 | attn=RoPE | model_moex3_32_rev
            "DPSR": model_dpsr,
            "ESRGAN": model_esrgan,
            # "SwinIR": model_swinir,
            "Bicubic": default_resizer,
        }

        metrics = ["psnr", "ssim", "lpips", "dists", "brisque"]
        metric_data = {
            metric: {
                method: {dataset: {} for dataset in opt["datasets"].keys()}
                for method in models.keys()
            }
            for metric in metrics
        }
        average_metric_data = {}

        timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_dir = os.path.join(opt["path"]["root"], "metrics")
        latex_dir = os.path.join(opt["path"]["root"], "latex")
        util.mkdir(csv_dir)
        util.mkdir(latex_dir)

        for phase, dataset_opt in opt["datasets"].items():
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=16,
                drop_last=False,
                pin_memory=True,
                collate_fn=util.custom_pad_collate_fn,
            )

            H_img_size = dataset_opt["H_size"]
            degrdation = dataset_opt["degradation_type"]
            scale = f'x{dataset_opt["scale"]}'
            dataset_name = dataset_opt["name"]

            for method in models.keys():
                for metric in metrics:
                    metric_data[metric][method][dataset_name][scale] = []

            fmetric_name = os.path.join(
                csv_dir,
                degrdation
                + "_"
                + timestamp.replace(" ", "_").replace(":", "-")
                + ".csv",
            )
            flatex_table = os.path.join(
                latex_dir,
                timestamp.replace(" ", "_").replace(":", "-") + "_" + "latex_table.txt",
            )
            idx = 0

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
                error_map_figure_path = os.path.join(
                    img_dir,
                    f"error-map-{img_name}_{degrdation}_{dataset_name}_{timestamp.replace(' ', '_').replace(':', '-')}.pdf",
                )

                results = calc_metrics(test_data, models, metrics, device)

                for method in models.keys():
                    for metric in metrics:
                        value = results[method][metric]
                        scalar_value = (
                            value.item() if isinstance(value, torch.Tensor) else value
                        )
                        metric_data[metric][method][dataset_name][scale].append(
                            scalar_value
                        )

                for method in models.keys():
                    print(f"{method}:")
                    for metric in metrics:
                        print(f"  {metric.upper()}: {results[method][metric]}")

                if opt["visualize"] == True:
                    L_crop_img = util.tensor2uint(test_data["L"])
                    H_crop_img = util.tensor2uint(test_data["H"])

                    img_H = util.tensor2uint(test_data["O"])
                    img_H = util.modcrop(img_H, border)

                    images: Dict[str, Dict[str, Any]] = {
                        "H_img": {
                            "image": img_H,
                            "title": "High Resolution",
                        },
                        "H_crop_img": {
                            "image": H_crop_img,
                            "title": f"Ground Truth \nCrop",
                        },
                        "L_crop_img": {
                            "image": L_crop_img,
                            "title": f"Noisy Low \nResolution",
                        },
                        "E_Bicubic_img": {
                            "image": results["Bicubic"]["e_img"],
                            "title": "Bicubic",
                        },
                        "E_SMoE_img": {
                            "image": results["N-SMoE"]["e_img"],
                            "title": "N-SMoE",
                        },
                        "E_SMoE_II_img": {
                            "image": results["N-SMoE-II"]["e_img"],
                            "title": "N-SMoE-II",
                        },
                        "E_SMoE_III_img": {
                            "image": results["N-SMoE-III"]["e_img"],
                            "title": "N-SMoE-III",
                        },
                        "E_DPSR_img": {
                            "image": results["DPSR"]["e_img"],
                            "title": "DPSR",
                        },
                        "E_ESRGAN_img": {
                            "image": results["ESRGAN"]["e_img"],
                            "title": "ESRGAN",
                        },
                        # "E_SwinIR_img": {
                        #     "image": results["SwinIR"]["e_img"],
                        #     "title": "SwinIR",
                        # },
                    }

                    # visualize_with_segmentation(
                    #     images,
                    #     mask_generator,
                    #     cmap="gray",
                    #     save_path=seg_figure_path,
                    #     visualize=opt["visualize"],
                    # )

                    visualize_with_error_map(
                        images,
                        cmap="gray",
                        save_path=error_map_figure_path,
                        visualize=opt["visualize"],
                    )

                    visualize_data(
                        images,
                        cmap="gray",
                        save_path=figure_path,
                        visualize=opt["visualize"],
                    )

        for metric in metrics:
            average_metric_data[metric] = {}
            for method in models.keys():
                average_metric_data[metric][method] = {}
                for dataset in opt["datasets"].keys():
                    average_metric_data[metric][method][dataset] = {}
                    for scale in metric_data[metric][method][dataset].keys():

                        values = metric_data[metric][method][dataset][scale]

                        if values:
                            average = sum(values) / len(values)
                        else:
                            average = float("nan")

                        average_metric_data[metric][method][dataset][scale] = average

        with open(fmetric_name, "w", newline="") as csvfile:

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

            for dataset in sorted(opt["datasets"].keys()):
                for method in models.keys():
                    for scale in sorted(
                        average_metric_data["psnr"][method][dataset].keys()
                    ):
                        avg_psnr = average_metric_data["psnr"][method][dataset][scale]
                        avg_ssim = average_metric_data["ssim"][method][dataset][scale]
                        avg_lpips = average_metric_data["lpips"][method][dataset][scale]
                        avg_dists = average_metric_data["dists"][method][dataset][scale]
                        ref_psnr = average_metric_data["psnr"]["N-SMoE"][dataset][scale]
                        ref_ssim = average_metric_data["ssim"]["N-SMoE"][dataset][scale]
                        diff_psnr = ref_psnr - avg_psnr
                        diff_ssim = ref_ssim - avg_ssim

                        csvwriter.writerow(
                            [
                                dataset,
                                degrdation,
                                scale,
                                H_img_size,
                                method,
                                f"{avg_psnr:.4f}",
                                f"{avg_ssim:.4f}",
                                f"{avg_lpips:.4f}",
                                f"{avg_dists:.4f}",
                                f"{diff_psnr:.4f}",
                                f"{diff_ssim:.4f}",
                            ]
                        )

        print(f"Results for all datasets saved to CSV file: {fmetric_name}")

        latex_table = gen_latex_table(average_metric_data)
        with open(flatex_table, "w") as f:
            f.write(latex_table)
        print(f"Latex table saved to {flatex_table}")

    elif task == "sharpening":
        import matlab.engine
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        from collections import defaultdict
        from models.network_unetmoex1 import Autoencoder as ae1
        from models.network_unetmoex1 import AutoencoderConfig as ae1_cfg
        from models.network_unetmoex1 import EncoderConfig as enc1_cfg
        from models.network_unetmoex1 import MoEConfig as moe1_cfg

        eng = matlab.engine.start_matlab()
        matlab_func_dir = os.path.join(os.path.dirname(__file__), "matlab")
        eng.addpath(matlab_func_dir, nargout=0)

        def calculate_sharpness_index(image):
            image_np = (
                image.squeeze().cpu().numpy()
                if isinstance(image, torch.Tensor)
                else image
            )
            return eng.sharpness_index(matlab.double(image_np.tolist()))

        def matlab_imsharpen(image, radius=1.5, amount=0.8):
            image_np = image.squeeze().cpu().numpy()
            sharpened = eng.imsharpen(
                matlab.double(image_np.tolist()),
                "Radius",
                float(radius),
                "Amount",
                float(amount),
            )
            return torch.tensor(np.array(sharpened, dtype=np.float32)).float()

        json_moex1 = """
        {
            "netG": {
                "net_type": "unet_moex1",
                "kernel": 16,
                "sharpening_factor": 1.3,
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
                "n_channels": 1,
                "z": 0
            }
        }
        """
        netG_moex1 = json.loads(json_moex1)["netG"]
        netG_moex1["z"] = int(
            2 * netG_moex1["kernel"] + 4 * netG_moex1["kernel"] + netG_moex1["kernel"]
        )
        encoder_cfg = enc1_cfg(
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
        moex1_conf = ModelConfig(
            encoder_config=encoder_cfg,
            moe_cfg_class=moe1_cfg,
            ae_cfg_class=ae1_cfg,
            ae_class=ae1,
            model_params=netG_moex1,
            opt=opt,
        )

        sharpening_factors = [1.0, 1.1, 1.2, 1.3, 1.4]
        timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for phase, dataset_opt in opt["datasets"].items():
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=16,
                drop_last=False,
                pin_memory=True,
                collate_fn=util.custom_pad_collate_fn,
            )
            H_img_size = dataset_opt["H_size"]
            degrdation = dataset_opt["degradation_type"]
            scale = f'x{dataset_opt["scale"]}'
            dataset_name = dataset_opt["name"]

            for idx, test_data in enumerate(test_loader, 1):
                if test_data is None:
                    continue

                image_name = os.path.splitext(os.path.basename(test_data["L_path"][0]))[
                    0
                ]
                img_dir = os.path.join(opt["path"]["images"], image_name)
                util.mkdir(img_dir)
                fname = os.path.join(
                    img_dir,
                    f"{image_name}_{degrdation}_{dataset_name}_sharpening_{timestamp}",
                )
                si_figure_path = f"{fname}.pdf"

                img_L, img_H, img_L_p = (
                    test_data["L"].to(device),
                    test_data["H"].clamp(0, 1).to(device),
                    test_data["L_p"].to(device),
                )
                sharpened_images, metrics = defaultdict(dict), defaultdict(dict)

                with torch.no_grad():
                    for factor in sharpening_factors:
                        weights_path = opt["pretrained_models"]["moex1_x2"]
                        model = load_model(
                            config=moex1_conf,
                            sharpening_factor=factor,
                            weights_path=weights_path,
                            device=device,
                        )

                        E_img = model(img_L_p, img_L.size()).clamp(0, 1).to(torch.float)
                        E_bicubic = matlab_imsharpen(
                            default_resizer(img_L, img_H.size()[2:])
                            .clamp(0, 1)
                            .to(torch.float),
                            1,
                            factor,
                        )

                        sharpened_images["N-SMoE"][factor] = (
                            E_img.squeeze().cpu().numpy()
                        )
                        sharpened_images["Bicubic"][factor] = E_bicubic.cpu().numpy()

                        for method in ["N-SMoE", "Bicubic"]:
                            img = sharpened_images[method][factor]
                            si = calculate_sharpness_index(img)
                            data_min, data_max = img.min(), img.max()
                            data_range = data_max - data_min
                            psnr_val = psnr(
                                img_H.cpu().squeeze().numpy(),
                                img,
                                data_range=data_range,
                            )
                            ssim_val = ssim(
                                img_H.cpu().squeeze().numpy(),
                                img,
                                data_range=data_range,
                                channel_axis=-1,
                            )

                            metrics[method][factor] = {
                                "PSNR": psnr_val,
                                "SSIM": ssim_val,
                                "SI": si,
                            }

                            print(
                                f"Image {idx}, {method}, Factor {factor}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, SI={si:.4f}"
                            )

                visualize_sharpening_results(
                    img_L.cpu().numpy().squeeze(),
                    img_H.cpu().numpy().squeeze(),
                    sharpened_images,
                    metrics,
                    save_path=si_figure_path,
                    visualize=opt["visualize"],
                )

        eng.quit()

    elif task == "upsampling":
        print("Upsampling task")


if __name__ == "__main__":
    main()
