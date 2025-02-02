import argparse
import os
import commentjson as json
import sys
import time
import math

from pathlib import Path
from collections import OrderedDict
from loss.ELBO_simple import elbo_sisr

# from networks.VIRNet import VIRAttResUNetSR
from datasets.SISRDatasets import GeneralTrainFloder, GeneralTest

from util import util_net
from util import util_sisr
from util import util_image
from util import util_denoising
from util import util_opts
from util import util_common

import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist
import torch.utils.data as udata
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime, uuid
import torch._dynamo

torch._dynamo.config.cache_size_limit = 0


from networks.network_moex import (
    EncoderConfig,
    MoEConfig,
    AutoencoderConfig,
    Autoencoder,
    KernelType,
)

# from networks.network_transformer_moex import (
#     Autoencoder,
#     EncoderConfig,
#     MoEConfig,
#     BackboneResnetCfg,
#     AutoencoderConfig,
#     BackboneDinoCfg,
#     KernelType,
# )


def init_dist(backend="nccl", **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def close_dist():
    dist.destroy_process_group()


def main():
    # set parameters
    with open(opts_parser.config, "r") as f:
        args = json.load(f)
    util_opts.update_args(args, opts_parser)

    # set the available GPUs
    num_gpus = torch.cuda.device_count()
    args["dist"] = True if num_gpus > 1 else False
    dir_path = Path(args["save_dir"])
    args["save_dir"] = str(dir_path)

    # noise types
    noise_types_list = [
        "Gaussian",
    ]
    if util_opts.str2bool(args["add_jpeg"]):
        noise_types_list.append("JPEG")

    # distributed settings
    if args["dist"]:
        init_dist()
        rank = dist.get_rank()
    else:
        rank = 0
    device = torch.device(f"cuda:{rank % num_gpus}")
    # print the arg pamameters
    if rank == 0:
        for key, value in args.items():
            print("{:<20s}: {:s}".format(key, str(value)))

    # torch.manual_seed(1234)
    # torch.cuda.manual_seed_all(1234)

    # build the model
    # net = VIRAttResUNetSR(
    #     im_chn=args["im_chn"],
    #     sigma_chn=args["sigma_chn"],
    #     dep_K=args["dep_K"],
    #     dep_S=args["dep_S"],
    #     n_feat=args["n_feat"],
    #     n_resblocks=args["n_resblocks"],
    #     noise_cond=util_opts.str2bool(args["noise_cond"]),
    #     kernel_cond=util_opts.str2bool(args["kernel_cond"]),
    #     extra_mode=args["extra_mode"],
    #     noise_avg=(not util_opts.str2bool(args["add_jpeg"])),
    # ).cuda()

    encoder_cfg = EncoderConfig(
        sigma_chn=args["sigma_chn"],
        kernel_chn=args["kernel_chn"],
        noise_cond=util_opts.str2bool(args["noise_cond"]),
        kernel_cond=util_opts.str2bool(args["kernel_cond"]),
        noise_avg=util_opts.str2bool(args["noise_avg"]),
        model_channels=args["model_channels"],
        num_res_blocks=args["num_res_blocks"],
        attention_resolutions=args["attention_resolutions"],
        dropout=args["dropout"],
        channel_mult=tuple(args["channel_mult"]),
        conv_resample=util_opts.str2bool(args["conv_resample"]),
        dims=args["dims"],
        use_checkpoint=util_opts.str2bool(args["use_checkpoint"]),
        use_fp16=util_opts.str2bool(args["use_fp16"]),
        num_heads=args["num_heads"],
        num_head_channels=args["num_head_channels"],
        resblock_updown=util_opts.str2bool(args["resblock_updown"]),
        num_groups=args["num_groups"],
        resample_2d=util_opts.str2bool(args["resample_2d"]),
        scale_factor=args["sf"],
        resizer_num_layers=args["resizer_num_layers"],
        resizer_avg_pool=util_opts.str2bool(args["resizer_avg_pool"]),
        activation=args["activation"],
        rope_theta=args["rope_theta"],
        attention_type=args["attention_type"],
    )

    decoder_cfg = MoEConfig(
        kernel=args["kernel"],
        sharpening_factor=args.get("sharpening_factor", 1),
        kernel_type=KernelType(args["kernel_type"]),
    )

    autoencoder_cfg = AutoencoderConfig(
        EncoderConfig=encoder_cfg,
        DecoderConfig=decoder_cfg,
        d_in=args["im_chn"],
        phw=args["phw"],
        overlap=args["overlap"],
        dep_S=args["dep_S"],
        dep_K=args["dep_K"],
    )

    net = Autoencoder(cfg=autoencoder_cfg)
    net = torch.compile(net)
    net = net.cuda()

    # encoder_cfg = EncoderConfig(
    #     embed_dim=args["embed_dim"],
    #     depth=args["depth"],
    #     heads=args["heads"],
    #     dim_head=args["dim_head"],
    #     mlp_dim=args["mlp_dim"],
    #     dropout=args["dropout"],
    #     patch_size=args["patch_size"],
    #     scale_factor=args["sf"],
    #     resizer_num_layers=args["resizer_num_layers"],
    #     resizer_avg_pool=util_opts.str2bool(args["resizer_avg_pool"]),
    #     activation=args["activation"],
    #     backbone_cfg=BackboneDinoCfg(
    #         name="dino",
    #         model=args[
    #             "dino_model"
    #         ],  # "dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8",
    #         backbone_cfg=BackboneResnetCfg(
    #             name="resnet",
    #             model=args["resnet_model"],  # "resnet18", "resnet50", "resnet101"
    #             num_layers=args["resnet_num_layers"],
    #             use_first_pool=util_opts.str2bool(args["resnet_use_first_pool"]),
    #         ),
    #     ),
    #     kernel_chn=args["kernel_chn"],
    #     sigma_chn=args["sigma_chn"],
    #     noise_cond=util_opts.str2bool(args["noise_cond"]),
    #     kernel_cond=util_opts.str2bool(args["kernel_cond"]),
    #     noise_avg=util_opts.str2bool(args["noise_avg"]),
    # )

    # decoder_cfg = MoEConfig(
    #     kernel=args["kernel"],
    #     sharpening_factor=args["sharpening_factor"],
    #     kernel_type=KernelType(args["kernel_type"]),
    # )

    # autoencoder_cfg = AutoencoderConfig(
    #     EncoderConfig=encoder_cfg,
    #     DecoderConfig=decoder_cfg,
    #     d_in=args["im_chn"],
    #     phw=args["phw"],
    #     overlap=args["overlap"],
    #     dep_S=args["dep_S"],
    #     dep_K=args["dep_K"],
    # )

    # net = Autoencoder(cfg=autoencoder_cfg).to(device)

    if rank == 0:
        print(
            "Number of parameters in SNet: {:.2f}M".format(
                util_net.calculate_parameters(net.snet) / (1000**2)
            ),
            flush=True,
        )

        # print(
        #     "Number of parameters in SNet: {:.2f}M".format(
        #         util_net.calculate_parameters(net.SNet) / (1000**2)
        #     ),
        #     flush=True,
        # )
        print(
            "Number of parameters in KNet: {:.2f}M".format(
                util_net.calculate_parameters(net.knet) / (1000**2)
            ),
            flush=True,
        )
        # print(
        #     "Number of parameters in KNet: {:.2f}M".format(
        #         util_net.calculate_parameters(net.KNet) / (1000**2)
        #     ),
        #     flush=True,
        # )
        print(
            "Number of parameters in MULLER: {:.2f}M".format(
                util_net.calculate_parameters(net.encoder.resizer) / (1000**2)
            ),
            flush=True,
        )

        # print(
        #     "Number of parameters in Encoder: {:.2f}M".format(
        #         util_net.calculate_parameters(net.RNet) / (1000**2)
        #     ),
        #     flush=True,
        # )
        print(
            "Number of parameters in Encoder: {:.2f}M".format(
                util_net.calculate_parameters(net.encoder) / (1000**2)
            ),
            flush=True,
        )
        print(net)
    if args["dist"]:
        net = DDP(
            net, device_ids=[rank], find_unused_parameters=True
        )  # wrap the network

    optimizer = optim.Adam(net.parameters(), lr=args["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args["epochs"], eta_min=args["lr_min"]
    )
    if rank == 0:
        print("T_max = {:d}, eta_min={:.2e}".format(args["epochs"], args["lr_min"]))
        # util_net.test_scheduler(scheduler, optimizer, args['epochs'])

    # resume from specific epoch
    if rank == 0:
        log_dir = Path(args["save_dir"]) / "logs"
        model_dir = Path(args["save_dir"]) / "models"
    if args["resume"]:
        if os.path.isfile(args["resume"]):
            checkpoint = torch.load(args["resume"], map_location="cuda:%d" % rank)
            args["epoch_start"] = checkpoint["epoch"]
            try:
                net.load_state_dict(checkpoint["model_state_dict"])
            except:
                net.load_state_dict(
                    OrderedDict(
                        {
                            "module." + key: value
                            for key, value in checkpoint["model_state_dict"].items()
                        }
                    )
                )
            for _ in range(args["epoch_start"]):
                scheduler.step()
            if rank == 0:
                args["step"] = checkpoint["step"]
                args["step_img"] = checkpoint["step_img"]
                print(
                    "=> Loaded checkpoint {:s} (epoch {:d})".format(
                        args["resume"], checkpoint["epoch"]
                    ),
                    flush=True,
                )
        else:
            sys.exit("Please provide corrected model path!")
    else:
        args["epoch_start"] = 0
        if rank == 0:
            util_common.mkdir(log_dir, delete=True, parents=True)
            util_common.mkdir(model_dir, delete=False, parents=True)

    db_length = args.get("db_length", 100000)
    train_dataset = GeneralTrainFloder(
        hr_dir=args["train_hr_patchs"],
        sf=args["sf"],
        length=db_length * args["batch_size"],
        hr_size=args["hr_size"],
        k_size=args["k_size"],
        kernel_shift=util_opts.str2bool(args["kernel_shift"]),
        downsampler=args["downsampler"],
        add_jpeg=util_opts.str2bool(args["add_jpeg"]),
        noise_jpeg=args["noise_jpeg"],
        noise_level=args["noise_level"],
        chn=args["chn"],
    )
    if rank == 0:
        print(
            "Number of Patches in training data set: {:d}".format(
                train_dataset.num_images
            ),
            flush=True,
        )
    if num_gpus > 1:
        shuffle_flag = False
        train_sampler = udata.distributed.DistributedSampler(
            train_dataset, num_replicas=num_gpus, rank=rank
        )
    else:
        shuffle_flag = True
        train_sampler = None
    train_dataloader = udata.DataLoader(
        train_dataset,
        batch_size=args["batch_size"] // num_gpus,
        shuffle=shuffle_flag,
        drop_last=False,
        num_workers=args["num_workers"] // num_gpus,
        pin_memory=True,
        prefetch_factor=args["prefetch_factor"],
        sampler=train_sampler,
    )

    test_datasets = {
        x: GeneralTest(
            args["val_hr_path"],
            sf=args["sf"],
            k_size=args["k_size"],
            kernel_shift=util_opts.str2bool(args["kernel_shift"]),
            downsampler=args["downsampler"],
            noise_type=x,
            chn=args["chn"],
        )
        for x in noise_types_list
    }
    test_dataloaders = {
        x: udata.DataLoader(
            test_datasets[x],
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            drop_last=False,
        )
        for x in noise_types_list
    }

    if rank == 0:
        num_iter_epoch = {
            "train": math.ceil(len(train_dataset) / args["batch_size"]),
            "test": len(test_datasets["Gaussian"]),
        }
        writer = SummaryWriter(str(log_dir))
        step = args["step"] if args["resume"] else 0
        step_img = (
            args["step_img"]
            if args["resume"]
            else {
                phase: 0
                for phase in [
                    "train",
                ]
                + noise_types_list
            }
        )
    chn = args["im_chn"]
    alpha0 = 0.5 * torch.tensor([args["var_window"] ** 2], dtype=torch.float32).cuda()
    kappa0 = torch.tensor([args["kappa0"]], dtype=torch.float32).cuda()
    param_rnet = [x for name, x in net.named_parameters() if "encoder" in name.lower()]
    # param_rnet = [x for name, x in net.named_parameters() if "rnet " in name.lower()]
    param_snet = [x for name, x in net.named_parameters() if "snet" in name.lower()]
    param_knet = [x for name, x in net.named_parameters() if "knet" in name.lower()]
    for epoch in range(args["epoch_start"], args["epochs"]):
        train_dataset.reset_seed(epoch)
        if num_gpus > 1:
            train_sampler.set_epoch(epoch)
        if rank == 0:
            loss_per_epoch = {x: 0 for x in ["Loss", "lh", "KLR", "KLS", "KLK"]}
            tic = time.time()

        phase = "train"
        net.train()
        lr = optimizer.param_groups[0]["lr"]
        phase = "train"
        for ii, data in enumerate(train_dataloader):

            im_hr, im_lr, im_blur, kinfo_gt, nlevel = [x.cuda() for x in data]
            if util_opts.str2bool(args["add_jpeg"]):
                sigma_prior = util_denoising.noise_estimate_fun(
                    im_lr, im_blur, args["var_window"]
                )
            else:
                sigma_prior = nlevel  # N x 1 x 1 x1 for Gaussian noise

            optimizer.zero_grad()
            # mu, kinfo_est, sigma_est = net(im_lr, args["sf"])
            mu, kinfo_est, sigma_est = net(im_lr)
            loss, loss_detail = elbo_sisr(
                mu=mu,
                sigma_est=sigma_est,
                kinfo_est=kinfo_est,
                im_hr=im_hr,
                im_lr=im_lr,
                sigma_prior=sigma_prior,
                alpha0=alpha0,
                kinfo_gt=kinfo_gt,
                kappa0=kappa0,
                r2=args["r2"],
                eps2=args["eps2"],
                sf=args["sf"],
                k_size=args["k_size"],
                penalty_K=args["penalty_K"],
                downsampler=args["downsampler"],
                shift=util_opts.str2bool(args["kernel_shift"]),
            )
            loss.backward()
            # clip the gradnorm
            total_norm_R = nn.utils.clip_grad_norm_(param_rnet, args["clip_grad_R"])
            total_norm_S = nn.utils.clip_grad_norm_(param_snet, args["clip_grad_S"])
            total_norm_K = nn.utils.clip_grad_norm_(param_knet, args["clip_grad_K"])
            # total_norm_resizer = nn.utils.clip_grad_norm_(
            #     param_resizer, args["clip_grad_Resizer"]
            # )
            optimizer.step()

            if rank == 0:
                loss_per_epoch["Loss"] += loss.item() / num_iter_epoch[phase]
                loss_per_epoch["lh"] += loss_detail[0].item() / num_iter_epoch[phase]
                loss_per_epoch["KLR"] += loss_detail[1].item() / num_iter_epoch[phase]
                loss_per_epoch["KLS"] += loss_detail[2].item() / num_iter_epoch[phase]
                loss_per_epoch["KLK"] += loss_detail[3].item() / num_iter_epoch[phase]
                im_hr_est = mu[0].detach() if isinstance(mu, list) else mu.detach()
                if ((ii + 1) % args["print_freq"] == 0 or ii == 0) and rank == 0:
                    log_str = (
                        "[Epoch:{:>3d}/{:<3d}] {:s}:{:0>5d}/{:0>5d}, lh:{:+>5.2f}, KL:{:+>7.2f}/{:+>6.2f}/{:+>6.2f}, "
                        + "Grad:{:.1e}/{:.1e}/{:.1e}, lr={:.1e}"
                    )
                    print(
                        log_str.format(
                            epoch + 1,
                            args["epochs"],
                            phase,
                            ii + 1,
                            num_iter_epoch[phase],
                            loss_detail[0].item(),
                            loss_detail[1].item(),
                            loss_detail[2].item(),
                            loss_detail[3].item(),
                            total_norm_R,
                            total_norm_S,
                            total_norm_K,
                            # total_norm_resizer,
                            lr,
                        )
                    )
                    writer.add_scalar("Train Loss Iter", loss.item(), step)
                    step += 1
                if (ii + 1) % (20 * args["print_freq"]) == 0 and rank == 0:
                    x1 = vutils.make_grid(im_hr_est, normalize=True, scale_each=True)
                    writer.add_image(phase + " Recover Image", x1, step_img[phase])
                    x2 = vutils.make_grid(im_hr, normalize=True, scale_each=True)
                    writer.add_image(phase + " HR Image", x2, step_img[phase])
                    kernel_blur = util_sisr.kinfo2sigma(
                        kinfo_gt, k_size=args["k_size"], sf=args["sf"]
                    )
                    x3 = vutils.make_grid(kernel_blur, normalize=True, scale_each=True)
                    writer.add_image(phase + " GT Blur Kernel", x3, step_img[phase])
                    x4 = vutils.make_grid(im_lr, normalize=True, scale_each=True)
                    writer.add_image(phase + " LR Image", x4, step_img[phase])
                    x5 = vutils.make_grid(
                        loss_detail[7].detach(), normalize=True, scale_each=True
                    )
                    writer.add_image(
                        phase + " Est Blur Kernel Resample", x5, step_img[phase]
                    )
                    kernel_blur_est = util_sisr.kinfo2sigma(
                        kinfo_est.detach(),
                        k_size=args["k_size"],
                        sf=args["sf"],
                        shift=util_opts.str2bool(args["kernel_shift"]),
                    )
                    x6 = vutils.make_grid(
                        kernel_blur_est, normalize=True, scale_each=True
                    )
                    writer.add_image(phase + " Est Blur Kernel", x6, step_img[phase])
                    step_img[phase] += 1

        if rank == 0:
            log_str = "{:s}: Loss={:+.2e}, lh={:>4.2f}, KLR={:+>7.2f}, KLS={:+>6.2f}, KLK={:+>5.2f}"
            print(
                log_str.format(
                    phase,
                    loss_per_epoch["Loss"],
                    loss_per_epoch["lh"],
                    loss_per_epoch["KLR"],
                    loss_per_epoch["KLS"],
                    loss_per_epoch["KLK"],
                )
            )
            writer.add_scalar("Loss_epoch", loss_per_epoch["Loss"], epoch)
            print("-" * 105)

        # save model
        if rank == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            save_path_model = str(model_dir / f"model_{timestamp}_{unique_id}.pth")
            with open(save_path_model, "wb") as f:
                torch.save(
                    {
                        "metadata": {
                            "time_created": datetime.datetime.now().isoformat(),
                            "uuid_short": uuid.uuid4().hex[:8],
                        },
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "step_img": {
                            x: step_img[x]
                            for x in [
                                "train",
                            ]
                            + noise_types_list
                        },
                        "model_state_dict": net.state_dict(),
                    },
                    f,
                )
            toc = time.time()
            print("This epoch take time {:.2f}".format(toc - tic))

        # test stage
        if rank == 0:
            phase = "test"
            net.eval()
            for noise_type in noise_types_list:
                psnr_per_epoch = ssim_per_epoch = 0
                for ii, data in enumerate(test_dataloaders[noise_type]):
                    im_hr, im_lr, kinfo_gt = [x.cuda() for x in data]
                    with torch.set_grad_enabled(False):
                        # mu, kinfo_est, sigma_est = net(im_lr, args["sf"])
                        mu, kinfo_est, sigma_est = net(im_lr)
                        im_hr_est = mu[0] if isinstance(mu, list) else mu

                    psnr_iter = util_image.batch_PSNR(
                        im_hr_est, im_hr, args["sf"] ** 2, True
                    )
                    ssim_iter = util_image.batch_SSIM(
                        im_hr_est, im_hr, args["sf"] ** 2, True
                    )
                    psnr_per_epoch += psnr_iter
                    ssim_per_epoch += ssim_iter
                    # print statistics every log_interval mini_batches
                    if (ii + 1) % 3 == 0:
                        log_str = "Noise: {:s}, Epoch:{:>3d}/{:<3d}] {:s}:{:0>3d}/{:0>3d}, psnr={:4.2f}, ssim={:5.4f}"
                        print(
                            log_str.format(
                                noise_type,
                                epoch + 1,
                                args["epochs"],
                                phase,
                                ii + 1,
                                num_iter_epoch[phase],
                                psnr_iter,
                                ssim_iter,
                            )
                        )
                        # tensorboardX summary
                        x1 = vutils.make_grid(
                            im_hr_est, normalize=True, scale_each=True
                        )
                        writer.add_image(
                            "Test " + noise_type + " Recover images",
                            x1,
                            step_img[noise_type],
                        )
                        x2 = vutils.make_grid(im_hr, normalize=True, scale_each=True)
                        writer.add_image(
                            "Test " + noise_type + " HR Image", x2, step_img[noise_type]
                        )
                        x3 = vutils.make_grid(im_lr, normalize=True, scale_each=True)
                        writer.add_image(
                            "Test " + noise_type + " LR Image", x3, step_img[noise_type]
                        )
                        kernel_blur = util_sisr.kinfo2sigma(
                            kinfo_gt,
                            k_size=args["k_size"],
                            sf=args["sf"],
                            shift=util_opts.str2bool(args["kernel_shift"]),
                        )
                        x4 = vutils.make_grid(
                            kernel_blur, normalize=True, scale_each=True
                        )
                        writer.add_image(
                            "Test " + noise_type + " GT Blur Kernel",
                            x4,
                            step_img[noise_type],
                        )
                        kernel_blur_est = util_sisr.kinfo2sigma(
                            kinfo_est,
                            k_size=args["k_size"],
                            sf=args["sf"],
                            shift=util_opts.str2bool(args["kernel_shift"]),
                        )
                        x5 = vutils.make_grid(
                            kernel_blur_est, normalize=True, scale_each=True
                        )
                        writer.add_image(
                            "Test " + noise_type + " Est Blur Kernel",
                            x5,
                            step_img[noise_type],
                        )
                        step_img[noise_type] += 1

                psnr_per_epoch /= ii + 1
                ssim_per_epoch /= ii + 1
                writer.add_scalar(noise_type + " PSNR Test", psnr_per_epoch, epoch)
                writer.add_scalar(noise_type + " SSIM Test", ssim_per_epoch, epoch)
                log_str = "Noise: {:s}, {:s}: PSNR={:4.2f}, SSIM={:5.4f}"
                print(log_str.format(noise_type, phase, psnr_per_epoch, ssim_per_epoch))
                print("-" * 60)

        scheduler.step()

    if rank == 0:
        writer.close()
    if num_gpus > 1:
        close_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=0, help="Passed by launch.py")

    parser.add_argument(
        "--config",
        type=str,
        default="/home/ozkan/works/n-smoe/tpami/VIRNet/configs/local_sisr_x4.json",
        help="Path for the config file",
    )
    parser.add_argument(
        "--save_dir",
        default="/mnt/d/virnet_smoe",
        type=str,
        metavar="PATH",
        help="Path to save the log file",
    )
    opts_parser = parser.parse_args()

    main()
