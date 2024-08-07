import argparse
import logging
import math
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Tuple, Dict, Any

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils_n import utils_image as util
from utils_n import utils_logger
from utils_n import utils_option as option
from utils_n.utils_dist import init_dist


def setup_logging(opt) -> logging.Logger | None:
    if opt["rank"] == 0:
        logger_name = "train"
        log_file = os.path.join(opt["path"]["log"], f"{logger_name}.log")
        utils_logger.logger_info(logger_name, log_file)
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))
    else:
        logger = None
    return logger


def initialize_distributed(opt: Dict[str, Any]) -> Dict[str, Any]:
    if opt["dist"]:
        init_dist("pytorch")
        opt["world_size"] = dist.get_world_size()
        opt["rank"] = dist.get_rank()
        torch.cuda.set_device(opt["rank"])
    else:
        opt["rank"], opt["world_size"] = 0, 1
    return opt


def create_data_loaders(
    opt: Dict[str, Any], logger: logging.Logger
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    train_loader, test_loader = None, None

    for phase, dataset_opt in opt["datasets"].items():
        local_batch_size = dataset_opt["dataloader_batch_size"]

        if phase == "train":
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / local_batch_size))
            if opt["rank"] == 0:
                logger.info(
                    f"Number of train images: {len(train_set):,d}, iters: {train_size:,d}"
                )
            if opt["dist"] and local_batch_size % opt["num_gpu"] != 0:
                local_batch_size = (local_batch_size // opt["num_gpu"]) * opt["num_gpu"]
                if opt["rank"] == 0:
                    logger.info(
                        f"Adjusted train batch size to {local_batch_size} for better GPU utilization"
                    )
            if opt["dist"]:
                train_sampler = DistributedSampler(
                    train_set,
                    num_replicas=opt["world_size"],
                    rank=opt["rank"],
                    shuffle=dataset_opt["dataloader_shuffle"],
                    drop_last=True,
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=local_batch_size // opt["num_gpu"],
                    shuffle=False,
                    num_workers=dataset_opt["dataloader_num_workers"] // opt["num_gpu"],
                    drop_last=True,
                    pin_memory=True,
                    sampler=train_sampler,
                    collate_fn=util.custom_collate,
                )
            else:
                train_loader = DataLoader(
                    train_set,
                    batch_size=local_batch_size,
                    shuffle=dataset_opt["dataloader_shuffle"],
                    num_workers=dataset_opt["dataloader_num_workers"],
                    drop_last=True,
                    pin_memory=True,
                    collate_fn=util.custom_collate,
                )
        elif phase == "test":
            test_set = define_Dataset(dataset_opt)
            test_size = int(math.ceil(len(test_set) / local_batch_size))
            if opt["rank"] == 0:
                logger.info(
                    f"Number of test images: {len(test_set):,d}, iters: {test_size:,d}"
                )
            if opt["dist"]:
                if local_batch_size % opt["num_gpu"] != 0:
                    local_batch_size = (local_batch_size // opt["num_gpu"]) * opt[
                        "num_gpu"
                    ]
                    if opt["rank"] == 0:
                        logger.info(
                            f"Adjusted test batch size to {local_batch_size} for consistency"
                        )
                test_sampler = DistributedSampler(
                    test_set,
                    num_replicas=opt["world_size"],
                    rank=opt["rank"],
                    shuffle=False,
                )
                test_loader = DataLoader(
                    test_set,
                    batch_size=max(1, local_batch_size // opt["num_gpu"]),
                    shuffle=False,
                    num_workers=max(
                        1, dataset_opt["dataloader_num_workers"] // opt["num_gpu"]
                    ),
                    drop_last=False,
                    pin_memory=True,
                    sampler=test_sampler,
                    collate_fn=util.custom_collate,
                )
            else:
                test_loader = DataLoader(
                    test_set,
                    batch_size=max(1, local_batch_size // 10),
                    shuffle=False,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=True,
                    collate_fn=util.custom_collate,
                )
        else:
            logger.error(f"Phase [{phase}] is not recognized.")
            raise NotImplementedError(f"Phase [{phase}] is not recognized.")

    return train_loader, test_loader


def main(json_path: str = "options/train_transformer_x2_gan_local.json"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default=json_path)
    parser.add_argument("--launcher", type=str, default="pytorch")
    parser.add_argument("--dist", action="store_true", default=False)
    parser.add_argument("--visualize", action="store_true", default=False)

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt["dist"] = args.dist
    opt["visualize"] = args.visualize

    opt = initialize_distributed(opt)

    if opt["rank"] == 0:
        util.mkdirs(
            (path for key, path in opt["path"].items() if "pretrained" not in key)
        )

    init_iter_G, init_path_G = option.find_last_checkpoint(
        opt["path"]["models"],
        net_type="G",
        pretrained_path=opt["path"]["pretrained_netG"],
    )
    init_iter_D, init_path_D = option.find_last_checkpoint(
        opt["path"]["models"],
        net_type="D",
        pretrained_path=opt["path"]["pretrained_netD"],
    )
    init_iter_E, init_path_E = option.find_last_checkpoint(
        opt["path"]["models"],
        net_type="E",
        pretrained_path=opt["path"]["pretrained_netE"],
    )
    opt["path"]["pretrained_netG"] = init_path_G
    opt["path"]["pretrained_netD"] = init_path_D
    opt["path"]["pretrained_netE"] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
        opt["path"]["models"], net_type="optimizerG"
    )
    init_iter_optimizerD, init_path_optimizerD = option.find_last_checkpoint(
        opt["path"]["models"], net_type="optimizerD"
    )
    opt["path"]["pretrained_optimizerG"] = init_path_optimizerG
    opt["path"]["pretrained_optimizerD"] = init_path_optimizerD
    current_step = max(
        init_iter_G,
        init_iter_D,
        init_iter_E,
        init_iter_optimizerG,
        init_iter_optimizerD,
    )

    border = opt["scale"]

    if opt["rank"] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)
    logger = setup_logging(opt)
    train_loader, test_loader = create_data_loaders(opt, logger)
    model = define_Model(opt)

    model.init_train()
    if opt["rank"] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    for epoch in range(4000000):
        if opt["dist"] and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        for i, train_data in enumerate(train_loader):
            if train_data is None:
                logger.warning(f"Train data is None at iteration {i} in epoch {epoch}")
                continue

            current_step += 1

            model.feed_data(train_data)

            if opt["visualize"]:
                model.visualize_data()

            model.optimize_parameters(current_step)
            model.update_learning_rate(current_step)

            if (
                current_step % opt["train"]["checkpoint_print"] == 0
                and opt["rank"] == 0
            ):
                logs = model.current_log()
                message = f"<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{model.current_learning_rate():.3e}>"
                for k, v in logs.items():
                    message += f" {k}: {v:.3e}"
                logger.info(message)

            if current_step % opt["train"]["checkpoint_save"] == 0 and opt["rank"] == 0:
                logger.info("Saving the model.")
                model.save(current_step)

            if current_step % opt["train"]["checkpoint_test"] == 0:
                local_psnr_sum = 0.0
                local_count = 0

                for test_data in test_loader:
                    if test_data is None:
                        continue

                    image_name_ext = os.path.basename(test_data["L_path"][0])

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals["E"])
                    H_img = util.tensor2uint(visuals["H"])
                    current_psnr = util.calculate_psnr(E_img, H_img, border)

                    local_psnr_sum += current_psnr
                    local_count += 1

                    if opt["rank"] == 0:
                        logger.info(
                            f"{local_count:->4d}--> {image_name_ext:>10s} | {current_psnr:<4.2f}dB"
                        )

                if opt["dist"]:
                    local_psnr_sum_tensor = torch.tensor(
                        local_psnr_sum, device=torch.device(f"cuda:{opt['rank']}")
                    )
                    local_count_tensor = torch.tensor(
                        local_count, device=torch.device(f"cuda:{opt['rank']}")
                    )
                    dist.all_reduce(local_psnr_sum_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

                    if local_count_tensor > 0:
                        global_avg_psnr = (
                            local_psnr_sum_tensor.item() / local_count_tensor.item()
                        )
                    else:
                        global_avg_psnr = 0.0

                    if opt["rank"] == 0:
                        logger.info(
                            f"<epoch:{epoch:3d}, iter:{current_step:8,d}, Average PSNR: {global_avg_psnr:.2f} dB>"
                        )
                else:
                    if local_count > 0:
                        avg_psnr = local_psnr_sum / local_count
                    else:
                        avg_psnr = 0.0

                    if opt["rank"] == 0:
                        logger.info(
                            f"<epoch:{epoch:3d}, iter:{current_step:8,d}, Average PSNR: {avg_psnr:.2f} dB>"
                        )

            if opt["dist"]:
                dist.barrier()


if __name__ == "__main__":
    main()
