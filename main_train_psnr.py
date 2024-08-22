import argparse
import logging
import math
import os
import sys
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from data.select_dataset import define_Dataset
from models.model_gan import ModelGAN
from models.model_plain import ModelPlain
from models.model_plain2 import ModelPlain2
from models.model_plain4 import ModelPlain4
from models.model_vrt import ModelVRT
from models.select_model import define_Model
from utils_n import utils_image as util
from utils_n import utils_option as option
from utils_n.utils_dist import init_dist


def setup_logging(opt) -> Optional[logging.Logger]:
    if opt["rank"] == 0:
        logger_name = "train"
        slurm_jobid = os.getenv("SLURM_JOB_ID", "0")
        log_file = os.path.join(
            opt["path"]["log"], f"{logger_name}_slurm_{slurm_jobid}.log"
        )
        logger = logging.getLogger(logger_name)

        if logger.hasHandlers():
            logger.handlers.clear()

        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        logger.propagate = False

        logger.info(f"SLURM Job ID: {slurm_jobid}")
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


def build_loaders(
    opt: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:

    def log_stats(phase: str, dataset: torch.utils.data.Dataset, batch_size: int):
        size = len(dataset)
        iters = math.ceil(size / batch_size)
        if opt["rank"] == 0:
            logger.info(f"{phase.capitalize()}: {size:,d} imgs, {iters:,d} iters")
        return size, iters

    def create_loader(dataset, batch_size, shuffle, workers, sampler, drop_last):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=workers,
            drop_last=drop_last,
            pin_memory=True,
            sampler=sampler,
            collate_fn=util.custom_collate,
        )

    def adjust_batch_size(batch_size: int) -> int:
        if opt["dist"] and batch_size % opt["num_gpu"] != 0:
            adjusted_size = max(1, (batch_size // opt["num_gpu"]) * opt["num_gpu"])
            if opt["rank"] == 0:
                logger.info(f"Adjusted batch size: {adjusted_size}")
            return adjusted_size
        return batch_size

    train_loader, test_loader = None, None

    for phase, dataset_opt in opt["datasets"].items():
        dataset = define_Dataset(dataset_opt)
        batch_size = adjust_batch_size(dataset_opt["dataloader_batch_size"])

        if phase == "train":
            log_stats(phase, dataset, batch_size)
            sampler = (
                DistributedSampler(
                    dataset,
                    num_replicas=opt["world_size"],
                    rank=opt["rank"],
                    shuffle=dataset_opt["dataloader_shuffle"],
                    drop_last=True,
                )
                if opt["dist"]
                else None
            )
            train_loader = create_loader(
                dataset,
                batch_size,
                shuffle=dataset_opt["dataloader_shuffle"],
                workers=dataset_opt["dataloader_num_workers"],
                sampler=sampler,
                drop_last=True,
            )

        elif phase == "test":
            log_stats(phase, dataset, batch_size)
            sampler = (
                DistributedSampler(
                    dataset,
                    num_replicas=opt["world_size"],
                    rank=opt["rank"],
                    shuffle=False,
                    drop_last=False,
                )
                if opt["dist"]
                else None
            )
            test_loader = create_loader(
                dataset,
                batch_size,
                shuffle=False,
                workers=dataset_opt["dataloader_num_workers"],
                sampler=sampler,
                drop_last=False,
            )

        else:
            raise NotImplementedError(f"Phase [{phase}] not recognized.")

    return train_loader, test_loader


def main(json_path: str = "options/smoe/train_unet_moex3_psnr_local.json"):
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
    logger = None
    if opt["rank"] == 0:
        util.mkdirs(
            (path for key, path in opt["path"].items() if "pretrained" not in key)
        )
        logger = setup_logging(opt)

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

    if opt["dist"]:
        train_loader, test_loader = build_loaders(opt, logger)
        dist.barrier()
    else:
        train_loader, test_loader = build_loaders(opt, logger)

    model: ModelPlain2 | ModelPlain4 | ModelGAN | ModelPlain | ModelVRT = define_Model(
        opt
    )
    model.init_train()

    if opt["rank"] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    num_epochs = 4000000
    checkpoint_interval = opt["train"].get("checkpoint_save", 1000)
    test_interval = opt["train"].get("checkpoint_test", 1000)
    log_interval = opt["train"].get("checkpoint_print", 100)

    for epoch in range(num_epochs):
        if (
            opt["dist"]
            and train_loader is not None
            and isinstance(train_loader.sampler, DistributedSampler)
        ):
            train_loader.sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):
            if train_data is None:
                if opt["rank"] == 0:
                    logger.warning(
                        f"Train data is None at iteration {i} in epoch {epoch}"
                    )
                continue

            current_step += 1
            model.feed_data(train_data)

            if opt["visualize"]:
                model.visualize_data()

            model.optimize_parameters(current_step)
            model.update_learning_rate(current_step)

            if i % log_interval == 0 and opt["rank"] == 0:
                logs = model.current_log()
                message = f"<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{model.current_learning_rate():.3e}>"
                for k, v in logs.items():
                    message += f" {k}: {v:.3e}"
                logger.info(message)

        if opt["dist"] and epoch % test_interval == 0:
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

                global_avg_psnr = (
                    (local_psnr_sum_tensor.item() / local_count_tensor.item())
                    if local_count_tensor.item() > 0
                    else 0.0
                )

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

            model.train()

        if opt["dist"] and epoch % checkpoint_interval == 0:
            if opt["rank"] == 0:
                model.save(current_step)
            dist.barrier()

    if opt["dist"]:
        dist.barrier()


if __name__ == "__main__":
    main()
