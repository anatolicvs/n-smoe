# type: ignore
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

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils_n import utils_image as util
from utils_n import utils_option as option
from utils_n.utils_dist import init_dist

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(2024)

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def setup_logging(opt):
    if opt["rank"] == 0:
        logger_name = "train"
        slurm_jobid = os.getenv("SLURM_JOB_ID", "0")
        log_file = os.path.join(
            opt["path"]["log"], f"{logger_name}_slurm_{slurm_jobid}.log"
        )
        logger = logging.getLogger(logger_name)

        if not logger.hasHandlers():
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                "%(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            logger.setLevel(logging.INFO)

            logger.propagate = False

        logger.info(f"SLURM Job ID: {slurm_jobid}")
        logger.info(option.dict2str(opt))

    else:
        logger = None

    return logger


def initialize_distributed(opt):
    if opt["dist"]:
        init_dist("pytorch")
        opt["world_size"] = dist.get_world_size()
        opt["rank"] = dist.get_rank()
        torch.cuda.set_device(opt["rank"])
    else:
        opt["rank"], opt["world_size"] = 0, 1
    return opt


def build_loaders(opt, logger=None):
    def log_stats(phase, dataset, batch_size, logger):
        size = len(dataset)
        iters = math.ceil(size / batch_size)
        if opt["rank"] == 0:
            logger.info(f"{phase.capitalize()}: {size:,d} imgs, {iters:,d} iters")
        return size, iters

    def adjust_batch_size(batch_size, opt, logger=None):
        if opt["dist"]:
            gpu_ids = opt.get("gpu_ids", [0])
            num_gpu = len(gpu_ids)
            if num_gpu <= 0:
                if logger and opt["rank"] == 0:
                    logger.error("No valid GPUs found. Defaulting to 1.")
                num_gpu = 1

            if batch_size % num_gpu != 0:
                adjusted_size = max(1, (batch_size // num_gpu) * num_gpu)
                if logger and opt["rank"] == 0:
                    logger.info(
                        f"Adjusted batch size to: {adjusted_size} for {num_gpu} GPUs."
                    )
                return adjusted_size
        return batch_size

    def create_loader(
        dataset, batch_size, shuffle, workers, sampler, drop_last, pin_memory
    ):
        try:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle if sampler is None else False,
                num_workers=workers or os.cpu_count(),
                drop_last=drop_last,
                pin_memory=pin_memory,
                sampler=sampler,
                collate_fn=util.custom_collate,
            )
            return loader
        except Exception as e:
            if logger and opt["rank"] == 0:
                logger.error(f"Failed to create DataLoader: {e}")
            return None

    train_loader, test_loader = None, None

    for phase, dataset_opt in opt["datasets"].items():
        dataset = define_Dataset(dataset_opt)
        if dataset is None:
            if logger and opt["rank"] == 0:
                logger.error(f"Failed to create dataset for phase: {phase}")
            continue

        batch_size = adjust_batch_size(
            dataset_opt["dataloader_batch_size"], opt, logger
        )

        if phase == "train":
            log_stats(phase, dataset, batch_size, logger)
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
                pin_memory=torch.cuda.is_available(),
            )
            if train_loader is None and logger and opt["rank"] == 0:
                logger.error("Train loader creation failed.")

        elif phase == "test":
            log_stats(phase, dataset, batch_size, logger)
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
                pin_memory=torch.cuda.is_available(),
            )
            if test_loader is None and logger and opt["rank"] == 0:
                logger.error("Test loader creation failed.")

        else:
            if logger and opt["rank"] == 0:
                logger.error(f"Phase [{phase}] not recognized.")

    return train_loader, test_loader


def main(json_path="options/sam2/sam2.json"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default=json_path)
    parser.add_argument("--launcher", type=str, default="pytorch")
    parser.add_argument("--dist", action="store_true", default=False)

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt["dist"] = args.dist

    opt = initialize_distributed(opt)
    logger = None
    if opt["rank"] == 0:
        util.mkdirs(
            [path for key, path in opt["path"].items() if "pretrained" not in key]
        )
        logger = setup_logging(opt)

    init_iter_G, init_path_G = option.find_last_checkpoint(
        opt["path"]["models"],
        net_type="G",
        pretrained_path=opt["path"]["pretrained_netG"],
    )

    opt["path"]["pretrained_netG"] = init_path_G

    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
        opt["path"]["models"], net_type="optimizerG"
    )

    opt["path"]["pretrained_optimizerG"] = init_path_optimizerG

    current_step = max(init_iter_G, init_iter_optimizerG)

    if opt["rank"] == 0:
        option.save(opt)

    if opt["dist"]:
        train_loader, _ = build_loaders(opt, logger)
        dist.barrier()
    else:
        train_loader, _ = build_loaders(opt, logger)

    model = define_Model(opt)
    model.init_train()

    if opt["rank"] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    num_epochs = opt["train"].get("num_epochs", 4000000)
    checkpoint_interval = opt["train"].get("checkpoint_save", 1000)
    test_interval = opt["train"].get("checkpoint_test", 1000)
    log_interval = opt["train"].get("checkpoint_print", 500)

    for epoch in range(num_epochs):
        if opt["dist"] and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):
            try:
                if train_data is None and opt["rank"] == 0:
                    logger.warning(
                        f"Train data is None at iteration {i} in epoch {epoch}"
                    )
                    continue

                current_step += 1
                model.feed_data(train_data)
                model.optimize_parameters(current_step)
                model.update_learning_rate(current_step)
            except Exception as e:
                if opt["rank"] == 0:
                    logger.error(f"Error during training iteration {i}: {e}")
                continue

            if current_step % log_interval == 0 and opt["rank"] == 0:
                logs = model.current_log()
                message = f"<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{model.current_learning_rate():.3e}>"
                for k, v in logs.items():
                    message += f" {k}: {v:.3e}"
                logger.info(message)

            if current_step % checkpoint_interval == 0 and opt["rank"] == 0:
                try:
                    logger.info("Saving the model.")
                    model.save(current_step)
                except Exception as e:
                    logger.error(f"Error saving model at step {current_step}: {e}")


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
