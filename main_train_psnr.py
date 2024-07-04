import os.path
import math
import argparse

import random
import numpy as np

import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils_n import utils_logger
from utils_n import utils_image as util
from utils_n import utils_option as option
from utils_n.utils_dist import get_dist_info, init_dist
import torch.distributed as dist
from data.select_dataset import define_Dataset
from models.select_model import define_Model


def main(json_path="options/train_unet_moex1_psnr_local.json"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt", type=str, default=json_path, help="Path to option JSON file."
    )
    parser.add_argument("--launcher", default="pytorch", help="job launcher")
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
    )
    parser.add_argument(
        "--dist", default=False, action="store_true", help="Use distributed training"
    )

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt["dist"] = args.dist

    if opt["dist"]:
        init_dist("pytorch")
        opt["rank"], opt["world_size"] = get_dist_info()
    else:
        opt["rank"], opt["world_size"] = 0, 1

    if opt["rank"] == 0:
        util.mkdirs(
            (path for key, path in opt["path"].items() if "pretrained" not in key)
        )

    init_iter_G, init_path_G = option.find_last_checkpoint(
        opt["path"]["models"], net_type="G"
    )
    init_iter_E, init_path_E = option.find_last_checkpoint(
        opt["path"]["models"], net_type="E"
    )
    opt["path"]["pretrained_netG"] = init_path_G
    opt["path"]["pretrained_netE"] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
        opt["path"]["models"], net_type="optimizerG"
    )
    opt["path"]["pretrained_optimizerG"] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt["scale"]

    if opt["rank"] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)

    if opt["rank"] == 0:
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
        if phase == "train":
            train_set = define_Dataset(dataset_opt)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt["dataloader_batch_size"])
            )
            if opt["rank"] == 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
            if opt["dist"]:
                train_sampler = DistributedSampler(
                    train_set,
                    shuffle=dataset_opt["dataloader_shuffle"],
                    drop_last=True,
                    seed=seed,
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=dataset_opt["dataloader_batch_size"] // opt["num_gpu"],
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
                    batch_size=dataset_opt["dataloader_batch_size"],
                    shuffle=dataset_opt["dataloader_shuffle"],
                    num_workers=dataset_opt["dataloader_num_workers"],
                    drop_last=True,
                    pin_memory=True,
                    collate_fn=util.custom_collate,
                )

        elif phase == "test":
            test_set = define_Dataset(dataset_opt)
            test_size = int(
                math.ceil(len(test_set) / dataset_opt["dataloader_batch_size"])
            )
            if opt["rank"] == 0:
                logger.info(
                    "Number of test images in [{:s}]: {:,d}, iters: {:,d}".format(
                        dataset_opt["name"], len(test_set), test_size
                    )
                )
            if opt["dist"]:
                test_sampler = DistributedSampler(
                    test_set,
                    num_replicas=opt["world_size"],
                    rank=opt["rank"],
                    shuffle=False,
                )
                test_loader = DataLoader(
                    test_set,
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=True,
                    sampler=test_sampler,
                    collate_fn=util.custom_collate,
                )
            else:
                test_loader = DataLoader(
                    test_set,
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=True,
                    collate_fn=util.custom_collate,
                )
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    model = define_Model(opt)
    model.init_train()
    if opt["rank"] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    for epoch in range(1000000):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            if train_data is None:
                logger.warning(f"train_data is None at iteration {i} in epoch {epoch}")
                continue

            current_step += 1

            model.feed_data(train_data)

            model.optimize_parameters(current_step)

            model.update_learning_rate(current_step)

            if (
                current_step % opt["train"]["checkpoint_print"] == 0
                and opt["rank"] == 0
            ):
                logs = model.current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.3e} ".format(k, v)
                logger.info(message)

            if current_step % opt["train"]["checkpoint_save"] == 0 and opt["rank"] == 0:
                logger.info("Saving the model.")
                model.save(current_step)

            if current_step % opt["train"]["checkpoint_test"] == 0:
                local_psnr_sum = 0.0
                local_count = 0

                if "test_loader" in locals():  # Check if test_loader is defined
                    for test_data in test_loader:
                        if test_data is None:
                            continue

                        model.feed_data(test_data)
                        model.test()

                        visuals = model.current_visuals()
                        E_img = util.tensor2uint(visuals["E"])
                        H_img = util.tensor2uint(visuals["H"])

                        current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                        local_psnr_sum += current_psnr
                        local_count += 1

                    if opt[
                        "dist"
                    ]:  # If distributed, reduce the PSNR sum and count across all GPUs
                        local_psnr_sum_tensor = torch.tensor(local_psnr_sum).cuda()
                        local_count_tensor = torch.tensor(local_count).cuda()
                        dist.all_reduce(local_psnr_sum_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)
                        if opt["rank"] == 0:
                            global_avg_psnr = (
                                local_psnr_sum_tensor / local_count_tensor
                                if local_count_tensor > 0
                                else torch.tensor(0.0).cuda()
                            )
                            logger.info(
                                f"<epoch:{epoch:3d}, iter:{current_step:8,d}, Average PSNR: {global_avg_psnr.item():.2f} dB>"
                            )
                    else:  # For single-GPU, just compute the average locally
                        if local_count > 0:
                            avg_psnr = local_psnr_sum / local_count
                            logger.info(
                                f"<epoch:{epoch:3d}, iter:{current_step:8,d}, Average PSNR: {avg_psnr:.2f} dB>"
                            )


if __name__ == "__main__":
    main()


# def main(json_path="options/train_unet_moex1_psnr_local.json"):

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--opt", type=str, default=json_path, help="Path to option JSON file."
#     )
#     parser.add_argument("--launcher", default="pytorch", help="job launcher")
#     parser.add_argument(
#         "--local_rank",
#         "--local-rank",
#         type=int,
#         default=int(os.environ.get("LOCAL_RANK", 0)),
#     )
#     parser.add_argument("--dist", default=False)

#     opt = option.parse(parser.parse_args().opt, is_train=True)
#     opt["dist"] = parser.parse_args().dist

#     if opt["dist"]:
#         init_dist("pytorch")
#         opt["rank"], opt["world_size"] = get_dist_info()
#     else:
#         opt["rank"], opt["world_size"] = 0, 1

#     if opt["rank"] == 0:
#         util.mkdirs(
#             (path for key, path in opt["path"].items() if "pretrained" not in key)
#         )

#     init_iter_G, init_path_G = option.find_last_checkpoint(
#         opt["path"]["models"], net_type="G"
#     )
#     init_iter_E, init_path_E = option.find_last_checkpoint(
#         opt["path"]["models"], net_type="E"
#     )
#     opt["path"]["pretrained_netG"] = init_path_G
#     opt["path"]["pretrained_netE"] = init_path_E
#     init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
#         opt["path"]["models"], net_type="optimizerG"
#     )
#     opt["path"]["pretrained_optimizerG"] = init_path_optimizerG
#     current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

#     border = opt["scale"]

#     if opt["rank"] == 0:
#         option.save(opt)

#     opt = option.dict_to_nonedict(opt)

#     if opt["rank"] == 0:
#         logger_name = "train"
#         utils_logger.logger_info(
#             logger_name, os.path.join(opt["path"]["log"], logger_name + ".log")
#         )
#         logger = logging.getLogger(logger_name)
#         logger.info(option.dict2str(opt))

#     seed = opt["train"]["manual_seed"]
#     if seed is None:
#         seed = random.randint(1, 10000)
#     print("Random seed: {}".format(seed))
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     for phase, dataset_opt in opt["datasets"].items():
#         if phase == "train":
#             train_set = define_Dataset(dataset_opt)
#             train_size = int(
#                 math.ceil(len(train_set) / dataset_opt["dataloader_batch_size"])
#             )
#             if opt["rank"] == 0:
#                 logger.info(
#                     "Number of train images: {:,d}, iters: {:,d}".format(
#                         len(train_set), train_size
#                     )
#                 )
#             if opt["dist"]:

#                 train_sampler = DistributedSampler(
#                     train_set,
#                     shuffle=dataset_opt["dataloader_shuffle"],
#                     drop_last=True,
#                     seed=seed,
#                 )
#                 train_loader = DataLoader(
#                     train_set,
#                     batch_size=dataset_opt["dataloader_batch_size"] // opt["num_gpu"],
#                     shuffle=False,
#                     num_workers=dataset_opt["dataloader_num_workers"] // opt["num_gpu"],
#                     drop_last=True,
#                     pin_memory=True,
#                     sampler=train_sampler,
#                     collate_fn=util.custom_collate,
#                 )
#             else:
#                 train_loader = DataLoader(
#                     train_set,
#                     batch_size=dataset_opt["dataloader_batch_size"],
#                     shuffle=dataset_opt["dataloader_shuffle"],
#                     num_workers=dataset_opt["dataloader_num_workers"],
#                     drop_last=True,
#                     pin_memory=True,
#                     collate_fn=util.custom_collate,
#                 )

#         elif phase == "test":
#             test_set = define_Dataset(dataset_opt)
#             test_size = int(
#                 math.ceil(len(test_set) / dataset_opt["dataloader_batch_size"])
#             )
#             if opt["rank"] == 0:
#                 logger.info(
#                     "Number of test images in [{:s}]: {:,d}, iters: {:,d}".format(
#                         dataset_opt["name"], len(test_set), test_size
#                     )
#                 )
#             if opt["dist"]:
#                 test_sampler = DistributedSampler(
#                     test_set,
#                     num_replicas=opt["world_size"],
#                     rank=opt["rank"],
#                     shuffle=False,
#                 )
#                 test_loader = DataLoader(
#                     test_set,
#                     batch_size=1,
#                     shuffle=False,
#                     num_workers=1,
#                     drop_last=False,
#                     pin_memory=True,
#                     sampler=test_sampler,
#                     collate_fn=util.custom_collate,
#                 )
#             else:
#                 test_loader = DataLoader(
#                     test_set,
#                     batch_size=1,
#                     shuffle=False,
#                     num_workers=1,
#                     drop_last=False,
#                     pin_memory=True,
#                     collate_fn=util.custom_collate,
#                 )
#         else:
#             raise NotImplementedError("Phase [%s] is not recognized." % phase)

#     model = define_Model(opt)
#     model.init_train()
#     if opt["rank"] == 0:
#         logger.info(model.info_network())
#         logger.info(model.info_params())

#     for epoch in range(1000000):
#         if opt["dist"]:
#             train_sampler.set_epoch(epoch)

#         for i, train_data in enumerate(train_loader):

#             current_step += 1

#             model.feed_data(train_data)

#             model.optimize_parameters(current_step)

#             model.update_learning_rate(current_step)

#             if (
#                 current_step % opt["train"]["checkpoint_print"] == 0
#                 and opt["rank"] == 0
#             ):
#                 logs = model.current_log()
#                 message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
#                     epoch, current_step, model.current_learning_rate()
#                 )
#                 for k, v in logs.items():
#                     message += "{:s}: {:.3e} ".format(k, v)
#                 logger.info(message)

#             if current_step % opt["train"]["checkpoint_save"] == 0 and opt["rank"] == 0:
#                 logger.info("Saving the model.")
#                 model.save(current_step)

#             if current_step % opt["train"]["checkpoint_test"] == 0:
#                 local_psnr_sum = 0.0
#                 local_count = 0

#                 for test_data in test_loader:
#                     if test_data is None:
#                         continue

#                     model.feed_data(test_data)
#                     model.test()

#                     visuals = model.current_visuals()
#                     E_img = util.tensor2uint(visuals["E"])
#                     H_img = util.tensor2uint(visuals["H"])

#                     current_psnr = util.calculate_psnr(E_img, H_img, border=border)
#                     local_psnr_sum += current_psnr
#                     local_count += 1

#                 if opt[
#                     "dist"
#                 ]:  # If distributed, reduce the PSNR sum and count across all GPUs
#                     local_psnr_sum_tensor = torch.tensor(local_psnr_sum).cuda()
#                     local_count_tensor = torch.tensor(local_count).cuda()
#                     dist.all_reduce(local_psnr_sum_tensor, op=dist.ReduceOp.SUM)
#                     dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)
#                     if opt["rank"] == 0:
#                         global_avg_psnr = (
#                             local_psnr_sum_tensor / local_count_tensor
#                             if local_count_tensor > 0
#                             else torch.tensor(0.0).cuda()
#                         )
#                         logger.info(
#                             f"<epoch:{epoch:3d}, iter:{current_step:8,d}, Average PSNR: {global_avg_psnr.item():.2f} dB>"
#                         )
#                 else:  # For single-GPU, just compute the average locally
#                     if local_count > 0:
#                         avg_psnr = local_psnr_sum / local_count
#                         logger.info(
#                             f"<epoch:{epoch:3d}, iter:{current_step:8,d}, Average PSNR: {avg_psnr:.2f} dB>"
#                         )

#             # region test for single gpu
#             # if current_step % opt["train"]["checkpoint_test"] == 0 and opt["rank"] == 0:

#             #     avg_psnr = 0.0
#             #     idx = 0

#             #     for test_data in test_loader:
#             #         if test_data is None:
#             #             continue

#             #         idx += 1
#             #         image_name_ext = os.path.basename(test_data["L_path"][0])

#             #         # img_name, ext = os.path.splitext(image_name_ext)
#             #         # img_dir = os.path.join(opt["path"]["images"], img_name)
#             #         # util.mkdir(img_dir)

#             #         model.feed_data(test_data)
#             #         model.test()

#             #         visuals = model.current_visuals()
#             #         E_img = util.tensor2uint(visuals["E"])
#             #         H_img = util.tensor2uint(visuals["H"])

#             #         # -----------------------
#             #         # save estimated image E
#             #         # -----------------------
#             #         # save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
#             #         # util.imsave(E_img, save_img_path)

#             #         # -----------------------
#             #         # calculate PSNR
#             #         # -----------------------
#             #         current_psnr = util.calculate_psnr(E_img, H_img, border=border)

#             #         logger.info(
#             #             "{:->4d}--> {:>10s} | {:<4.2f}dB".format(
#             #                 idx, image_name_ext, current_psnr
#             #             )
#             #         )

#             #         avg_psnr += current_psnr

#             #     avg_psnr = avg_psnr / idx

#             #     logger.info(
#             #         "<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n".format(
#             #             epoch, current_step, avg_psnr
#             #         )
#             #     )
#             # endregion


# if __name__ == "__main__":
#     main()
