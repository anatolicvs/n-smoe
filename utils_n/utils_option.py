import os
from collections import OrderedDict
from datetime import datetime
import json
import re
import glob


def get_timestamp():
    return datetime.now().strftime("_%y%m%d_%H%M%S")


def parse(opt_path, is_train=True):

    # ----------------------------------------
    # remove comments starting with '//'
    # ----------------------------------------
    json_str = ""
    with open(opt_path, "r") as f:
        for line in f:
            line = line.split("//")[0] + "\n"
            json_str += line

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt["opt_path"] = opt_path
    opt["is_train"] = is_train

    # ----------------------------------------
    # set default
    # ----------------------------------------
    if "merge_bn" not in opt:
        opt["merge_bn"] = False
        opt["merge_bn_startpoint"] = -1

    if "scale" not in opt:
        opt["scale"] = 1

    # ----------------------------------------
    # datasets
    # ----------------------------------------
    default_ang_res = 5
    default_kernel_path = None
    default_n_channels = 1
    default_phw = 16
    default_overlap = 14
    default_scale = 1

    for phase, dataset in opt["datasets"].items():
        phase_key = phase.split("_")[0]
        dataset.update(
            {
                "phase": phase_key,
                "scale": opt.get("scale", default_scale),
                "kernel_path": opt.get("kernel_path", default_kernel_path),
                "n_channels": opt.get("n_channels", default_n_channels),
                "ang_res": opt.get("ang_res", default_ang_res),
                "phw": opt.get("phw", default_phw),
                "overlap": opt.get("overlap", default_overlap),
            }
        )

        dataroot_H = dataset.get("dataroot_H")
        if dataroot_H:
            dataset["dataroot_H"] = dataroot_H

        if dataset.get("dataroot_L"):
            dataset["dataroot_L"] = dataroot_H

    # ----------------------------------------
    # path
    # ----------------------------------------
    for key, path in opt["path"].items():
        if path and key in opt["path"]:
            opt["path"][key] = os.path.expanduser(path)

    path_task = os.path.join(opt["path"]["root"], opt["task"])
    opt["path"]["task"] = path_task
    opt["path"]["log"] = path_task
    opt["path"]["options"] = os.path.join(path_task, "options")

    if is_train:
        opt["path"]["models"] = os.path.join(path_task, "models")
        opt["path"]["images"] = os.path.join(path_task, "images")
    else:  # test
        opt["path"]["images"] = os.path.join(path_task, "test_images")

    # ----------------------------------------
    # network
    # ----------------------------------------
    if "netG" in opt:
        opt["netG"]["scale"] = opt["scale"] if "scale" in opt else 1
        opt["netG"]["n_channels"] = opt["n_channels"] if "n_channels" in opt else 3
        opt["netG"]["ang_res"] = opt["ang_res"] if "ang_res" in opt else 5

        if "phw" and "overlap" in opt:
            opt["netG"]["phw"] = opt["phw"]
            opt["netG"]["overlap"] = opt["overlap"]

    if "netD" in opt:
        opt["netD"]["in_nc"] = opt["n_channels"] if "n_channels" in opt else 3

    # ----------------------------------------
    # GPU devices
    # ----------------------------------------
    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    # print("export CUDA_VISIBLE_DEVICES=" + gpu_list)

    # ----------------------------------------
    # default setting for distributeddataparallel
    # ----------------------------------------
    if "find_unused_parameters" not in opt:
        opt["find_unused_parameters"] = True
    if "use_static_graph" not in opt:
        opt["use_static_graph"] = False
    if "dist" not in opt:
        opt["dist"] = False
    opt["num_gpu"] = len(opt["gpu_ids"])
    # print("number of GPUs is: " + str(opt["num_gpu"]))

    # ----------------------------------------
    # default setting for perceptual loss
    # ----------------------------------------
    if "train" in opt:
        if "F_feature_layer" not in opt["train"]:
            opt["train"]["F_feature_layer"] = 34  # 25; [2,7,16,25,34]
        if "F_weights" not in opt["train"]:
            opt["train"]["F_weights"] = 1.0  # 1.0; [0.1,0.1,1.0,1.0,1.0]
        if "F_lossfn_type" not in opt["train"]:
            opt["train"]["F_lossfn_type"] = "l1"
        if "F_use_input_norm" not in opt["train"]:
            opt["train"]["F_use_input_norm"] = True
        if "F_use_range_norm" not in opt["train"]:
            opt["train"]["F_use_range_norm"] = False

    # ----------------------------------------
    # default setting for optimizer
    # ----------------------------------------
    if "train" in opt:
        if "G_optimizer_type" not in opt["train"]:
            opt["train"]["G_optimizer_type"] = "adam"
        if "G_optimizer_betas" not in opt["train"]:
            opt["train"]["G_optimizer_betas"] = [0.9, 0.999]
        if "G_scheduler_restart_weights" not in opt["train"]:
            opt["train"]["G_scheduler_restart_weights"] = 1
        if "G_optimizer_wd" not in opt["train"]:
            opt["train"]["G_optimizer_wd"] = 0
        if "G_optimizer_reuse" not in opt["train"]:
            opt["train"]["G_optimizer_reuse"] = False
        if "netD" in opt and "D_optimizer_reuse" not in opt["train"]:
            opt["train"]["D_optimizer_reuse"] = False

    # ----------------------------------------
    # default setting of strict for model loading
    # ----------------------------------------
    if "train" in opt:
        if "G_param_strict" not in opt["train"]:
            opt["train"]["G_param_strict"] = True
        if "netD" in opt and "D_param_strict" not in opt["path"]:
            opt["train"]["D_param_strict"] = True
        if "E_param_strict" not in opt["path"]:
            opt["train"]["E_param_strict"] = True

    # ----------------------------------------
    # Exponential Moving Average
    # ----------------------------------------
    if "train" in opt:
        if "E_decay" not in opt["train"]:
            opt["train"]["E_decay"] = 0

    # ----------------------------------------
    # default setting for discriminator
    # ----------------------------------------
    if "netD" in opt:
        if "net_type" not in opt["netD"]:
            opt["netD"]["net_type"] = "discriminator_patchgan"  # discriminator_unet
        if "in_nc" not in opt["netD"]:
            opt["netD"]["in_nc"] = 3
        if "base_nc" not in opt["netD"]:
            opt["netD"]["base_nc"] = 64
        if "n_layers" not in opt["netD"]:
            opt["netD"]["n_layers"] = 3
        if "norm_type" not in opt["netD"]:
            opt["netD"]["norm_type"] = "spectral"

    return opt


def find_last_checkpoint(save_dir, net_type="G", pretrained_path=None):
    """
    Args:
        save_dir: model folder
        net_type: 'G' or 'D' or 'optimizerG' or 'optimizerD'
        pretrained_path: pretrained model path. If save_dir does not have any model, load from pretrained_path

    Return:
        init_iter: iteration number
        init_path: model path
    """
    file_list = glob.glob(os.path.join(save_dir, "*_{}.pth".format(net_type)))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+)_{}.pth".format(net_type), file_)
            iter_exist.append(int(iter_current[0]))
        init_iter = max(iter_exist)
        init_path = os.path.join(save_dir, "{}_{}.pth".format(init_iter, net_type))
    else:
        init_iter = 0
        init_path = pretrained_path
    return init_iter, init_path


"""
# --------------------------------------------
# convert the opt into json file
# --------------------------------------------
"""


def save(opt):
    opt_path = opt["opt_path"]
    opt_path_copy = opt["path"]["options"]
    dirname, filename_ext = os.path.split(opt_path)
    filename, ext = os.path.splitext(filename_ext)
    dump_path = os.path.join(opt_path_copy, filename + get_timestamp() + ext)
    with open(dump_path, "w") as dump_file:
        json.dump(opt, dump_file, indent=2)


"""
# --------------------------------------------
# dict to string for logger
# --------------------------------------------
"""


def dict2str(opt, indent_l=1):
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_l * 2) + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + k + ": " + str(v) + "\n"
    return msg


"""
# --------------------------------------------
# convert OrderedDict to NoneDict,
# return None for missing key
# --------------------------------------------
"""


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None
