from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss
from models.model_base import ModelBase
from models.select_network import define_G
from utils_n.utils_model import test_mode
from utils_n.utils_regularizers import regularizer_clip, regularizer_orth


class ModelPlain(ModelBase):
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)

        self.opt_train = self.opt["train"]
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train["E_decay"] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    def init_train(self):
        self.load()
        self.netG.train()
        self.define_loss()
        self.define_optimizer()
        self.load_optimizers()
        self.define_scheduler()
        self.log_dict = OrderedDict()

    def load(self):
        load_path_G = self.opt["path"]["pretrained_netG"]
        if load_path_G is not None:
            print("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(
                load_path_G,
                self.netG,
                strict=self.opt_train["G_param_strict"],
                param_key="params",
            )
        load_path_E = self.opt["path"]["pretrained_netE"]
        if self.opt_train["E_decay"] > 0:
            if load_path_E is not None:
                print("Loading model for E [{:s}] ...".format(load_path_E))
                self.load_network(
                    load_path_E,
                    self.netE,
                    strict=self.opt_train["E_param_strict"],
                    param_key="params_ema",
                )
            else:
                print("Copying model for E ...")
                self.update_E(0)
            self.netE.eval()

    def load_optimizers(self):
        load_path_optimizerG = self.opt["path"]["pretrained_optimizerG"]
        if load_path_optimizerG is not None and self.opt_train["G_optimizer_reuse"]:
            print("Loading optimizerG [{:s}] ...".format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, "G", iter_label)
        if self.opt_train["E_decay"] > 0:
            self.save_network(self.save_dir, self.netE, "E", iter_label)
        if self.opt_train["G_optimizer_reuse"]:
            self.save_optimizer(
                self.save_dir, self.G_optimizer, "optimizerG", iter_label
            )

    def define_loss(self):
        G_lossfn_type = self.opt_train["G_lossfn_type"]
        if G_lossfn_type == "l1":
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == "l2":
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == "l2sum":
            self.G_lossfn = nn.MSELoss(reduction="sum").to(self.device)
        elif G_lossfn_type == "ssim":
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == "charbonnier":
            self.G_lossfn = CharbonnierLoss(self.opt_train["G_charbonnier_eps"]).to(
                self.device
            )
        else:
            raise NotImplementedError(
                "Loss type [{:s}] is not found.".format(G_lossfn_type)
            )
        self.G_lossfn_weight = self.opt_train["G_lossfn_weight"]

    def define_optimizer(self):
        G_optim_params = [v for k, v in self.netG.named_parameters() if v.requires_grad]
        self.G_optimizer = self.create_optimizer(
            self.opt_train["G_optimizer_type"],
            G_optim_params,
            self.opt_train.get("G_optimizer_lr", 0.001),
            self.opt_train.get("G_optimizer_betas", (0.9, 0.999)),
            self.opt_train.get("G_optimizer_wd", 0.01),
            self.opt_train.get("G_optimizer_momentum", 0.0),
        )

    def define_scheduler(self):
        self.schedulers.append(
            self.create_scheduler(
                self.opt_train["G_scheduler_type"],
                self.G_optimizer,
                self.opt_train,
                "G",
            )
        )

    def feed_data(self, data, need_H=True):
        self.L = data["L"].to(self.device)
        if self.opt["train"]["is_moe"]:
            self.L_p = data["L_p"].to(self.device)
        if need_H:
            self.H = data["H"].to(self.device)

    def netG_forward(self):
        if self.opt["train"]["is_moe"]:
            self.E = self.netG(self.L_p, self.L.size())
        else:
            self.E = self.netG(self.L)

    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()

        self.H = self.H.to(self.E.device)

        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = (
            self.opt_train["G_optimizer_clipgrad"]
            if self.opt_train["G_optimizer_clipgrad"]
            else 0
        )
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.netG.parameters(),
                max_norm=self.opt_train["G_optimizer_clipgrad"],
                norm_type=2,
            )

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = (
            self.opt_train["G_regularizer_orthstep"]
            if self.opt_train["G_regularizer_orthstep"]
            else 0
        )
        if (
            G_regularizer_orthstep > 0
            and current_step % G_regularizer_orthstep == 0
            and current_step % self.opt["train"]["checkpoint_save"] != 0
        ):
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = (
            self.opt_train["G_regularizer_clipstep"]
            if self.opt_train["G_regularizer_clipstep"]
            else 0
        )
        if (
            G_regularizer_clipstep > 0
            and current_step % G_regularizer_clipstep == 0
            and current_step % self.opt["train"]["checkpoint_save"] != 0
        ):
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict["G_loss"] = G_loss.item()
        # self.log("G_loss", G_loss.item())

        if self.opt_train["E_decay"] > 0:
            self.update_E(self.opt_train["E_decay"])

        torch.cuda.synchronize()

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        # self.synchronize() # ERROR - Error during testing: Detected mismatch between collectives on ranks. Rank 0 is running collective: CollectiveFingerPrint(SequenceNumber=1649OpType=BARRIER), but Rank 1 is running collective: CollectiveFingerPrint(SequenceNumber=0OpType=GATHER).Collectives differ in the following aspects: 	 Sequence number: 1649vs 0  Op type: BARRIERvs GATHER
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            if self.opt["train"]["is_moe"]:
                self.E = test_mode(
                    self.netG,
                    self.L_p,
                    self.L.size(),
                    mode=3,
                    sf=self.opt["scale"],
                    modulo=1,
                )
            else:
                self.E = test_mode(
                    self.netG, self.L, mode=3, sf=self.opt["scale"], modulo=1
                )
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict["L"] = self.L.detach()[0].float().cpu()
        out_dict["E"] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict["H"] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict["L"] = self.L.detach().float().cpu()
        out_dict["E"] = self.E.detach().float().cpu()
        if need_H:
            out_dict["H"] = self.H.detach().float().cpu()
        return out_dict

    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg

    def synchronize(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # def log(self, key, value) -> None:
    #     wandb.log({key: value})
