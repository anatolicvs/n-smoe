import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim import lr_scheduler
from models.model_base import ModelBase
from models.select_network import define_G
from collections import OrderedDict
from monai.losses.dice import DiceLoss


class ModelSeg(ModelBase):
    def __init__(self, opt):
        super(ModelSeg, self).__init__(opt)

        self.opt_train = self.opt["train"]
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train["E_decay"] > 0:
            self.netE = define_G(opt).to(self.device).eval()

        self.iou = torch.Tensor([0.0]).to(self.device)
        self.iou.requires_grad = False

        self.b_loss: float = 1e10

    def init_train(self):
        self.load()
        self.netG.train()
        self.define_loss()
        self.define_optimizer()
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

    def _loss(
        self, prd_mask: torch.Tensor, gt_mask: torch.Tensor, prd_scores: torch.Tensor
    ):
        prd_mask = torch.sigmoid(prd_mask[:, 0])

        seg_loss = (
            -gt_mask * torch.log(prd_mask + 0.00001)
            - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)
        ).mean()

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05

        self.iou = iou
        return loss

    def define_loss(self):
        self.G_seg_loss = DiceLoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        ).to(self.device)
        self.G_ce_loss = nn.BCEWithLogitsLoss(reduction="mean").to(self.device)

    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print("Params [{:s}] will not optimize.".format(k))

        if self.opt_train["G_optimizer_type"] == "adamw":
            self.G_optimizer = AdamW(
                params=G_optim_params,
                lr=self.opt_train["G_optimizer_lr"],
                betas=self.opt_train["G_optimizer_betas"],
                weight_decay=self.opt_train["G_optimizer_wd"],
            )
        else:
            raise NotImplementedError

    def load_optimizers(self):
        load_path_optimizerG = self.opt["path"]["pretrained_optimizerG"]
        if load_path_optimizerG is not None and self.opt_train["G_optimizer_reuse"]:
            print("Loading optimizerG [{:s}] ...".format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    def define_scheduler(self):
        g_scheduler_type = self.opt_train["G_scheduler_type"]
        if g_scheduler_type == "MultiStepLR":
            self.schedulers.append(
                lr_scheduler.MultiStepLR(
                    self.G_optimizer,
                    self.opt_train["G_scheduler_milestones"],
                    self.opt_train["G_scheduler_gamma"],
                )
            )

        elif g_scheduler_type == "CyclicLR":
            self.schedulers.append(
                lr_scheduler.CyclicLR(
                    self.G_optimizer,
                    self.opt_train["G_optimizer_lr"],
                    self.opt_train["G_scheduler_max_lr"],
                    step_size_up=self.opt_train["G_scheduler_step_size_up"],
                    step_size_down=self.opt_train["G_scheduler_step_size_down"],
                    mode=self.opt_train["G_scheduler_mode"],
                    gamma=1.0,
                    cycle_momentum=self.opt_train["G_scheduler_cycle_momentum"],
                    base_momentum=0.8,
                    max_momentum=0.9,
                    last_epoch=-1,
                )
            )

        elif g_scheduler_type == "CosineAnnealingLR":
            self.schedulers.append(
                lr_scheduler.CosineAnnealingLR(
                    self.G_optimizer,
                    T_max=self.opt_train["G_scheduler_T_max"],
                    eta_min=self.opt_train["G_scheduler_eta_min"],
                )
            )
        elif g_scheduler_type == "ReduceLROnPlateau":
            self.schedulers.append(
                lr_scheduler.ReduceLROnPlateau(
                    self.G_optimizer,
                    mode="min",
                    patience=self.opt_train["G_scheduler_lr_patience"],
                    factor=0.1,
                    min_lr=self.opt_train["G_scheduler_lr_min"],
                )
            )
        else:
            raise NotImplementedError

    def save_network(
        self,
        save_dir: str,
        network: torch.nn.Module,
        network_label: str,
        iter_label: str,
    ) -> None:
        save_filename = f"{iter_label}_{network_label}.pth"
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        # state_dict = {key: param.cpu() for key, param in network.state_dict().items()}

        checkpoint = {
            "model": network.sam2.state_dict(),
        }
        torch.save(checkpoint, save_path)

    def save(self, iter_label):
        if self.log_dict["G_loss"] < self.b_loss:
            self.b_loss = self.log_dict["G_loss"]
            self.save_network(self.save_dir, self.netG, "G", iter_label)
        if self.opt_train["E_decay"] > 0:
            self.save_network(self.save_dir, self.netE, "E", iter_label)
        if self.opt_train["G_optimizer_reuse"]:
            self.save_optimizer(
                self.save_dir, self.G_optimizer, "optimizerG", iter_label
            )

    def feed_data(self, data):
        self.img = data["img"].to(self.device)
        self.box = data["box"].to(self.device)
        self.label = data["label"].to(self.device)
        # self.mask = data["mask"].to(self.device)
        # self.input_point = data["input_point"].to(self.device)

    def netG_forward(self):
        boxes_np = self.box.detach().cpu().numpy()
        self.E = self.netG(self.img, boxes_np)

    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()

        G_loss = self.G_seg_loss(self.E, self.label) + self.G_ce_loss(
            self.E, self.label.float()
        )
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
                self.parameters(),
                max_norm=self.opt_train["G_optimizer_clipgrad"],
                norm_type=2,
            )

        self.G_optimizer.step()
        self.log_dict["G_loss"] = G_loss.item()
        if self.opt_train["E_decay"] > 0:
            self.update_E(self.opt_train["E_decay"])

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    def current_log(self):
        return self.log_dict

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
