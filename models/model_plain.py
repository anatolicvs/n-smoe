from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss

from utils_n.utils_model import test_mode
from utils_n.utils_regularizers import regularizer_orth, regularizer_clip
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.ndimage import zoom
from scipy.stats import entropy
from scipy.signal import correlate
from matplotlib.table import Table
from scipy.fftpack import fftshift, fft2


class ModelPlain(ModelBase):
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)

        self.opt_train = self.opt["train"]
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train["E_decay"] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    def init_train(self):
        self.load()  # load model
        self.netG.train()  # set training mode,for BN
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.load_optimizers()  # load optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = OrderedDict()  # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
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

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
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
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print("Params [{:s}] will not optimize.".format(k))
        if self.opt_train["G_optimizer_type"] == "adam":
            self.G_optimizer = Adam(
                G_optim_params,
                lr=self.opt_train["G_optimizer_lr"],
                betas=self.opt_train["G_optimizer_betas"],
                weight_decay=self.opt_train["G_optimizer_wd"],
            )
        else:
            raise NotImplementedError

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
        elif g_scheduler_type == "CosineAnnealingLR":
            self.schedulers.append(
                lr_scheduler.CosineAnnealingLR(
                    self.G_optimizer,
                    T_max=self.opt_train["G_scheduler_T_max"],
                    eta_min=self.opt_train["G_scheduler_eta_min"],
                )
            )
        else:
            raise NotImplementedError

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data["L"].to(self.device)
        if self.opt["train"]["is_moe"]:
            self.L_p = data["L_p"].to(self.device)
        if need_H:
            self.H = data["H"].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        if self.opt["train"]["is_moe"]:
            self.E = self.netG(self.L_p, self.L.size())
        else:
            self.E = self.netG(self.L)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
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
                self.parameters(),
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

        if self.opt_train["E_decay"] > 0:
            self.update_E(self.opt_train["E_decay"])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
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

    def visualize_data(self):
        L_images = self.L.cpu().numpy()
        H_images = self.H.cpu().numpy()

        num_pairs = L_images.shape[0]
        fig = plt.figure(figsize=(24, num_pairs * 12))
        gs = GridSpec(
            num_pairs * 12, 6, figure=fig
        )  # Adjust grid spec to allow for better layout

        for i in range(num_pairs):
            L_np_stochastic = L_images[i][0]
            H_np = H_images[i][0]
            L_np_bicubic = zoom(H_np, 0.5, order=3)

            L_freq_stochastic = fftshift(fft2(L_np_stochastic))
            H_freq = fftshift(fft2(H_np))
            L_freq_bicubic = fftshift(fft2(L_np_bicubic))

            L_signal_stochastic_x = np.sum(
                np.log(np.abs(L_freq_stochastic) + 1), axis=0
            )
            L_signal_stochastic_y = np.sum(
                np.log(np.abs(L_freq_stochastic) + 1), axis=1
            )
            H_signal_x = np.sum(np.log(np.abs(H_freq) + 1), axis=0)
            H_signal_y = np.sum(np.log(np.abs(H_freq) + 1), axis=1)
            L_signal_bicubic_x = np.sum(np.log(np.abs(L_freq_bicubic) + 1), axis=0)
            L_signal_bicubic_y = np.sum(np.log(np.abs(L_freq_bicubic) + 1), axis=1)

            metrics = {}
            for label, signal_x, signal_y in [
                ("High Res. Ground Truth", H_signal_x, H_signal_y),
                (
                    "Low Res. Stochastic Deg",
                    L_signal_stochastic_x,
                    L_signal_stochastic_y,
                ),
                ("Low Res. Bicubic Deg.", L_signal_bicubic_x, L_signal_bicubic_y),
            ]:
                metrics[label] = {
                    "Energy X": np.sum(signal_x**2),
                    "Energy Y": np.sum(signal_y**2),
                    "Entropy X": entropy(signal_x, base=2),
                    "Entropy Y": entropy(signal_y, base=2),
                    "Corr X with H": (
                        np.max(correlate(signal_x, H_signal_x))
                        if label != "High Res. Ground Truth"
                        else "-"
                    ),
                    "Corr Y with H": (
                        np.max(correlate(signal_y, H_signal_y))
                        if label != "High Res. Ground Truth"
                        else "-"
                    ),
                }

            image_row = i * 12
            spectra_row = image_row + 2
            freq_spectra_row = spectra_row + 2
            table_row = freq_spectra_row + 2

            ax0 = fig.add_subplot(gs[image_row : image_row + 2, 0])
            ax0.imshow(L_np_stochastic, cmap="gray")
            ax0.set_title("Low Res. Stochastic Deg")

            ax1 = fig.add_subplot(gs[image_row : image_row + 2, 1])
            ax1.imshow(L_np_bicubic, cmap="gray")
            ax1.set_title("Low Res. Bicubic Deg.")

            ax2 = fig.add_subplot(gs[image_row : image_row + 2, 2])
            ax2.imshow(H_np, cmap="gray")
            ax2.set_title("High Res. Ground Truth")

            ax3 = fig.add_subplot(gs[spectra_row, 0])
            ax3.plot(L_signal_stochastic_x)
            ax3.set_title("Low Stochastic Deg. X-Spectrum")

            ax4 = fig.add_subplot(gs[spectra_row, 1])
            ax4.plot(L_signal_bicubic_x)
            ax4.set_title("Low Res Bicubic Deg. X-Spectrum")

            ax5 = fig.add_subplot(gs[spectra_row, 2])
            ax5.plot(H_signal_x)
            ax5.set_title("High Res X-Spectrum")

            ax6 = fig.add_subplot(gs[spectra_row + 1, 0])
            ax6.plot(L_signal_stochastic_y)
            ax6.set_title("Low Res Stochastic Deg. Y-Spectrum")

            ax7 = fig.add_subplot(gs[spectra_row + 1, 1])
            ax7.plot(L_signal_bicubic_y)
            ax7.set_title("Low Res Bicubic Deg. Y-Spectrum")

            ax8 = fig.add_subplot(gs[spectra_row + 1, 2])
            ax8.plot(H_signal_y)
            ax8.set_title("High Res Y-Spectrum")

            ax9 = fig.add_subplot(gs[freq_spectra_row : freq_spectra_row + 2, 0])
            ax9.imshow(np.log(np.abs(L_freq_stochastic) + 1), cmap="gray")
            ax9.set_title("Low Res Stochastic Deg. 2D Spectrum")

            ax10 = fig.add_subplot(gs[freq_spectra_row : freq_spectra_row + 2, 1])
            ax10.imshow(np.log(np.abs(L_freq_bicubic) + 1), cmap="gray")
            ax10.set_title("Low Res Bicubic Deg. 2D Spectrum")

            ax11 = fig.add_subplot(gs[freq_spectra_row : freq_spectra_row + 2, 2])
            ax11.imshow(np.log(np.abs(H_freq) + 1), cmap="gray")
            ax11.set_title("High Res 2D Spectrum")

            ax_table = fig.add_subplot(gs[table_row : table_row + 2, :])
            table = Table(ax_table, bbox=[0, 0, 1, 1])
            row_labels = [
                "Metric",
                "Energy X",
                "Energy Y",
                "Entropy X",
                "Entropy Y",
                "Corr X with H",
                "Corr Y with H",
            ]
            col_labels = [
                "Low Res. Stochastic Deg",
                "Low Res. Bicubic Deg.",
                "High Res. Ground Truth",
            ]

            cell_height = 0.1
            cell_width = 0.25

            for i, row_label in enumerate(row_labels):
                for j, col_label in enumerate([""] + col_labels):
                    if i == 0:
                        table.add_cell(
                            i,
                            j,
                            text=col_label,
                            width=cell_width,
                            height=cell_height,
                            loc="center",
                            facecolor="gray",
                        )
                    else:
                        if j == 0:
                            table.add_cell(
                                i,
                                j,
                                text=row_label,
                                width=cell_width,
                                height=cell_height,
                                loc="center",
                                facecolor="gray",
                            )
                        else:
                            value = metrics[col_labels[j - 1]][row_label]
                            formatted_value = (
                                value if isinstance(value, str) else f"{value:.2e}"
                            )
                            table.add_cell(
                                i,
                                j,
                                text=formatted_value,
                                width=cell_width,
                                height=cell_height,
                                loc="center",
                            )

            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)
            ax_table.add_table(table)
            ax_table.axis("off")

        plt.tight_layout()
        plt.show()
