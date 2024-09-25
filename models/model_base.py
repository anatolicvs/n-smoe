import os
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.modules.module import Module
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import SGD, Adam, Adamax, AdamW, RAdam, RMSprop, lr_scheduler
from torch_optimizer import Lookahead

from utils_n.utils_bnorm import merge_bn, tidy_sequential


class ModelBase(ABC):
    def __init__(self, opt: dict) -> None:
        self.opt = opt
        self.save_dir = opt["path"]["models"]

        if opt["dist"]:
            local_rank = int(os.environ.get("LOCAL_RANK", opt["rank"]))
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.is_train = opt["is_train"]
        self.schedulers = []

    @abstractmethod
    def init_train(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def define_loss(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def define_optimizer(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def define_scheduler(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def feed_data(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def optimize_parameters(self, *args, **kwargs) -> Any:
        pass

    def current_visuals(self) -> dict:
        return {}

    def current_losses(self) -> dict:
        return {}

    def update_learning_rate(self, n: int) -> Any:
        for scheduler in self.schedulers:
            scheduler.step()

    def current_learning_rate(self) -> float:
        if self.schedulers and self.schedulers[0].get_last_lr():
            return self.schedulers[0].get_last_lr()[0]
        return 0.0

    def requires_grad(self, model: torch.nn.Module, flag: bool = True) -> None:
        for p in model.parameters():
            p.requires_grad = flag

    @abstractmethod
    def print_network(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def info_network(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def print_params(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def info_params(self, *args, **kwargs) -> Any:
        pass

    def get_bare_model(self, network: torch.nn.Module) -> torch.nn.Module:
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def model_to_device(self, network: torch.nn.Module) -> torch.nn.Module:
        network = network.to(self.device)
        if self.opt.get("dist", False):
            device_id = self.device.index
            print(f"Assigning Rank {dist.get_rank()} to GPU {device_id}")
            network = DistributedDataParallel(
                network,
                device_ids=[device_id],
                output_device=device_id,
                find_unused_parameters=False,
            )
            if self.opt.get("use_static_graph", False):
                self._set_static_graph(network)
        else:
            network = DataParallel(network)
        return network

    def describe_network(self, network: torch.nn.Module) -> str:
        network = self.get_bare_model(network)
        msg = (
            f"Networks on Device: {next(network.parameters()).device} \n"
            f"Networks name: {network.__class__.__name__}\n"
            f"Params number: {sum(p.numel() for p in network.parameters())}\n"
            f"Net structure:\n{network}\n"
        )
        return msg

    def describe_params(self, network: torch.nn.Module) -> str:
        network = self.get_bare_model(network)
        msg = (
            " | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}".format(
                "mean", "min", "max", "std", "shape", "param_name"
            )
            + "\n"
        )
        for name, param in network.state_dict().items():
            if "num_batches_tracked" not in name:
                v = param.data.clone().float()
                std_value = v.std().item() if v.numel() > 1 else float("nan")
                msg += " | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}\n".format(
                    v.mean().item(),
                    v.min().item(),
                    v.max().item(),
                    std_value,
                    v.shape,
                    name,
                )
        return msg

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
        state_dict = {key: param.cpu() for key, param in network.state_dict().items()}
        torch.save(state_dict, save_path)

    def load_network(
        self,
        load_path: str,
        network: torch.nn.Module,
        strict: bool = True,
        param_key: str = None,
    ) -> None:
        network = self.get_bare_model(network)
        state_dict = torch.load(load_path, map_location=self.device, weights_only=True)
        if param_key and param_key in state_dict:
            state_dict = state_dict[param_key]
        network.load_state_dict(state_dict, strict=strict)

    def save_optimizer(
        self,
        save_dir: str,
        optimizer: torch.optim.Optimizer,
        optimizer_label: str,
        iter_label: str,
    ) -> None:
        save_filename = f"{iter_label}_{optimizer_label}.pth"
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(optimizer, Lookahead):
            state = {
                "base_optimizer_state_dict": optimizer.optimizer.state_dict(),
                "lookahead_state_dict": {
                    "state": optimizer.state,
                    "param_groups": optimizer.param_groups,
                },
            }
        else:
            state = optimizer.state_dict()
        torch.save(state, save_path)

    def load_optimizer(self, load_path: str, optimizer: torch.optim.Optimizer) -> None:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Optimizer state file not found: {load_path}")

        state = torch.load(load_path, map_location=self.device, weights_only=False)

        if isinstance(optimizer, Lookahead):
            if "base_optimizer_state_dict" in state and "lookahead_state_dict" in state:
                optimizer.optimizer.load_state_dict(state["base_optimizer_state_dict"])
                lookahead_state = state["lookahead_state_dict"]
                optimizer.state = lookahead_state["state"]
                optimizer.param_groups = lookahead_state["param_groups"]
            else:
                raise KeyError("Missing optimizer state keys in the saved state.")
        else:
            optimizer.load_state_dict(state)

    def update_E(self, decay: float = 0.999) -> None:
        netG = self.get_bare_model(self.netG)
        netE = self.get_bare_model(self.netE)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(netE.named_parameters())
        for k in netG_params.keys():
            if k in netE_params:
                netE_params[k].data.mul_(decay).add_(
                    netG_params[k].data, alpha=1 - decay
                )
            else:
                raise KeyError(f"Parameter {k} not found in netE")

    def merge_bnorm_train(self) -> None:
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    def merge_bnorm_test(self) -> None:
        merge_bn(self.netG)
        tidy_sequential(self.netG)

    def _set_static_graph(self, network: torch.nn.Module) -> None:
        network._set_static_graph()

    def log(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def create_optimizer(
        optim_type, params, lr=0.001, betas=(0.9, 0.999), wd=0.01, momentum=0.0
    ):
        if optim_type == "adam":
            base_optimizer = Adam(params, lr=lr, betas=betas, weight_decay=wd)
        elif optim_type == "radam":
            base_optimizer = RAdam(params, lr=lr, betas=betas, weight_decay=wd)
        elif optim_type == "sgd":
            base_optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        elif optim_type == "adamw":
            base_optimizer = AdamW(params, lr=lr, betas=betas, weight_decay=wd)
        elif optim_type == "adamax":
            base_optimizer = Adamax(params, lr=lr, betas=betas, weight_decay=wd)
        elif optim_type == "rmsprop":
            base_optimizer = RMSprop(
                params,
                lr=lr,
                alpha=0.99,
                eps=1e-8,
                weight_decay=wd,
                momentum=momentum,
            )
        else:
            raise NotImplementedError(f"Optimizer '{optim_type}' is not implemented.")

        return Lookahead(base_optimizer)

    @staticmethod
    def create_scheduler(scheduler_type, optimizer, opt_train, prefix):
        key = f"{prefix}_scheduler_"
        if scheduler_type == "MultiStepLR":
            return lr_scheduler.MultiStepLR(
                optimizer,
                milestones=opt_train.get(f"{key}milestones", [30, 80]),
                gamma=opt_train.get(f"{key}gamma", 0.1),
            )
        elif scheduler_type == "CyclicLR":
            return lr_scheduler.CyclicLR(
                optimizer,
                base_lr=opt_train.get(f"{key}lr", 0.001),
                max_lr=opt_train.get(f"{key}max_lr", 0.1),
                step_size_up=opt_train.get(f"{key}step_size_up", 2000),
                step_size_down=opt_train.get(f"{key}step_size_down", 2000),
                mode=opt_train.get(f"{key}mode", "triangular"),
                gamma=1.0,
                cycle_momentum=opt_train.get(f"{key}cycle_momentum", True),
                base_momentum=0.8,
                max_momentum=0.9,
                last_epoch=-1,
            )
        elif scheduler_type == "CosineAnnealingLR":
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=opt_train.get(f"{key}T_max", 50),
                eta_min=opt_train.get(f"{key}eta_min", 0),
            )
        elif scheduler_type == "ReduceLROnPlateau":
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=opt_train.get(f"{key}lr_patience", 10),
                factor=opt_train.get(f"{key}lr_factor", 0.1),
                min_lr=opt_train.get(f"{key}lr_min", 0),
            )
        elif scheduler_type == "OneCycleLR":
            return lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opt_train.get(f"{key}max_lr", 0.1),
                total_steps=opt_train.get(f"{key}total_steps", 10000),
                pct_start=opt_train.get(f"{key}pct_start", 0.3),
                anneal_strategy=opt_train.get(f"{key}anneal_strategy", "cos"),
                cycle_momentum=opt_train.get(f"{key}cycle_momentum", True),
                base_momentum=0.8,
                max_momentum=0.9,
                last_epoch=-1,
            )
        else:
            raise NotImplementedError(
                f"Scheduler '{scheduler_type}' is not implemented."
            )
