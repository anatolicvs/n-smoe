import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn.modules.module import Module
from torch.nn.parallel import DataParallel, DistributedDataParallel

from utils_n.utils_bnorm import merge_bn, tidy_sequential


class ModelBase(ABC):
    def __init__(self, opt: dict) -> None:
        self.opt = opt
        self.save_dir = opt["path"]["models"]

        if opt["dist"]:
            local_rank = int(os.environ.get("LOCAL_RANK", opt["rank"]))
            self.device = torch.device(f"cuda:{local_rank}")
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
        if self.opt["dist"]:
            local_rank = int(os.environ.get("LOCAL_RANK", self.opt["rank"]))
            network = DistributedDataParallel(
                network, device_ids=[local_rank], output_device=local_rank,
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
        param_key: str = "params",
    ) -> None:
        network = self.get_bare_model(network)
        state_dict = torch.load(load_path, map_location=self.device, weights_only=True)
        if param_key in state_dict:
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
        torch.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, load_path: str, optimizer: torch.optim.Optimizer) -> None:
        optimizer.load_state_dict(
            torch.load(
                load_path,
                map_location=self.device,
                weights_only=True,
            )
        )

    def update_E(self, decay: float = 0.999) -> None:
        netG: Module = self.get_bare_model(self.netG)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1 - decay)

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
