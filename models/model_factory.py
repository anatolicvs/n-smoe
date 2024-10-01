from dataclasses import dataclass
from typing import Any, Type, TypeVar, Generic
import torch
from torch import nn

T = TypeVar("T", bound=nn.Module)


@dataclass
class ModelConfig(Generic[T]):
    encoder_config: Any
    moe_cfg_class: Type
    ae_cfg_class: Type
    ae_class: Type[T]
    model_params: dict
    opt: dict


class ModelFactory:
    @staticmethod
    def create_model(
        config: ModelConfig,
        sharpening_factor: float,
        weights_path: str,
        device: torch.device,
    ) -> T:
        decoder_cfg = config.moe_cfg_class(
            kernel=config.model_params["kernel"], sharpening_factor=sharpening_factor
        )
        auto_cfg = config.ae_cfg_class(
            EncoderConfig=config.encoder_config,
            DecoderConfig=decoder_cfg,
            d_in=config.model_params["n_channels"],
            d_out=config.model_params["z"],
            phw=config.opt["phw"],
            overlap=config.opt["overlap"],
        )
        model = config.ae_class(cfg=auto_cfg)
        model.load_state_dict(
            torch.load(weights_path, map_location=device), strict=True
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model.to(device)


def load_model(
    config: ModelConfig,
    sharpening_factor: float,
    weights_path: str,
    device: torch.device,
) -> T:
    return ModelFactory.create_model(config, sharpening_factor, weights_path, device)
