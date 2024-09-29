import torch
import piq
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)
from typing import Any
from utils_n import utils_image as util


def calc_metrics(data, models, metrics, device) -> dict[Any, Any]:
    results = {}
    for method, model in models.items():
        with torch.no_grad():
            if method in ["N-SMoE", "N-SMoE-II", "N-SMoE-III"]:
                E_img: torch.Tensor = model(data["L_p"].to(device), data["L"].size())
            elif method == "Bicubic":
                E_img: torch.Tensor = model(data["L"], data["H"].size()[2:])
                E_img = E_img.to(device)
            elif method == "DPSR":
                E_img: torch.Tensor = model(data["L"].to(device))
            elif method == "ESRGAN":
                E_img: torch.Tensor = model(data["L"].to(device))

        gt: torch.Tensor = data["H"].to(device)

        gt_uint = util.tensor2uint(gt)
        E_img_uint = util.tensor2uint(E_img)

        data_min: float = min(gt_uint.min().item(), E_img_uint.min().item())
        data_max: float = max(gt_uint.max().item(), E_img_uint.max().item())
        data_range: float = data_max - data_min

        metric_results = {"e_img": E_img_uint}
        if "psnr" in metrics:
            metric_results["psnr"] = psnr(gt_uint, E_img_uint, data_range=data_range)
        if "ssim" in metrics:
            metric_results["ssim"] = ssim(
                gt_uint, E_img_uint, data_range=data_range, multichannel=False
            )

        if "lpips" in metrics:
            metric_results["lpips"] = piq.LPIPS()(E_img, gt).item()
        if "dists" in metrics:
            metric_results["dists"] = piq.DISTS()(E_img, gt).item()
        if "brisque" in metrics:
            metric_results["brisque"] = piq.brisque(
                E_img.clamp(0, 1).to(torch.float), data_range=1.0, reduction="none"
            )

        results[method] = metric_results

    return results
