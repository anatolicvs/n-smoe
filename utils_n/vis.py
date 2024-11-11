from typing import Dict, Optional, Any
import torch
import matplotlib
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

matplotlib.use("TkAgg")

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from numpy.fft import fft2, fftshift
from skimage.metrics import adapted_rand_error
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import variation_of_information
import seaborn as sns


def run_model_with_memory_optimization(model, input_data):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        output = model(input_data)

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    return output


plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        # --- Font Settings ---
        "font.family": "serif",  # Set default font family to serif
        "font.serif": [
            "Times New Roman",
            "Times",
            "Palatino",
            "serif",
        ],  # Primary serif fonts
        "font.sans-serif": [
            "Arial",
            "Liberation Sans",
            "Bitstream Vera Sans",
            "sans-serif",
        ],  # Primary sans-serif fonts
        # --- Font Sizes ---
        "font.size": 12,  # Base font size
        "axes.labelsize": 12,  # Font size for axis labels
        "axes.titlesize": 12,  # Font size for titles
        "xtick.labelsize": 12,  # Font size for x-axis ticks
        "ytick.labelsize": 12,  # Font size for y-axis ticks
        "legend.fontsize": 11,  # Font size for legends
        # --- Line and Marker Settings ---
        "lines.linewidth": 1.5,  # Line width for clarity
        "lines.markersize": 5,  # Marker size for data points
        # --- Grid Settings ---
        # "grid.color": "gray",  # Light gray grid lines
        # "grid.alpha": 0.5,  # 50% transparency for grid lines
        # "axes.grid": True,  # Enable grid by default
        # --- Figure Settings ---
        # "figure.dpi": 300,  # High resolution for on-screen display
        "savefig.dpi": 600,  # High resolution for saved figures
        # "figure.figsize": (3.5, 2.5),  # Single-column width in inches
        # --- Font Embedding ---
        "pdf.fonttype": 42,  # Embed fonts as TrueType in PDF
        "ps.fonttype": 42,  # Embed fonts as TrueType in PS
        # --- Miscellaneous ---
        "legend.frameon": False,  # Remove legend frame for a cleaner look
        "text.usetex": False,
        "pgf.texsystem": "pdflatex",
        # "style.use": "seaborn-v0_8-paper",
    }
)


def format_metric(
    title: str, sorted_metrics: list, value: float, metric_name: str
) -> str:
    if sorted_metrics and title == sorted_metrics[0][0]:
        if metric_name == "PSNR":
            return rf"\textbf{{{metric_name}: {value:.2f} dB}}"
        else:
            return rf"\textbf{{{metric_name}: {value:.4f}}}"
    elif len(sorted_metrics) > 1 and title == sorted_metrics[1][0]:
        if metric_name == "PSNR":
            return rf"\underline{{{metric_name}: {value:.2f} dB}}"
        else:
            return rf"\underline{{{metric_name}: {value:.4f}}}"
    else:
        if metric_name == "PSNR":
            return f"{metric_name}: {value:.2f} dB"
        else:
            return f"{metric_name}: {value:.4f}"


def visualize_with_segmentation(
    images: Dict[str, Dict[str, Any]],
    mask_generator: SAM2AutomaticMaskGenerator,
    hr_key: str = "H_img",
    ref_key: str = "H_crop_img",
    lrcrop_key: str = "L_crop_img",
    cmap: str = "gray",
    save_path: str = None,
    visualize: bool = False,
    error_map: bool = False,
):
    plt.rcParams.update({"text.usetex": True})

    def calculate_metrics(gt_mask, pred_mask):
        gt_seg = gt_mask[0]["segmentation"]
        pred_seg = pred_mask[0]["segmentation"]
        vi_split, vi_merge = variation_of_information(gt_seg, pred_seg)
        vi_score = vi_split + vi_merge
        are_score, _, _ = adapted_rand_error(gt_seg, pred_seg)
        return vi_score, are_score

    def show_anns(anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        img = np.ones(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
                4,
            )
        )
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann["segmentation"]
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
        ax.imshow(img)

    ordered_keys = list(images.keys())
    ncols = len(ordered_keys) + 1
    nrows = 3
    width_ratios = [2, 2] + [1] * (ncols - 2)
    fig_width = 21 / 13.3 * sum(width_ratios)
    fig_height = 3.5

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=nrows,
        ncols=ncols,
        height_ratios=[2, 2, 0.5],
        width_ratios=width_ratios,
        hspace=0.01,
        wspace=0.01,
        figure=fig,
    )

    annotated_mask = mask_generator.generate(
        np.repeat(images[hr_key]["image"][:, :, np.newaxis], 3, axis=-1)
    )

    # annotated_mask = run_model_with_memory_optimization(
    #     mask_generator.generate,
    #     np.repeat(images[hr_key]["image"][:, :, np.newaxis], 3, axis=-1),
    # )

    ax_annotated = fig.add_subplot(gs[0:2, 0])
    ax_annotated.imshow(images[hr_key]["image"], cmap=cmap)
    show_anns(annotated_mask)
    ax_annotated.axis("off")
    ax_annotated.set_title("Annotated Segmentation", fontweight="bold")

    ax_img_hr = fig.add_subplot(gs[0:2, 1])
    ax_img_hr.imshow(images[hr_key]["image"], cmap=cmap)

    gray_hr = (
        cv2.cvtColor(images[hr_key]["image"], cv2.COLOR_RGB2GRAY)
        if images[hr_key]["image"].ndim == 3
        else images[hr_key]["image"]
    )
    gray_crop = (
        cv2.cvtColor(images[ref_key]["image"], cv2.COLOR_RGB2GRAY)
        if images[ref_key]["image"].ndim == 3
        else images[ref_key]["image"]
    )
    res = cv2.matchTemplate(gray_hr, gray_crop, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    x, y = max_loc
    crop_height, crop_width = images[ref_key]["image"].shape[:2]
    rect = patches.Rectangle(
        (x, y), crop_width, crop_height, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax_img_hr.add_patch(rect)
    ax_img_hr.axis("off")
    title = images[hr_key]["title"]
    ax_img_hr.set_title(title, fontweight="bold")

    gt_mask = None
    vi_scores = {}
    are_scores = {}

    for idx, key in enumerate(ordered_keys[1:], start=2):

        recon_img = images[key]["image"]
        recon_title = images[key]["title"]

        ax_img = fig.add_subplot(gs[0, idx])
        ax_img.imshow(recon_img, cmap=cmap, vmax=recon_img.max(), vmin=recon_img.min())
        ax_img.axis("off")
        ax_img.set_title(recon_title, fontweight="bold")

        mask = mask_generator.generate(
            np.repeat(images[key]["image"][:, :, np.newaxis], 3, axis=-1)
        )
        ax_seg = fig.add_subplot(gs[1, idx])
        ax_seg.imshow(recon_img, cmap=cmap, vmax=recon_img.max(), vmin=recon_img.min())
        show_anns(mask)
        ax_seg.axis("off")

        if key == ref_key:
            gt_mask = mask

        if key not in [hr_key, ref_key, lrcrop_key]:
            vi_score, are_score = calculate_metrics(gt_mask, mask)
            vi_scores[key] = vi_score
            are_scores[key] = are_score

    sorted_vi = sorted(vi_scores.items(), key=lambda x: x[1])
    sorted_are = sorted(are_scores.items(), key=lambda x: x[1])

    for idx, key in enumerate(ordered_keys[1:], start=2):
        ax_title = fig.add_subplot(gs[2, idx])
        if key not in [hr_key, ref_key, lrcrop_key]:
            vi_score = vi_scores[key]
            are_score = are_scores[key]

            vi_text = f"VoI: {vi_score:.4f}"
            are_text = f"ARE: {are_score:.4f}"

            if key == sorted_vi[0][0]:
                vi_text = r"\textbf{VoI: %.4f}" % vi_score
            elif key == sorted_vi[1][0]:
                vi_text = r"\underline{VoI: %.4f}" % vi_score

            if key == sorted_are[0][0]:
                are_text = r"\textbf{ARE: %.4f}" % are_score
            elif key == sorted_are[1][0]:
                are_text = r"\underline{ARE: %.4f}" % are_score

            # display_title = f"{images[key]['title']}\n{vi_text}\n{are_text}"
            display_title = f"{vi_text}\n{are_text}"
            ax_title.text(
                0.5,
                0.0,
                display_title,
                va="center",
                ha="center",
                transform=ax_title.transAxes,
            )
        else:
            pass
            # display_title = images[key]["title"]
            # ax_title.text(
            #     0.5,
            #     0.2,
            #     display_title,
            #     weight="bold",
            #     va="center",
            #     ha="center",
            #     transform=ax_title.transAxes,
            # )
        ax_title.axis("off")

    plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)
    plt.subplots_adjust(
        left=0.12, bottom=0.12, right=0.89, top=0.89, wspace=0, hspace=0
    )
    if visualize:
        plt.show()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=600)
        plt.close(fig)


def visualize_with_error_map(
    images: Dict[str, Dict[str, Any]],
    hr_key: str = "H_img",
    ref_key: str = "H_crop_img",
    lrcrop_key: str = "L_crop_img",
    cmap: str = "gray",
    save_path: str = None,
    visualize: bool = True,
) -> None:

    plt.rcParams.update({"text.usetex": True})

    def create_error_cmap() -> LinearSegmentedColormap:
        colors = ["navy", "blue", "cyan", "limegreen", "yellow", "red"]
        return LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)

    def calculate_error_map(
        gt_image: np.ndarray, reconstructed_image: np.ndarray
    ) -> np.ndarray:
        return (gt_image.astype(float) - reconstructed_image.astype(float)) ** 2

    hr_image = images[hr_key]["image"]
    hr_title = images[hr_key]["title"]

    lrcrop_image = images[lrcrop_key]["image"]
    lrcrop_title = images[lrcrop_key]["title"]

    ref_img = images[ref_key]["image"]
    ref_title = images[ref_key]["title"]

    recon_items = {
        k: v for k, v in images.items() if k not in [hr_key, ref_key, lrcrop_key]
    }
    num_recon = len(recon_items)
    if num_recon < 1:
        raise ValueError("At least one reconstructed image is required.")

    ncols = 3 + num_recon
    nrows = 3
    width_ratios = [2, 1, 1] + [1] * num_recon
    height_ratios = [2, 2, 0.5]

    unit_width = 21 / 13
    fig_width = unit_width * sum(width_ratios)
    fig_height = 3.6

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=nrows,
        ncols=ncols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.01,
        wspace=0.01,
        figure=fig,
    )

    error_cmap = create_error_cmap()

    ax_hr = fig.add_subplot(gs[0:2, 0])
    ax_hr.imshow(hr_image, cmap=cmap)
    ax_hr.axis("off")
    ax_hr.set_title(hr_title, fontweight="bold")

    ax_lrcrop = fig.add_subplot(gs[0, 1])
    ax_lrcrop.imshow(lrcrop_image, cmap=cmap)
    ax_lrcrop.axis("off")
    ax_lrcrop.set_title(lrcrop_title, fontweight="bold")

    ax_ref = fig.add_subplot(gs[0, 2])
    ax_ref.imshow(ref_img, cmap=cmap)
    ax_ref.axis("off")
    ax_ref.set_title(ref_title, fontweight="bold")

    gray_hr = (
        cv2.cvtColor(hr_image, cv2.COLOR_RGB2GRAY)
        if len(hr_image.shape) == 3 and hr_image.shape[2] == 3
        else hr_image.copy()
    )
    gray_ref = (
        cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
        if len(ref_img.shape) == 3 and ref_img.shape[2] == 3
        else ref_img.copy()
    )

    res = cv2.matchTemplate(gray_hr, gray_ref, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    x, y = max_loc
    crop_height, crop_width = gray_ref.shape[:2]

    rect = patches.Rectangle(
        (x, y), crop_width, crop_height, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax_hr.add_patch(rect)

    psnr_values = {}
    ssim_values = {}
    # mse_values = {}

    for idx, (key, item) in enumerate(recon_items.items(), start=3):
        recon_img = item["image"]
        recon_title = item["title"]

        ax_recon = fig.add_subplot(gs[0, idx])
        ax_recon.imshow(
            recon_img, cmap=cmap, vmax=recon_img.max(), vmin=recon_img.min()
        )
        ax_recon.axis("off")
        ax_recon.set_title(recon_title, fontweight="bold")

        error_map = calculate_error_map(ref_img, recon_img)
        error_map_normalized = (error_map - error_map.min()) / (
            error_map.max() - error_map.min()
            if error_map.max() - error_map.min() != 0
            else 1
        )

        ax_error = fig.add_subplot(gs[1, idx])
        # ax_error.imshow(error_map_normalized, cmap=error_cmap, vmin=0, vmax=1)
        im_error = ax_error.imshow(
            error_map_normalized, cmap=error_cmap, vmin=0, vmax=1
        )
        ax_error.axis("off")

        try:
            if ref_img.shape != recon_img.shape:
                raise ValueError(
                    f"Shape mismatch between reference image '{ref_key}' {ref_img.shape} "
                    f"and reconstructed image '{key}' {recon_img.shape}."
                )

            data_min: float = min(ref_img.min().item(), recon_img.min().item())
            data_max: float = max(ref_img.max().item(), recon_img.max().item())
            data_range: float = data_max - data_min

            current_psnr = psnr(ref_img, recon_img, data_range=data_range)
            current_ssim = ssim(
                ref_img,
                recon_img,
                data_range=data_range,
                multichannel=True,
            )
            # current_mse = mse(ref_img, recon_img)

            psnr_values[recon_title] = current_psnr
            ssim_values[recon_title] = current_ssim
            # mse_values[recon_title] = current_mse
        except Exception as e:
            print(f"Error calculating PSNR/SSIM/MSE for '{recon_title}': {str(e)}")

    bbox = ax_error.get_position()
    cbar_width = 0.005
    cbar_ax = fig.add_axes([bbox.x1 + 0.004, bbox.y0, cbar_width, bbox.height])

    cbar = fig.colorbar(im_error, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)

    sorted_psnr = sorted(psnr_values.items(), key=lambda x: x[1], reverse=True)
    sorted_psnr = sorted(psnr_values.items(), key=lambda x: x[1], reverse=True)
    sorted_ssim = sorted(ssim_values.items(), key=lambda x: x[1], reverse=True)
    # sorted_mse = sorted(mse_values.items(), key=lambda x: x[1])

    for idx, (key, item) in enumerate(recon_items.items(), start=3):
        recon_title = item["title"]
        ax_title = fig.add_subplot(gs[2, idx])
        ax_title.axis("off")

        # if (
        #     recon_title in psnr_values
        #     and recon_title in ssim_values
        #     and recon_title in mse_values
        # ): # removed upon for Sikora's request.

        if recon_title in psnr_values and recon_title in ssim_values:
            psnr_val = psnr_values[recon_title]
            ssim_val = ssim_values[recon_title]
            # mse_val = mse_values[recon_title]

            psnr_display = format_metric(recon_title, sorted_psnr, psnr_val, "PSNR")
            ssim_display = format_metric(recon_title, sorted_ssim, ssim_val, "SSIM")
            # mse_display = format_metric(recon_title, sorted_mse, mse_val, "MSE")

            # display_title = f"${psnr_display}$\n${ssim_display}$\n${mse_display}$" # removed upon for Sikora's request.
            display_title = f"${psnr_display}$\n${ssim_display}$"

            ax_title.text(
                0.5,
                0.0,
                display_title,
                ha="center",
                va="center",
                transform=ax_title.transAxes,
            )
        else:
            ax_title.text(
                0.4,
                0.4,
                recon_title,
                ha="center",
                va="center",
                transform=ax_title.transAxes,
            )

    plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)
    # plt.subplots_adjust(
    #     left=0.12, bottom=0.12, right=0.88, top=0.88, wspace=0, hspace=0
    # )
    plt.subplots_adjust(right=0.9)
    if visualize:
        plt.show()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=600)
        plt.close(fig)


def visualize_data(
    images: Dict[str, Dict[str, Any]],
    ref_key: str = "H_crop_img",
    lrcrop_key: str = "L_crop_img",
    hr_key: str = "H_img",
    cmap: str = "gray",
    save_path: str = None,
    visualize: bool = True,
) -> None:

    plt.rcParams.update({"text.usetex": True})

    ordered_keys = [k for k in images.keys() if k != hr_key]

    num_images = len(ordered_keys)

    width_ratios = [1] * num_images
    total_width_ratio = sum(width_ratios)

    unit_width = 21 / 8
    fig_width = unit_width * total_width_ratio
    fig_height = 8

    ncols = num_images
    nrows = 5

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)

    gs = GridSpec(
        nrows=nrows, ncols=ncols, figure=fig, height_ratios=[3, 0.5, 0.5, 1, 2]
    )

    axes_colors = ["darkslategray", "olive", "steelblue", "darkred", "slategray"]

    ref_img = images[ref_key]["image"].squeeze()
    psnr_values = {}
    ssim_values = {}

    ax_img_list = []

    for i, key in enumerate(ordered_keys):
        img = images[key]["image"]
        title = images[key]["title"]

        ax_img = fig.add_subplot(gs[0, i])
        ax_img_list.append(ax_img)

        if img is not None and img.size > 0:
            ax_img.imshow(img, cmap=cmap, vmax=img.max(), vmin=img.min(), aspect="auto")
        else:
            print(f"Warning: Invalid image data for {title}")
        ax_img.axis("on")
        for spine in ax_img.spines.values():
            spine.set_color(axes_colors[i % len(axes_colors)])

        ax_img.set_title(title, fontweight="bold")

        if key not in [ref_key, lrcrop_key]:

            data_min: float = min(ref_img.min().item(), img.min().item())
            data_max: float = max(ref_img.max().item(), img.max().item())
            data_range: float = data_max - data_min

            current_psnr = psnr(ref_img, img, data_range=data_range)
            current_ssim = ssim(
                ref_img,
                img,
                data_range=data_range,
                multichannel=True,
            )
            psnr_values[title] = current_psnr
            ssim_values[title] = current_ssim

        freq = fftshift(fft2(img))
        freq_magnitude = np.log(np.abs(freq) + 1)

        ax_x_spectrum = fig.add_subplot(gs[1, i])
        ax_x_spectrum.plot(np.sum(freq_magnitude, axis=0), color="blue")
        ax_x_spectrum.set_title("X-Spectrum", fontweight="bold")
        ax_x_spectrum.set_xticks([])
        ax_x_spectrum.set_yticks([])

        ax_x_spectrum.set_xlabel(r"$\mathrm{Frequency\ (pixels)}$")
        ax_y_spectrum = fig.add_subplot(gs[2, i])
        ax_y_spectrum.plot(np.sum(freq_magnitude, axis=1), color="blue")
        ax_y_spectrum.set_title(
            "Y-Spectrum",
            fontweight="bold",
        )
        ax_y_spectrum.set_xlabel(r"$\mathrm{Frequency\ (pixels)}$")
        ax_y_spectrum.set_xticks([])
        ax_y_spectrum.set_yticks([])
        # ax_y_spectrum.axis("off")

        ax_2d_spectrum = fig.add_subplot(gs[3, i])
        ax_2d_spectrum.imshow(freq_magnitude, cmap="gray")
        ax_2d_spectrum.set_title(
            "2D Spectrum",
            fontweight="bold",
        )
        ax_2d_spectrum.axis("on")

    sorted_psnr = sorted(psnr_values.items(), key=lambda x: x[1], reverse=True)
    sorted_ssim = sorted(ssim_values.items(), key=lambda x: x[1], reverse=True)

    for i, key in enumerate(ordered_keys):
        title = images[key]["title"]
        if key not in [ref_key, lrcrop_key]:
            current_psnr = psnr_values[title]
            current_ssim = ssim_values[title]

            psnr_display = format_metric(title, sorted_psnr, current_psnr, "PSNR")
            ssim_display = format_metric(title, sorted_ssim, current_ssim, "SSIM")

            ax_img_list[i].set_title(
                title,
                verticalalignment="baseline",
                horizontalalignment="center",
                y=1.0,
                pad=10,
            )

            display_title = f"${psnr_display}$\n${ssim_display}$"

            ax_img_list[i].text(
                0.5,
                -0.15,
                display_title,
                ha="center",
                va="top",
                transform=ax_img_list[i].transAxes,
                usetex=True,
            )

    # plt.tight_layout()
    if visualize:
        plt.show()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=600)
        plt.close(fig)


def visualize_sharpening_results_(
    img_L: np.ndarray,
    img_H: np.ndarray,
    si_H: float,
    si_L: float,
    sharpened_images: dict,
    metrics: dict,
    save_path: str = None,
    visualize: bool = True,
):
    num_factors = len(next(iter(sharpened_images.values())))
    num_models = len(sharpened_images)

    total_rows = 1 + num_models
    total_cols = 1 + num_factors + 4

    golden_ratio = 1.8
    fig_width = 3.0 * total_cols
    fig_height = fig_width / golden_ratio
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = GridSpec(
        total_rows + 1,
        total_cols,
        figure=fig,
        height_ratios=[1] + [1] * num_models + [0.4],
        width_ratios=[0.3] + [1] * num_factors + [1.2] * 4,
    )
    gs.update(wspace=0.3, hspace=0.5)

    ax_L = fig.add_subplot(gs[0, 1])
    ax_L.imshow(img_L, cmap="gray")
    ax_L.set_title("Low Resolution (LR)", fontweight="bold")
    ax_L.set_xlabel(f"SI: {si_L:.4f}")
    ax_L.axis("on")

    ax_H = fig.add_subplot(gs[0, 2])
    ax_H.imshow(img_H, cmap="gray")
    ax_H.set_title("High Resolution (HR)", fontweight="bold")
    ax_H.set_xlabel(f"SI: {si_H:.4f}")
    ax_H.axis("on")

    metrics_start_col = 3

    ax_psnr = fig.add_subplot(gs[0, metrics_start_col])
    for model_name in metrics.keys():
        psnr_values = [
            metrics[model_name][f]["PSNR"] for f in sorted(metrics[model_name].keys())
        ]
        ax_psnr.plot(
            sorted(metrics[model_name].keys()), psnr_values, label=f"{model_name}"
        )
    ax_psnr.set_title("PSNR (dB)", fontweight="bold")
    ax_psnr.set_xlabel("Sharpening Factor (SF)")
    ax_psnr.grid(True)
    ax_psnr.legend(loc="best")

    ax_ssim = fig.add_subplot(gs[0, metrics_start_col + 1])
    for model_name in metrics.keys():
        ssim_values = [
            metrics[model_name][f]["SSIM"] for f in sorted(metrics[model_name].keys())
        ]
        ax_ssim.plot(sorted(metrics[model_name].keys()), ssim_values)
    ax_ssim.set_title("SSIM", fontweight="bold")
    ax_ssim.set_xlabel("SF")
    ax_ssim.grid(True)

    ax_si = fig.add_subplot(gs[0, metrics_start_col + 2])
    for model_name in metrics.keys():
        si_values = [
            metrics[model_name][f]["SI"] for f in sorted(metrics[model_name].keys())
        ]
        ax_si.plot(sorted(metrics[model_name].keys()), si_values)
    ax_si.set_title("Sharpness Index (SI)", fontweight="bold")
    ax_si.set_xlabel("SF")
    ax_si.grid(True)

    ax_lpips = fig.add_subplot(gs[0, metrics_start_col + 3])
    for model_name in metrics.keys():
        lpips_values = [
            metrics[model_name][f]["LPIPS"] for f in sorted(metrics[model_name].keys())
        ]
        ax_lpips.plot(sorted(metrics[model_name].keys()), lpips_values)
    ax_lpips.set_title("LPIPS", fontweight="bold")
    ax_lpips.set_xlabel("SF")
    ax_lpips.grid(True)

    factors = sorted(next(iter(metrics.values())).keys())

    for model_index, (model_name, model_images) in enumerate(sharpened_images.items()):
        row = 1 + model_index
        ax_model_name = fig.add_subplot(gs[row, 0])
        ax_model_name.text(
            1.5,
            0.5,
            model_name,
            ha="right",
            va="center",
            rotation=90,
            fontweight="bold",
        )
        ax_model_name.axis("off")

        for i, factor in enumerate(factors):
            col = 1 + i
            ax_model = fig.add_subplot(gs[row, col])
            ax_model.imshow(model_images[factor], cmap="gray")

            ax_model.set_xlabel(
                f"PSNR: {metrics[model_name][factor]['PSNR']:.2f} dB | "
                f"SSIM: {metrics[model_name][factor]['SSIM']:.2f}\n"
                f"SI: {metrics[model_name][factor]['SI']:.2f} | "
                f"LPIPS: {metrics[model_name][factor]['LPIPS']:.4f}",
                fontsize=8,
                labelpad=10,
            )

            ax_model.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )

    for i, factor in enumerate(factors):
        col = 1 + i
        ax_alpha = fig.add_subplot(gs[-1, col])
        ax_alpha.text(
            0.5,
            0.5,
            f"$\\alpha = {factor:.2f}$",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax_alpha.transAxes,
        )
        ax_alpha.axis("off")

    plt.tight_layout(pad=3.0)

    if save_path:
        plt.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
            dpi=600,
            transparent=True,
        )
    if visualize:
        plt.show()
    plt.close()


def visualize_sharpening_results(
    img_L: np.ndarray,
    img_H: np.ndarray,
    si_H: float,
    si_L: float,
    sharpened_images: dict,
    metrics: dict,
    save_path: str = None,
    visualize: bool = True,
):
    num_factors = len(next(iter(sharpened_images.values())))
    num_models = len(sharpened_images)

    total_rows = 1 + num_models
    total_cols = 1 + num_factors + 4

    golden_ratio = 1.8
    fig_width = 3.0 * total_cols
    fig_height = fig_width / golden_ratio
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = GridSpec(
        total_rows + 1,
        total_cols,
        figure=fig,
        height_ratios=[1] + [1] * num_models + [0.4],
        width_ratios=[0.3] + [1] * num_factors + [1.2] * 4,
    )
    gs.update(wspace=0.3, hspace=0.5)

    ax_L = fig.add_subplot(gs[0, 1])
    ax_L.imshow(img_L, cmap="gray")
    ax_L.set_title("Low Resolution (LR)", fontweight="bold")
    ax_L.set_xlabel(f"SI: {si_L:.4f}")
    ax_L.axis("on")

    ax_H = fig.add_subplot(gs[0, 2])
    ax_H.imshow(img_H, cmap="gray")
    ax_H.set_title("High Resolution (HR)", fontweight="bold")
    ax_H.set_xlabel(f"SI: {si_H:.4f}")
    ax_H.axis("on")

    metrics_start_col = 3

    ax_psnr = fig.add_subplot(gs[0, metrics_start_col])
    for model_name in metrics.keys():
        psnr_values = [
            metrics[model_name][f]["PSNR"] for f in sorted(metrics[model_name].keys())
        ]
        ax_psnr.plot(
            sorted(metrics[model_name].keys()), psnr_values, label=f"{model_name}"
        )
    ax_psnr.set_title(r"PSNR (dB) $\uparrow$", fontweight="bold")
    ax_psnr.set_xlabel("Sharpening Factor (SF)")
    ax_psnr.grid(True)
    ax_psnr.legend(loc="best")

    ax_ssim = fig.add_subplot(gs[0, metrics_start_col + 1])
    for model_name in metrics.keys():
        ssim_values = [
            metrics[model_name][f]["SSIM"] for f in sorted(metrics[model_name].keys())
        ]
        ax_ssim.plot(sorted(metrics[model_name].keys()), ssim_values)
    ax_ssim.set_title(r"SSIM $\uparrow$", fontweight="bold")
    ax_ssim.set_xlabel("SF")
    ax_ssim.grid(True)

    ax_si = fig.add_subplot(gs[0, metrics_start_col + 2])
    for model_name in metrics.keys():
        si_values = [
            metrics[model_name][f]["SI"] for f in sorted(metrics[model_name].keys())
        ]
        ax_si.plot(sorted(metrics[model_name].keys()), si_values)
    ax_si.set_title(r"Sharpness Index (SI) $\uparrow$", fontweight="bold")
    ax_si.set_xlabel("SF")
    ax_si.grid(True)

    ax_lpips = fig.add_subplot(gs[0, metrics_start_col + 3])
    for model_name in metrics.keys():
        lpips_values = [
            metrics[model_name][f]["LPIPS"] for f in sorted(metrics[model_name].keys())
        ]
        ax_lpips.plot(sorted(metrics[model_name].keys()), lpips_values)
    ax_lpips.set_title(r"LPIPS $\downarrow$", fontweight="bold")
    ax_lpips.set_xlabel("SF")
    ax_lpips.grid(True)

    factors = sorted(next(iter(metrics.values())).keys())

    for model_index, (model_name, model_images) in enumerate(sharpened_images.items()):
        row = 1 + model_index
        ax_model_name = fig.add_subplot(gs[row, 0])
        ax_model_name.text(
            1.5,
            0.5,
            model_name,
            ha="right",
            va="center",
            rotation=90,
            fontweight="bold",
        )
        ax_model_name.axis("off")

        for i, factor in enumerate(factors):
            col = 1 + i
            ax_model = fig.add_subplot(gs[row, col])
            ax_model.imshow(model_images[factor], cmap="gray")

            ax_model.set_xlabel(
                # f"PSNR: {metrics[model_name][factor]['PSNR']:.2f} dB | "
                # f"SSIM: {metrics[model_name][factor]['SSIM']:.2f}\n"
                f"SI: {metrics[model_name][factor]['SI']:.2f} | "
                f"LPIPS: {metrics[model_name][factor]['LPIPS']:.4f}",
                labelpad=10,
            )

            ax_model.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )

    for i, factor in enumerate(factors):
        col = 1 + i
        ax_alpha = fig.add_subplot(gs[-1, col])
        ax_alpha.text(
            0.5,
            0.5,
            f"$\\alpha = {factor:.2f}$",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax_alpha.transAxes,
        )
        ax_alpha.axis("off")

    plt.tight_layout(pad=3.0)

    if save_path:
        plt.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
            dpi=600,
            transparent=True,
        )
    if visualize:
        plt.show()
    plt.close()


def visualize_data_with_spectra(self):
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

        L_signal_stochastic_x = np.sum(np.log(np.abs(L_freq_stochastic) + 1), axis=0)
        L_signal_stochastic_y = np.sum(np.log(np.abs(L_freq_stochastic) + 1), axis=1)
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


def visualize_data_pair(images, titles):
    num_images = len(images)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = GridSpec(5, num_images + 1, figure=fig, height_ratios=[3, 0.5, 0.5, 1, 2])

    axes_colors = ["darkslategray", "olive", "steelblue", "darkred", "slategray"]
    reference_title = "Ground Truth Crop"
    reference_index = titles.index(reference_title) if reference_title in titles else -1
    ref_img = images[reference_index].squeeze() if reference_index != -1 else None

    for i, (img, title) in enumerate(zip(images, titles)):
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(img, cmap="gray")
        ax_img.axis("on")
        for spine in ax_img.spines.values():  # Apply color to each spine
            spine.set_color(axes_colors[i % len(axes_colors)])

        if title in ["N-SMoE", "DPSR"] and ref_img is not None:
            current_psnr = psnr(ref_img, img, data_range=img.max() - img.min())
            current_ssim = ssim(ref_img, img, data_range=img.max() - img.min())
            title += f"\nPSNR: {current_psnr:.2f} dB, SSIM: {current_ssim:.4f}"

        ax_img.set_title(
            title, fontsize=12, family="Times New Roman", fontweight="bold"
        )

        freq = fftshift(fft2(img))
        freq_magnitude = np.log(np.abs(freq) + 1)

        ax_x_spectrum = fig.add_subplot(gs[1, i])
        ax_x_spectrum.plot(np.sum(freq_magnitude, axis=0), color="blue")
        ax_x_spectrum.set_title(
            "X-Spectrum", fontsize=12, family="Times New Roman", fontweight="bold"
        )
        ax_x_spectrum.set_xlabel(
            "Frequency (pixels)", fontsize=11, family="Times New Roman"
        )
        ax_x_spectrum.set_yticklabels([])
        ax_x_spectrum.tick_params(axis="both", which="major", labelsize=10)
        ax_x_spectrum.grid(True)

        ax_y_spectrum = fig.add_subplot(gs[2, i])
        ax_y_spectrum.plot(np.sum(freq_magnitude, axis=1), color="blue")
        ax_y_spectrum.set_title(
            "Y-Spectrum", fontsize=12, family="Times New Roman", fontweight="bold"
        )
        ax_y_spectrum.set_xlabel(
            "Frequency (pixels)", fontsize=11, family="Times New Roman"
        )
        ax_y_spectrum.set_yticklabels([])
        ax_y_spectrum.tick_params(axis="both", which="major", labelsize=10)
        ax_y_spectrum.grid(True)

        ax_2d_spectrum = fig.add_subplot(gs[3, i])
        ax_2d_spectrum.imshow(freq_magnitude, cmap="gray")
        ax_2d_spectrum.set_title(
            "2D Spectrum", fontsize=12, family="Times New Roman", fontweight="bold"
        )
        ax_2d_spectrum.axis("on")

    nsmoe_idx = titles.index("N-SMoE") if "N-SMoE" in titles else -1
    dpsr_idx = titles.index("DPSR") if "DPSR" in titles else -1

    if nsmoe_idx != -1 and dpsr_idx != -1:
        rec_image = images[nsmoe_idx].squeeze()
        dpsr_image = images[dpsr_idx].squeeze()
        error_map = np.abs(rec_image - dpsr_image)

        ax_error_map = fig.add_subplot(gs[0, dpsr_idx + 1])
        ax_error_map.imshow(error_map, cmap="viridis")
        ax_error_map.set_title(
            "Error Map (N-SMoE - DPSR)",
            fontsize=12,
            family="Times New Roman",
            fontweight="bold",
        )
        ax_error_map.axis("off")

    plt.show()
