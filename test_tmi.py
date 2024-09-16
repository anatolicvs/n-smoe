# type: ignore
import csv
import datetime
import json
import logging

import os.path
import random
from typing import Any, Dict, List

import click
import numpy as np
import piq
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from torch.utils.data import DataLoader

from data.select_dataset import define_Dataset
from dnnlib import EasyDict
from utils_n import utils_image as util
from utils_n import utils_logger
from utils_n import utils_option as option
from utils_n.utils_dist import get_dist_info, init_dist

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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


def visualize_with_segmentation(
    images: List[np.ndarray],
    titles: List[str],
    mask_generator: SAM2AutomaticMaskGenerator,
    cmap: str = "gray",
    save_path: str = None,
    visualize: bool = False,
    error_map: bool = False,
):
    import matplotlib

    matplotlib.use("TkAgg")

    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from skimage.metrics import adapted_rand_error, variation_of_information

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams["text.usetex"] = True

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
            if borders:
                contours, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                contours = [
                    cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                    for contour in contours
                ]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
        ax.imshow(img)

    fig = plt.figure(figsize=(18, 4))
    gs = GridSpec(
        3,
        len(images) + 1,
        height_ratios=[2, 2, 0.5],  # Adjusting the height ratios for better space usage
        width_ratios=[2, 2, 1, 1, 1, 1, 1, 1],
        hspace=0.01,  # Reducing space between rows
        wspace=0.01,  # Reducing space between columns
    )

    annotated_mask = mask_generator.generate(
        np.repeat(images[0][:, :, np.newaxis], 3, axis=-1)
    )
    ax_annotated = fig.add_subplot(gs[0:2, 0])
    ax_annotated.imshow(images[0], cmap=cmap)
    show_anns(annotated_mask)
    ax_annotated.axis("off")
    ax_annotated.set_title("Annotated Segmentation", fontsize=12, weight="bold")

    ax_img_hr = fig.add_subplot(gs[0:2, 1])
    ax_img_hr.imshow(images[0], cmap=cmap)
    ax_img_hr.axis("off")
    ax_img_hr.set_title(titles[0], fontsize=12, weight="bold")

    ground_truth_index = 2
    gt_mask = None
    vi_scores = {}
    are_scores = {}

    for i in range(1, len(images)):
        ax_img = fig.add_subplot(gs[0, i + 1])
        ax_img.imshow(images[i], cmap=cmap)
        ax_img.axis("off")

        mask = mask_generator.generate(np.repeat(images[i][:, :, None], 3, axis=-1))
        ax_seg = fig.add_subplot(gs[1, i + 1])
        ax_seg.imshow(images[i], cmap=cmap)
        show_anns(mask)
        ax_seg.axis("off")
        ax_title = fig.add_subplot(gs[2, i + 1])

        if i == ground_truth_index:
            gt_mask = mask

        if i > ground_truth_index:
            vi_score, are_score = calculate_metrics(gt_mask, mask)
            vi_scores[i] = vi_score
            are_scores[i] = are_score

    sorted_vi = sorted(vi_scores.items(), key=lambda x: x[1])
    sorted_are = sorted(are_scores.items(), key=lambda x: x[1])

    for i in range(1, len(images)):
        ax_title = fig.add_subplot(gs[2, i + 1])
        if i > ground_truth_index:
            vi_score = vi_scores[i]
            are_score = are_scores[i]

            vi_text = f"VoI: {vi_score:.4f}"
            are_text = f"ARE: {are_score:.4f}"

            if i == sorted_vi[0][0]:
                vi_text = r"\textbf{VoI: %.4f}" % vi_score
            elif i == sorted_vi[1][0]:
                vi_text = r"\underline{VoI: %.4f}" % vi_score

            if i == sorted_are[0][0]:
                are_text = r"\textbf{ARE: %.4f}" % are_score
            elif i == sorted_are[1][0]:
                are_text = r"\underline{ARE: %.4f}" % are_score

            display_title = f"{titles[i]}\n{vi_text}\n{are_text}"

            ax_title.text(
                0.5,
                0.0,
                display_title,
                va="center",
                ha="center",
                transform=ax_title.transAxes,
            )
        else:
            display_title = titles[i]
            ax_title.text(
                0.5,
                0.5,
                display_title,
                weight="bold",
                va="center",
                ha="center",
                transform=ax_title.transAxes,
            )

    plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)
    plt.subplots_adjust(wspace=0.02, hspace=0)

    for ax in fig.get_axes():
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0)
    if visualize:
        plt.show()


def visualize_with_error_map(
    images: List[np.ndarray],
    titles: List[str],
    cmap: str = "gray",
    save_path: str = None,
    visualize: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("TkAgg")

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec
    from skimage.metrics import mean_squared_error as mse
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    def create_error_cmap():
        colors = ["navy", "blue", "cyan", "limegreen", "yellow", "red"]
        return LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)

    def calculate_error_map(gt_image, reconstructed_image):
        return (gt_image.astype(float) - reconstructed_image.astype(float)) ** 2

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11
    plt.rcParams["text.usetex"] = True

    fig = plt.figure(figsize=(14, 4))
    gs = GridSpec(
        3,
        len(images),
        height_ratios=[2, 2, 0.5],
        width_ratios=[2] + [1] * (len(images) - 1),
        hspace=0.01,
        wspace=0.01,
    )

    reference_title = "Ground Truth Crop"
    reference_index = titles.index(reference_title)
    reference_image = images[reference_index].squeeze()

    psnr_values = {}
    ssim_values = {}
    mse_values = {}

    error_cmap = create_error_cmap()

    ax_img_hr = fig.add_subplot(gs[0:2, 0])
    ax_img_hr.imshow(images[0], cmap=cmap)
    ax_img_hr.axis("off")
    ax_img_hr.set_title(titles[0], fontsize=12, fontweight="bold")

    max_error = 0

    for i in range(1, len(images)):
        img = images[i]
        title = titles[i]

        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(img, cmap=cmap)
        ax_img.axis("off")

        if i > 2:
            error_map = calculate_error_map(reference_image, img)
            error_map_normalized = (error_map - error_map.min()) / (
                error_map.max() - error_map.min()
            )
            max_error = max(max_error, error_map.max())

            ax_error_map = fig.add_subplot(gs[1, i])
            im = ax_error_map.imshow(
                error_map_normalized, cmap=error_cmap, vmin=0, vmax=1
            )
            ax_error_map.axis("off")

            try:
                current_psnr = psnr(reference_image, img)
                current_ssim = ssim(
                    reference_image,
                    img,
                    data_range=reference_image.max() - reference_image.min(),
                )
                current_mse = mse(reference_image, img)
                psnr_values[title] = current_psnr
                ssim_values[title] = current_ssim
                mse_values[title] = current_mse
            except Exception as e:
                print(f"Error calculating PSNR/SSIM/MSE for {title}: {str(e)}")

    sorted_psnr = sorted(psnr_values.items(), key=lambda x: x[1], reverse=True)
    sorted_ssim = sorted(ssim_values.items(), key=lambda x: x[1], reverse=True)
    sorted_mse = sorted(mse_values.items(), key=lambda x: x[1])

    for i in range(1, len(images)):
        title = titles[i]
        ax_title = fig.add_subplot(gs[2, i])
        ax_title.axis("off")

        if i > 2 and title in psnr_values and title in ssim_values:
            psnr_text = f"PSNR: {psnr_values[title]:.2f}"
            ssim_text = f"SSIM: {ssim_values[title]:.4f}"
            mse_text = f"MSE: {mse_values[title]:.4f}"

            if title == sorted_psnr[0][0]:
                psnr_text = r"\textbf{" + psnr_text + "}"
            elif title == sorted_psnr[1][0]:
                psnr_text = r"\underline{" + psnr_text + "}"

            if title == sorted_ssim[0][0]:
                ssim_text = r"\textbf{" + ssim_text + "}"
            elif title == sorted_ssim[1][0]:
                ssim_text = r"\underline{" + ssim_text + "}"

            display_title = f"{title}\n${psnr_text}$ dB\n${ssim_text}$\n${mse_text}$"
            ax_title.text(
                0.5,
                0.0,
                display_title,
                ha="center",
                va="center",
                transform=ax_title.transAxes,
                fontsize=10,
            )
        else:
            display_title = title

            ax_title.text(
                0.5,
                4.5,
                display_title,
                ha="center",
                va="center",
                transform=ax_title.transAxes,
                fontsize=10,
            )

    plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)
    plt.subplots_adjust(wspace=0.02, hspace=0)

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    if visualize:
        plt.show()


def visualize_data_pair(images, titles):
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from numpy.fft import fft2, fftshift
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    num_images = len(images)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = GridSpec(5, num_images + 1, figure=fig, height_ratios=[3, 0.5, 0.5, 1, 2])

    axes_colors = ["darkslategray", "olive", "steelblue", "darkred", "slategray"]
    reference_title = "Ground Truth Crop"
    reference_index = titles.index(reference_title) if reference_title in titles else -1
    reference_image = (
        images[reference_index].squeeze() if reference_index != -1 else None
    )

    for i, (img, title) in enumerate(zip(images, titles)):
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(img, cmap="gray")
        ax_img.axis("on")
        for spine in ax_img.spines.values():  # Apply color to each spine
            spine.set_color(axes_colors[i % len(axes_colors)])

        if title in ["N-SMoE", "DPSR"] and reference_image is not None:
            current_psnr = psnr(reference_image, img, data_range=img.max() - img.min())
            current_ssim = ssim(reference_image, img, data_range=img.max() - img.min())
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


def visualize_data(
    images: List[np.ndarray],
    titles: List[str],
    cmap: str = "gray",
    save_path: str = None,
    visualize: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("TkAgg")

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from numpy.fft import fft2, fftshift
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams["text.usetex"] = True

    num_images = len(images)
    fig = plt.figure(figsize=(17, 9), constrained_layout=True)
    gs = GridSpec(5, num_images, figure=fig, height_ratios=[3, 0.5, 0.5, 1, 2])

    axes_colors = ["darkslategray", "olive", "steelblue", "darkred", "slategray"]
    reference_title = "Ground Truth Crop"
    low_res_title = "Noisy LR Crop"
    reference_index = titles.index(reference_title)
    reference_image = images[reference_index].squeeze()

    psnr_values = {}
    ssim_values = {}

    freq_table = {}

    for i, (img, title) in enumerate(zip(images, titles)):
        ax_img = fig.add_subplot(gs[0, i])
        if img is not None and img.size > 0:
            ax_img.imshow(img, cmap=cmap, aspect="auto")
        else:
            print(f"Warning: Invalid image data for {title}")
        ax_img.axis("on")
        for spine in ax_img.spines.values():
            spine.set_color(axes_colors[i % len(axes_colors)])

        if title != reference_title and title != low_res_title:
            current_psnr = psnr(reference_image, img, data_range=img.max() - img.min())
            current_ssim = ssim(
                reference_image, img, channel_axis=-1, data_range=img.max() - img.min()
            )
            psnr_values[title] = current_psnr
            ssim_values[title] = current_ssim

    sorted_psnr = sorted(psnr_values.items(), key=lambda x: x[1], reverse=True)
    sorted_ssim = sorted(ssim_values.items(), key=lambda x: x[1], reverse=True)

    for i, (img, title) in enumerate(zip(images, titles)):
        ax_img = fig.add_subplot(gs[0, i])
        if img is not None and img.size > 0:
            ax_img.imshow(img, cmap=cmap, aspect="auto")
        else:
            print(f"Warning: Invalid image data for {title}")
        ax_img.axis("on")
        for spine in ax_img.spines.values():
            spine.set_color(axes_colors[i % len(axes_colors)])

        if title != reference_title and title != low_res_title:
            current_psnr = psnr_values[title]
            current_ssim = ssim_values[title]

            psnr_text = f"PSNR: {current_psnr:.2f}"
            ssim_text = f"SSIM: {current_ssim:.4f}"

            if title == sorted_psnr[0][0]:
                psnr_text = r"\textbf{" + psnr_text + "}"
            elif title == sorted_psnr[1][0]:
                psnr_text = r"\underline{" + psnr_text + "}"

            if title == sorted_ssim[0][0]:
                ssim_text = r"\textbf{" + ssim_text + "}"
            elif title == sorted_ssim[1][0]:
                ssim_text = r"\underline{" + ssim_text + "}"

            title += f"\n${psnr_text}$ dB\n${ssim_text}$"

        ax_img.set_title(
            title, fontsize=12, family="Times New Roman", fontweight="bold"
        )

        freq = fftshift(fft2(img))
        freq_magnitude = np.log(np.abs(freq) + 1)

        # signal_x = np.sum(freq_magnitude, axis=0)
        # signal_y = np.sum(freq_magnitude, axis=1)

        # freq_table[title]["energy_x"] = np.sum(signal_x**2)
        # freq_table[title]["energy_y"] = np.sum(signal_y**2)
        # freq_table[title]["entropy_x"] = entropy(signal_x, base=2)
        # freq_table[title]["entropy_y"] = entropy(signal_y, base=2)

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
        ax_2d_spectrum.imshow(freq_magnitude, cmap=cmap)
        ax_2d_spectrum.set_title(
            "2D Spectrum", fontsize=12, family="Times New Roman", fontweight="bold"
        )
        ax_2d_spectrum.axis("on")

    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0)
    if visualize:
        plt.show()


def visualize_sharpening_results(
    img_L: np.ndarray,
    img_H: np.ndarray,
    sharpened_images: Dict[float, np.ndarray],
    metrics: Dict[float, Dict[str, float]],
    save_path: str = None,
    visualize: bool = True,
):

    import matplotlib

    matplotlib.use("TkAgg")

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams["text.usetex"] = True  # Enable LaTeX rendering

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1])
    gs.update(wspace=0.2, hspace=0.1)

    spine_color = "#004594"  # Deep blue color

    ax_L = fig.add_subplot(gs[0, 0])
    ax_L.imshow(img_L, cmap="gray")
    ax_L.set_title("Low Resolution", fontweight="bold")
    ax_L.axis("on")
    for spine in ax_L.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(1)
    ax_L.tick_params(axis="both", colors=spine_color)

    ax_H = fig.add_subplot(gs[0, 1])
    ax_H.imshow(img_H, cmap="gray")
    ax_H.set_title("High Resolution", fontweight="bold")
    ax_H.axis("on")
    for spine in ax_H.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(1)
    ax_H.tick_params(axis="both", colors=spine_color)

    factors = list(metrics.keys())
    psnr_values = [metrics[f]["PSNR"] for f in factors]
    ssim_values = [metrics[f]["SSIM"] for f in factors]
    si_values = [metrics[f]["SI"] for f in factors]

    ax_psnr = fig.add_subplot(gs[0, 2])
    ax_psnr.plot(factors, psnr_values, "bo-")
    ax_psnr.set_title("PSNR (dB)", fontweight="bold")
    ax_psnr.set_xlabel("Sharpening Factor (SF)")
    ax_psnr.grid(True)
    ax_psnr.tick_params(axis="both", which="major", labelsize=8)

    ax_ssim = fig.add_subplot(gs[0, 3])
    ax_ssim.plot(factors, ssim_values, "ro-")
    ax_ssim.set_title("SSIM", fontweight="bold")
    ax_ssim.set_xlabel("SF")
    ax_ssim.grid(True)
    ax_ssim.tick_params(axis="both", which="major", labelsize=8)

    ax_si = fig.add_subplot(gs[0, 4])
    ax_si.plot(factors, si_values, "go-")
    ax_si.set_title("Sharpness Index", fontweight="bold")
    ax_si.set_xlabel("SF")
    ax_si.grid(True)
    ax_si.tick_params(axis="both", which="major", labelsize=8)

    for ax in [ax_L, ax_H, ax_psnr, ax_ssim, ax_si]:
        ax.set_aspect("auto")

    for i, (factor, img) in enumerate(list(sharpened_images.items())[:5]):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.text(
            0.5,
            -0.05,
            r"$\alpha= %.2f$" % factor,
            ha="center",
            va="top",
            usetex=True,
            transform=ax.transAxes,
            fontsize=12,
        )

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

    if save_path:
        plt.savefig(
            save_path, format="pdf", bbox_inches="tight", dpi=300, transparent=True
        )
    if visualize:
        plt.show()
    plt.close()


def default_resizer(inputs, target_size):
    return F.interpolate(
        inputs, size=target_size, mode="bilinear", align_corners=False, antialias=True
    )


# def gen_latex_table(average_metric_data):
#     def best_vals(values, best_values):
#         formatted = []
#         for v in values:
#             if v == best_values["best"]:
#                 formatted.append("\\textbf{" + f"{v:.4f}" + "}")
#             elif v == best_values["second_best"]:
#                 formatted.append("\\underline{" + f"{v:.4f}" + "}")
#             else:
#                 formatted.append(f"{v:.4f}")
#         return formatted

#     metrics = ["psnr", "ssim", "lpips", "dists"]
#     methods = list(average_metric_data["psnr"].keys())
#     datasets = list(average_metric_data["psnr"][methods[0]].keys())
#     scales = list(average_metric_data["psnr"][methods[0]][datasets[0]].keys())

#     latex_str = r"\begin{table*}[t]" + "\n"
#     latex_str += r"\centering" + "\n"
#     latex_str += (
#         r"\caption{Quantitative comparison of reconstruction quality on different datasets for "
#         + ", ".join(f"{scale}" for scale in scales)
#         + r" scales. The best and second-best results are highlighted in \textbf{bold} and \underline{underline}.}"
#         + "\n"
#     )
#     latex_str += r"\resizebox{\textwidth}{!}{" + "\n"
#     latex_str += (
#         r"\begin{tabular}{ l c " + " ".join(["c c c c" for _ in datasets]) + " }" + "\n"
#     )
#     latex_str += r"\toprule" + "\n"
#     latex_str += (
#         r"\textbf{Methods} & \textbf{Scale} & "
#         + " & ".join(
#             [
#                 f"\\multicolumn{{4}}{{c}}{{\\textbf{{{dataset}}}}}"
#                 for dataset in datasets
#             ]
#         )
#         + r" \\"
#         + "\n"
#     )
#     latex_str += r"\cmidrule(lr){3-" + str(3 + 4 * len(datasets) - 1) + "}" + "\n"
#     latex_str += (
#         "& & "
#         + " & ".join(
#             [
#                 r"\multicolumn{2}{c}{\textbf{Fidelity}} & \multicolumn{2}{c}{\textbf{Perceptual}}"
#                 for _ in datasets
#             ]
#         )
#         + r" \\"
#         + "\n"
#     )
#     latex_str += r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} " * len(datasets) + "\n"
#     latex_str += (
#         "& & "
#         + " & ".join(
#             [
#                 r"\textbf{PSNR↑} & \textbf{SSIM↑} & \textbf{LPIPS↓} & \textbf{DISTS↓}"
#                 for _ in datasets
#             ]
#         )
#         + r" \\"
#         + "\n"
#     )
#     latex_str += r"\midrule" + "\n"

#     all_metric_values = {
#         metric: {dataset: {scale: [] for scale in scales} for dataset in datasets}
#         for metric in metrics
#     }
#     for method in methods:
#         for dataset in datasets:
#             for scale in scales:
#                 for metric in metrics:
#                     value = average_metric_data[metric][method][dataset][scale]
#                     all_metric_values[metric][dataset][scale].append(value)

#     best_values_dict = {
#         metric: {dataset: {scale: {} for scale in scales} for dataset in datasets}
#         for metric in metrics
#     }
#     for metric in metrics:
#         for dataset in datasets:
#             for scale in scales:
#                 values = all_metric_values[metric][dataset][scale]
#                 sorted_values = sorted(
#                     values, reverse=(metric not in ["lpips", "dists"])
#                 )
#                 best_values_dict[metric][dataset][scale]["best"] = sorted_values[0]
#                 best_values_dict[metric][dataset][scale]["second_best"] = (
#                     sorted_values[1] if len(sorted_values) > 1 else sorted_values[0]
#                 )

#     for method in methods:
#         for scale in scales:
#             latex_str += method + " & " + f"{scale}" + " & "
#             for dataset in datasets:
#                 for metric in metrics:
#                     value = average_metric_data[metric][method][dataset][scale]
#                     best_values = best_values_dict[metric][dataset][scale]
#                     is_best = value == best_values["best"]
#                     is_second_best = value == best_values["second_best"]
#                     if is_best:
#                         formatted_value = "\\textbf{" + f"{value:.4f}" + "}"
#                     elif is_second_best:
#                         formatted_value = "\\underline{" + f"{value:.4f}" + "}"
#                     else:
#                         formatted_value = f"{value:.4f}"
#                     latex_str += formatted_value + " & "
#             latex_str = latex_str.rstrip(" & ")
#             latex_str += r" \\" + "\n"
#         latex_str += r"\midrule" + "\n"

#     latex_str += r"\bottomrule" + "\n"
#     latex_str += r"\end{tabular}" + "\n"
#     latex_str += r"}" + "\n"
#     latex_str += r"\end{table*}" + "\n"

#     return latex_str


def gen_latex_table(average_metric_data):
    def best_vals(values, best_values):
        formatted = []
        for v in values:
            if v == best_values["best"]:
                formatted.append("\\textbf{" + f"{v:.4f}" + "}")
            elif v == best_values["second_best"]:
                formatted.append("\\underline{" + f"{v:.4f}" + "}")
            else:
                formatted.append(f"{v:.4f}")
        return formatted

    metrics = ["psnr", "ssim", "lpips", "dists"]
    methods = list(average_metric_data["psnr"].keys())
    datasets = list(average_metric_data["psnr"][methods[0]].keys())
    scales = list(average_metric_data["psnr"][methods[0]][datasets[0]].keys())

    latex_str = r"\begin{table*}[t]" + "\n"
    latex_str += r"\centering" + "\n"
    latex_str += (
        r"\caption{Quantitative comparison of reconstruction quality on different datasets for "
        + ", ".join(f"{scale}" for scale in scales)
        + r" scales. The best and second-best results are highlighted in \textbf{bold} and \underline{underline}.}"
        + "\n"
    )
    latex_str += r"\resizebox{\textwidth}{!}{" + "\n"
    latex_str += (
        r"\begin{tabular}{ l c " + " ".join(["c c c c" for _ in datasets]) + " }" + "\n"
    )
    latex_str += r"\toprule" + "\n"
    latex_str += (
        r"\textbf{Methods} & \textbf{Scale} & "
        + " & ".join(
            [
                f"\\multicolumn{{4}}{{c}}{{\\textbf{{{dataset}}}}}"
                for dataset in datasets
            ]
        )
        + r" \\"
        + "\n"
    )
    latex_str += r"\cmidrule(lr){3-" + str(3 + 4 * len(datasets) - 1) + "}" + "\n"
    latex_str += (
        "& & "
        + " & ".join(
            [
                "\\multicolumn{2}{c}{\\textbf{Fidelity}} & \\multicolumn{2}{c}{\\textbf{Perceptual}}"
                for _ in datasets
            ]
        )
        + r" \\"
        + "\n"
    )
    latex_str += r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} " * len(datasets) + "\n"
    latex_str += (
        "& & "
        + " & ".join(
            [
                "\\textbf{PSNR$\\uparrow$} & \\textbf{SSIM$\\uparrow$} & \\textbf{LPIPS$\\downarrow$} & \\textbf{DISTS$\\downarrow$}"
                for _ in datasets
            ]
        )
        + r" \\"
        + "\n"
    )
    latex_str += r"\midrule" + "\n"

    all_metric_values = {
        metric: {dataset: {scale: [] for scale in scales} for dataset in datasets}
        for metric in metrics
    }

    for method in methods:
        for dataset in datasets:
            for scale in scales:
                for metric in metrics:
                    value = average_metric_data[metric][method][dataset][scale]
                    all_metric_values[metric][dataset][scale].append(value)

    best_values_dict = {
        metric: {dataset: {scale: {} for scale in scales} for dataset in datasets}
        for metric in metrics
    }
    for metric in metrics:
        for dataset in datasets:
            for scale in scales:
                values = all_metric_values[metric][dataset][scale]
                sorted_values = sorted(
                    values, reverse=(metric not in ["lpips", "dists"])
                )
                best_values_dict[metric][dataset][scale]["best"] = sorted_values[0]
                best_values_dict[metric][dataset][scale]["second_best"] = (
                    sorted_values[1] if len(sorted_values) > 1 else sorted_values[0]
                )

    for method in methods:
        for scale in scales:
            latex_str += method + " & " + f"{scale}" + " & "
            for dataset in datasets:
                for metric in metrics:
                    value = average_metric_data[metric][method][dataset][scale]
                    best_values = best_values_dict[metric][dataset][scale]
                    is_best = value == best_values["best"]
                    is_second_best = value == best_values["second_best"]
                    if is_best:
                        formatted_value = "\\textbf{" + f"{value:.4f}" + "}"
                    elif is_second_best:
                        formatted_value = "\\underline{" + f"{value:.4f}" + "}"
                    else:
                        formatted_value = f"{value:.4f}"
                    latex_str += formatted_value + " & "
            latex_str = latex_str.rstrip(" & ")
            latex_str += r" \\" + "\n"
        latex_str += r"\midrule" + "\n"

    latex_str += r"\bottomrule" + "\n"
    latex_str += r"\end{tabular}" + "\n"
    latex_str += r"}" + "\n"
    latex_str += r"\end{table*}" + "\n"
    latex_str += r"\label{tab:quantitative_results_1}" + "\n"

    return latex_str


def process_data(data, models, metrics, device) -> dict[Any, Any]:
    results = {}
    for method, model in models.items():
        with torch.no_grad():
            if method == "N-SMoE" or method == "N-SMoE-II" or method == "N-SMoE-III":
                E_img = model(data["L_p"].to(device), data["L"].size())
            elif method == "Bicubic":
                E_img = model(data["L"], data["H"].size()[2:])
                E_img = E_img.to(device)
            else:
                E_img = model(data["L"].to(device))

            gt = data["H"].clamp(0, 1).to(torch.float).to(device)

            E_img = E_img.clamp(0, 1).to(torch.float)

            metric_results = {"e_img": E_img}
            if "psnr" in metrics:
                metric_results["psnr"] = piq.psnr(E_img, gt, data_range=1).float()
            if "ssim" in metrics:
                metric_results["ssim"] = piq.ssim(
                    E_img, gt, data_range=1, reduction="mean"
                )
            if "lpips" in metrics:
                metric_results["lpips"] = piq.LPIPS()(E_img, gt).item()
            if "dists" in metrics:
                metric_results["dists"] = piq.DISTS()(E_img, gt).item()
            if "brisque" in metrics:
                metric_results["brisque"] = piq.brisque(
                    E_img, data_range=1.0, reduction="none"
                )

            results[method] = metric_results

    return results


@click.command()
@click.option(
    "--opt",
    type=str,
    default="options/testing/test_tmi_local.json",
    help="Path to option JSON file.",
)
@click.option("--launcher", default="pytorch", help="job launcher")
@click.option("--local_rank", type=int, default=0)
@click.option("--dist", is_flag=True, default=False)
@click.option("--visualize", is_flag=True, default=True)
@click.option("--backend", default="TkAgg")
def main(**kwargs):

    args = EasyDict(kwargs)
    opt = option.parse(args.opt, is_train=True)
    opt["dist"] = args.dist
    opt["visualize"] = args.visualize
    opt["backend"] = args.backend

    if opt["dist"]:
        init_dist("pytorch")
    opt["rank"], opt["world_size"] = get_dist_info()

    border = opt["scale"]

    opt = option.dict_to_nonedict(opt)

    if opt["rank"] == 0:
        util.mkdirs(
            (path for key, path in opt["path"].items() if "pretrained" not in key)
        )

    if isinstance(opt, dict) and opt.get("rank") == 0:
        logger_name = "train"
        utils_logger.logger_info(
            logger_name, os.path.join(opt["path"]["log"], logger_name + ".log")
        )
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    seed = random.randint(1, 10000)
    print("Random seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = opt.get("task", "sr_x2")

    if task == "sr_x2":
        from models.network_dpsr import MSRResNet_prior as dpsr
        from models.network_rrdb import RRDB as rrdb

        from models.network_unetmoex1 import Autoencoder as ae1
        from models.network_unetmoex1 import AutoencoderConfig as ae1_cfg
        from models.network_unetmoex1 import EncoderConfig as enc1_cfg
        from models.network_unetmoex1 import MoEConfig as moe1_cfg

        from models.network_unetmoex3 import Autoencoder as ae2
        from models.network_unetmoex3 import AutoencoderConfig as ae2_cfg
        from models.network_unetmoex3 import EncoderConfig as enc2_cfg
        from models.network_unetmoex3 import MoEConfig as moe2_cfg

        json_moex3 = """
        {
        "netG": {
            "net_type": "unet_moex3",
            "kernel": 16,
            "sharpening_factor": 1,
            "model_channels": 72,
            "num_res_blocks": 8,
            "attention_resolutions": [16,8,4],
            "dropout": 0.1,
            "num_groups": 36,
            "num_heads": 36,
            "use_new_attention_order": true,
            "use_checkpoint": true,
            "use_fp16": false,
            "resblock_updown": true,
            "channel_mult": [2,4,8],
            "conv_resample": true,
            "resample_2d": false,
            "attention_type": "cross_attention",
            "activation": "GELU",
            "rope_theta": 960000.0,
            "resizer_num_layers": 8,
            "resizer_avg_pool": false,
            "init_type": "default",
            "scale": 2,
            "n_channels": 1
            }
        }
        """

        json_moex3_32 = """
        {
        "netG": {
                "net_type": "unet_moex3",
                "kernel": 32,
                "sharpening_factor": 1,
                "model_channels": 72,
                "num_res_blocks": 8,
                "attention_resolutions": [16,8,4],
                "dropout": 0.1,
                "num_groups": 36,
                "num_heads": 36,
                "use_new_attention_order": true,
                "use_checkpoint": true,
                "use_fp16": false,
                "resblock_updown": true,
                "channel_mult": [2,4,8],
                "conv_resample": true,
                "resample_2d": false,
                "attention_type": "cross_attention",
                "activation": "GELU",
                "rope_theta": 960000.0,
                "resizer_num_layers": 8,
                "resizer_avg_pool": false,
                "init_type": "default",
                "scale": 2,
                "n_channels": 1
            }
        }
        """

        json_moex1 = """
        {
            "netG": {
                "net_type": "unet_moex1",
                "kernel": 16,
                "sharpening_factor": 1.3,
                "model_channels": 64,
                "num_res_blocks": 8,
                "attention_resolutions": [16,8,4],
                "dropout": 0.2,
                "num_groups": 8,
                "num_heads": 32,
                "num_head_channels": 32,
                "use_new_attention_order": true,
                "use_checkpoint": true,
                "resblock_updown": false,
                "channel_mult": [1,2,4,8],
                "resample_2d": false,
                "pool": "attention",
                "activation": "GELU",
                "resizer_num_layers": 2,
                "resizer_avg_pool": false,
                "scale": 2,
                "n_channels": 1
            }
        }
        """

        netG_moex1 = json.loads(json_moex1)["netG"]

        z = 2 * netG_moex1["kernel"] + 4 * netG_moex1["kernel"] + netG_moex1["kernel"]

        encoder_cfg1 = enc1_cfg(
            model_channels=netG_moex1["model_channels"],
            num_res_blocks=netG_moex1["num_res_blocks"],
            attention_resolutions=netG_moex1["attention_resolutions"],
            dropout=netG_moex1["dropout"],
            num_groups=netG_moex1["num_groups"],
            scale_factor=netG_moex1["scale"],
            num_heads=netG_moex1["num_heads"],
            num_head_channels=netG_moex1["num_head_channels"],
            use_new_attention_order=netG_moex1["use_new_attention_order"],
            use_checkpoint=netG_moex1["use_checkpoint"],
            resblock_updown=netG_moex1["resblock_updown"],
            channel_mult=netG_moex1["channel_mult"],
            resample_2d=netG_moex1["resample_2d"],
            pool=netG_moex1["pool"],
            activation=netG_moex1["activation"],
        )

        decoder_cfg1 = moe1_cfg(
            kernel=netG_moex1["kernel"],
            sharpening_factor=netG_moex1["sharpening_factor"],
        )

        autoenocer_cfg1 = ae1_cfg(
            EncoderConfig=encoder_cfg1,
            DecoderConfig=decoder_cfg1,
            d_in=netG_moex1["n_channels"],
            d_out=z,
            phw=opt["phw"],
            overlap=opt["overlap"],
        )

        model_moex1 = ae1(cfg=autoenocer_cfg1)

        model_moex1.load_state_dict(
            torch.load(opt["pretrained_models"]["moex1_x2"], weights_only=True),
            strict=True,
        )
        model_moex1.eval()
        for k, v in model_moex1.named_parameters():
            v.requires_grad = False
        model_moex1 = model_moex1.to(device)

        netG_moex3 = json.loads(json_moex3)["netG"]

        z = 2 * netG_moex3["kernel"] + 4 * netG_moex3["kernel"] + netG_moex3["kernel"]

        encoder_cfg3 = enc2_cfg(
            model_channels=netG_moex3["model_channels"],  # 32,
            num_res_blocks=netG_moex3["num_res_blocks"],  # 4,
            attention_resolutions=netG_moex3["attention_resolutions"],  # [16, 8],
            dropout=netG_moex3["dropout"],  # 0.2,
            channel_mult=netG_moex3["channel_mult"],  # (2, 4, 8),
            conv_resample=netG_moex3["conv_resample"],  # False,
            dims=2,
            use_checkpoint=netG_moex3["use_checkpoint"],  # True,
            use_fp16=netG_moex3["use_fp16"],  # False,
            num_heads=netG_moex3["num_heads"],  # 4,
            # num_head_channels=netG_moex3["num_head_channels"],  # 8,
            resblock_updown=netG_moex3["resblock_updown"],  # False,
            num_groups=netG_moex3["num_groups"],  # 32,
            resample_2d=netG_moex3["resample_2d"],  # True,
            scale_factor=netG_moex3["scale"],
            resizer_num_layers=netG_moex3["resizer_num_layers"],  # 4,
            resizer_avg_pool=netG_moex3["resizer_avg_pool"],  # False,
            activation=netG_moex3["activation"],
            rope_theta=netG_moex3["rope_theta"],  # 10000.0,
            attention_type=netG_moex3[
                "attention_type"
            ],  # "cross_attention",  # "attention" or "cross_attention"
        )

        decoder_cfg3 = moe2_cfg(
            kernel=netG_moex3["kernel"],
            sharpening_factor=netG_moex3["sharpening_factor"],
        )

        autoenocer_cfg3 = ae2_cfg(
            EncoderConfig=encoder_cfg3,
            DecoderConfig=decoder_cfg3,
            d_in=netG_moex3["n_channels"],
            d_out=z,
            phw=opt["phw"],
            overlap=opt["overlap"],
        )

        model_moex3 = ae2(cfg=autoenocer_cfg3)

        model_moex3.load_state_dict(
            torch.load(opt["pretrained_models"]["moex3_x2"], weights_only=True),
            strict=True,
        )

        model_moex3.eval()
        for k, v in model_moex3.named_parameters():
            v.requires_grad = False
        model_moex3 = model_moex3.to(device)

        netG_moex3_32 = json.loads(json_moex3_32)["netG"]

        z_32 = (
            2 * netG_moex3_32["kernel"]
            + 4 * netG_moex3_32["kernel"]
            + netG_moex3_32["kernel"]
        )

        encoder_cfg3_32 = enc2_cfg(
            model_channels=netG_moex3_32["model_channels"],  # 32,
            num_res_blocks=netG_moex3_32["num_res_blocks"],  # 4,
            attention_resolutions=netG_moex3_32["attention_resolutions"],  # [16, 8],
            dropout=netG_moex3_32["dropout"],  # 0.2,
            channel_mult=netG_moex3_32["channel_mult"],  # (2, 4, 8),
            conv_resample=netG_moex3_32["conv_resample"],  # False,
            dims=2,
            use_checkpoint=netG_moex3_32["use_checkpoint"],  # True,
            use_fp16=netG_moex3_32["use_fp16"],  # False,
            num_heads=netG_moex3_32["num_heads"],  # 4,
            # num_head_channels=netG_moex3_32["num_head_channels"],  # 8,
            resblock_updown=netG_moex3_32["resblock_updown"],  # False,
            num_groups=netG_moex3_32["num_groups"],  # 32,
            resample_2d=netG_moex3_32["resample_2d"],  # True,
            scale_factor=netG_moex3_32["scale"],
            resizer_num_layers=netG_moex3_32["resizer_num_layers"],  # 4,
            resizer_avg_pool=netG_moex3_32["resizer_avg_pool"],  # False,
            activation=netG_moex3_32["activation"],
            rope_theta=netG_moex3_32["rope_theta"],  # 10000.0,
            attention_type=netG_moex3_32[
                "attention_type"
            ],  # "cross_attention",  # "attention" or "cross_attention"
        )

        decoder_cfg3_32 = moe2_cfg(
            kernel=netG_moex3_32["kernel"],
            sharpening_factor=netG_moex3_32["sharpening_factor"],
        )

        autoenocer_cfg3_32 = ae2_cfg(
            EncoderConfig=encoder_cfg3_32,
            DecoderConfig=decoder_cfg3_32,
            d_in=netG_moex3["n_channels"],
            d_out=z_32,
            phw=opt["phw"],
            overlap=opt["overlap"],
        )

        model_moex3_32 = ae2(cfg=autoenocer_cfg3_32)

        model_moex3_32.load_state_dict(
            torch.load(opt["pretrained_models"]["moex3_x2_32"], weights_only=True),
            strict=True,
        )

        model_moex3_32.eval()
        for k, v in model_moex3_32.named_parameters():
            v.requires_grad = False
        model_moex3_32 = model_moex3_32.to(device)

        json_dpsr = """
            {
            "netG": {
                "net_type": "dpsr",
                "in_nc": 1,
                "out_nc": 1,
                "nc": 96,
                "nb": 16,
                "gc": 32,
                "ng": 2,
                "reduction": 16,
                "act_mode": "R",
                "upsample_mode": "pixelshuffle",
                "downsample_mode": "strideconv",
                "init_type": "orthogonal",
                "init_bn_type": "uniform",
                "init_gain": 0.2,
                "scale": 2,
                "n_channels": 1,
                "ang_res": 5,
                "phw": 16,
                "overlap": 10
                }
            }
            """

        netG_dpsr = json.loads(json_dpsr)["netG"]

        model_dpsr = dpsr(
            in_nc=netG_dpsr["in_nc"],
            out_nc=netG_dpsr["out_nc"],
            nc=netG_dpsr["nc"],
            nb=netG_dpsr["nb"],
            upscale=netG_dpsr["scale"],
            act_mode=netG_dpsr["act_mode"],
            upsample_mode=netG_dpsr["upsample_mode"],
        )

        model_dpsr.load_state_dict(
            torch.load(opt["pretrained_models"]["dpsr_x2"], weights_only=True),
            strict=True,
        )
        model_dpsr.eval()
        for k, v in model_dpsr.named_parameters():
            v.requires_grad = False
        model_dpsr = model_dpsr.to(device)

        json_rrdb = """
        {
            "netG": {
                "net_type": "rrdb",
                "in_nc": 1,
                "out_nc": 1,
                "nc": 64,
                "nb": 23,
                "gc": 32,
                "ng": 2,
                "reduction": 16,
                "act_mode": "R",
                "upsample_mode": "upconv",
                "downsample_mode": "strideconv",
                "init_type": "orthogonal",
                "init_bn_type": "uniform",
                "init_gain": 0.2,
                "scale": 2,
                "n_channels": 1,
                "ang_res": 5
            }
        }
        """
        netG_rrdb = json.loads(json_rrdb)["netG"]

        model_esrgan = rrdb(
            in_nc=netG_rrdb["in_nc"],
            out_nc=netG_rrdb["out_nc"],
            nc=netG_rrdb["nc"],
            nb=netG_rrdb["nb"],
            gc=netG_rrdb["gc"],
            upscale=netG_rrdb["scale"],
            act_mode=netG_rrdb["act_mode"],
            upsample_mode=netG_rrdb["upsample_mode"],
        )

        model_esrgan.load_state_dict(
            torch.load(opt["pretrained_models"]["esrgan_x2"], weights_only=True),
            strict=True,
        )
        model_esrgan.eval()
        for k, v in model_esrgan.named_parameters():
            v.requires_grad = False
        model_esrgan = model_esrgan.to(device)

        # titles = [
        #     "High Resolution",
        #     "Low Resolution Crop",
        #     "High Resolution Crop",
        #     "N-SMoE",
        #     "DPSR",
        # ]

        model_cfg = "sam2_hiera_l.yaml"

        sam2 = build_sam2(
            model_cfg,
            opt["pretrained_models"]["sam2"],
            device="cuda",
            apply_postprocessing=True,
        )

        # mask_generator = SAM2AutomaticMaskGenerator(
        #     model=sam2,
        #     points_per_side=256,  # Very high density for the finest details
        #     points_per_batch=128,  # More points per batch for thorough segmentation
        #     pred_iou_thresh=0.7,  # Balanced IoU threshold for quality masks
        #     stability_score_thresh=0.95,  # High stability score threshold for the most stable masks
        #     stability_score_offset=1.0,
        #     mask_threshold=0.0,
        #     box_nms_thresh=0.7,
        #     crop_n_layers=4,  # More layers for multi-level cropping
        #     crop_nms_thresh=0.7,
        #     crop_overlap_ratio=512 / 1500,
        #     crop_n_points_downscale_factor=2,  # Adjusted for better point distribution
        #     min_mask_region_area=20,  # Small region processing to remove artifacts
        #     output_mode="binary_mask",
        #     use_m2m=True,  # Enable M2M refinement
        #     multimask_output=True,
        # )

        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            # points_per_side=128,
            # points_per_batch=128,
            # pred_iou_thresh=0.7,
            # stability_score_thresh=0.92,
            # stability_score_offset=0.7,
            # crop_n_layers=4,
            # crop_overlap_ratio=512 / 1500,
            # box_nms_thresh=0.7,
            # crop_n_points_downscale_factor=2,
            # min_mask_region_area=25.0,
            # use_m2m=True,
        )

        methods: List[str] = [
            "Bicubic",
            "DPSR",
            "ESRGAN",
            "N-SMoE",
            "N-SMoE-II",
            "N-SMoE-III",
        ]

        models = {
            "N-SMoE": model_moex1,
            "N-SMoE-II": model_moex3,
            "N-SMoE-III": model_moex3_32,
            "DPSR": model_dpsr,
            "ESRGAN": model_esrgan,
            "Bicubic": default_resizer,
        }

        metrics = ["psnr", "ssim", "lpips", "dists", "brisque"]
        metric_data = {
            metric: {
                method: {dataset: {} for dataset in opt["datasets"].keys()}
                for method in methods
            }
            for metric in metrics
        }
        average_metric_data = {}

        timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_dir = os.path.join(opt["path"]["root"], "metrics")
        latex_dir = os.path.join(opt["path"]["root"], "latex")
        util.mkdir(csv_dir)
        util.mkdir(latex_dir)

        for phase, dataset_opt in opt["datasets"].items():
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=16,
                drop_last=False,
                pin_memory=True,
                collate_fn=util.custom_pad_collate_fn,
            )

            H_img_size = dataset_opt["H_size"]
            degrdation = dataset_opt["degradation_type"]
            scale = f'x{dataset_opt["scale"]}'
            dataset_name = dataset_opt["name"]

            for method in methods:
                for metric in metrics:
                    metric_data[metric][method][dataset_name][scale] = []

            fmetric_name = os.path.join(
                csv_dir,
                degrdation
                + "_"
                + timestamp.replace(" ", "_").replace(":", "-")
                + ".csv",
            )
            flatex_table = os.path.join(
                latex_dir,
                timestamp.replace(" ", "_").replace(":", "-") + "_" + "latex_table.txt",
            )
            # avg_psnr = 0.0
            idx = 0

            for test_data in test_loader:
                if test_data is None:
                    continue

                idx += 1
                image_name_ext = os.path.basename(test_data["L_path"][0])
                img_name, ext = os.path.splitext(image_name_ext)

                img_dir = os.path.join(opt["path"]["images"], img_name)
                util.mkdir(img_dir)

                fname = os.path.join(
                    img_dir,
                    f"{img_name}_{degrdation}_{dataset_name}_{timestamp.replace(' ', '_').replace(':', '-')}",
                )
                figure_path = f"{fname}.pdf"
                seg_figure_path = os.path.join(
                    img_dir,
                    f"seg-{img_name}_{degrdation}_{dataset_name}_{timestamp.replace(' ', '_').replace(':', '-')}.pdf",
                )

                error_map_figure_path = os.path.join(
                    img_dir,
                    f"error-map-{img_name}_{degrdation}_{dataset_name}_{timestamp.replace(' ', '_').replace(':', '-')}.pdf",
                )

                results = process_data(test_data, models, metrics, device)

                for method in methods:
                    for metric in metrics:
                        value = results[method][metric]
                        scalar_value = (
                            value.item() if isinstance(value, torch.Tensor) else value
                        )
                        metric_data[metric][method][dataset_name][scale].append(
                            scalar_value
                        )

                for method in methods:
                    print(f"{method}:")
                    for metric in metrics:
                        print(f"  {metric.upper()}: {results[method][metric]}")

                # E_img_moex1 = util.tensor2uint(results["N-SMoE"]["e_img"])
                # E_img_dpsr = util._tensor2uint(results["DPSR"]["e_img"])
                # E_img_esrgan = util._tensor2uint(results["ESRGAN"]["e_img"])
                # E_bicubic = util._tensor2uint(results["Bicubic"]["e_img"])

                # L_crop_img = util.tensor2uint(test_data["L"])
                # H_crop_img = util.tensor2uint(test_data["H"])

                # img_H = util.tensor2uint(test_data["O"])
                # # img_H = util.imread_uint(test_data["H_path"][0], n_channels=1)
                # img_H = util.modcrop(img_H, border)

                # images: dict[str, Any] = {
                #     "H_img": img_H,
                #     "H_img_size": H_img_size,
                #     "L_crop_img": L_crop_img,
                #     "H_crop_img": H_crop_img,
                #     "E_Bicubic_img": E_bicubic,
                #     "E_SMoE_img": E_img_moex1,
                #     "E_DPSR_img": E_img_dpsr,
                #     "E_ESRGAN_img": E_img_esrgan,
                #     "Degradation_Model": degrdation,
                #     "scale": scale,
                # }

                # scipy.io.savemat(f"{fname}.mat", images)

                # visualize_data([L_crop_img, H_crop_img, E_img_moex1], titles)

                titles: list[str] = [
                    "HR",
                    "Noisy LR Crop",
                    "Ground Truth Crop",
                    "Bicubic",
                    "N-SMoE",
                    "DPSR",
                    "ESRGAN",
                ]

                # visualize_with_segmentation(
                #     [
                #         img_H,
                #         L_crop_img,
                #         H_crop_img,
                #         E_bicubic,
                #         E_img_moex1,
                #         E_img_dpsr,
                #         E_img_esrgan,
                #     ],
                #     titles,
                #     mask_generator,
                #     cmap="gray",
                #     save_path=seg_figure_path,
                #     visualize=opt["visualize"],
                # )

                # visualize_with_error_map(
                #     [
                #         img_H,
                #         L_crop_img,
                #         H_crop_img,
                #         E_bicubic,
                #         E_img_moex1,
                #         E_img_dpsr,
                #         E_img_esrgan,
                #     ],
                #     titles,
                #     cmap="gray",
                #     save_path=error_map_figure_path,
                #     visualize=opt["visualize"],
                # )
                # visualize_data(
                #     [
                #         L_crop_img,
                #         H_crop_img,
                #         E_bicubic,
                #         E_img_moex1,
                #         E_img_dpsr,
                #         E_img_esrgan,
                #     ],
                #     titles[1:],
                #     cmap="gray",
                #     save_path=figure_path,
                #     visualize=opt["visualize"],
                # )

                # current_psnr = util.calculate_psnr(
                #     E_img_moex1, H_crop_img, border=border
                # )
                # logger.info(
                #     "{:->4d}--> {:>10s} | {:<4.2f}dB".format(
                #         idx, image_name_ext, current_psnr
                #     )
                # )

                # avg_psnr += current_psnr

        for metric in metrics:
            average_metric_data[metric] = {}
            for method in methods:
                average_metric_data[metric][method] = {}
                for dataset in opt["datasets"].keys():
                    average_metric_data[metric][method][dataset] = {}
                    for scale in metric_data[metric][method][dataset].keys():

                        values = metric_data[metric][method][dataset][scale]

                        if values:
                            average = sum(values) / len(values)
                        else:
                            average = float("nan")

                        average_metric_data[metric][method][dataset][scale] = average

        with open(fmetric_name, "w", newline="") as csvfile:

            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [
                    "Dataset",
                    "Degradation",
                    "Scale",
                    "Image_Size",
                    "Method",
                    "PSNR",
                    "SSIM",
                    "LPIPS",
                    "DISTS",
                    "Diff_PSNR",
                    "Diff_SSIM",
                ]
            )

            for dataset in sorted(opt["datasets"].keys()):
                for method in methods:
                    for scale in sorted(
                        average_metric_data["psnr"][method][dataset].keys()
                    ):
                        avg_psnr = average_metric_data["psnr"][method][dataset][scale]
                        avg_ssim = average_metric_data["ssim"][method][dataset][scale]
                        avg_lpips = average_metric_data["lpips"][method][dataset][scale]
                        avg_dists = average_metric_data["dists"][method][dataset][scale]
                        ref_psnr = average_metric_data["psnr"]["N-SMoE"][dataset][scale]
                        ref_ssim = average_metric_data["ssim"]["N-SMoE"][dataset][scale]
                        diff_psnr = ref_psnr - avg_psnr
                        diff_ssim = ref_ssim - avg_ssim

                        csvwriter.writerow(
                            [
                                dataset,
                                degrdation,
                                scale,
                                H_img_size,
                                method,
                                f"{avg_psnr:.4f}",
                                f"{avg_ssim:.4f}",
                                f"{avg_lpips:.4f}",
                                f"{avg_dists:.4f}",
                                f"{diff_psnr:.4f}",
                                f"{diff_ssim:.4f}",
                            ]
                        )

        print(f"Results for all datasets saved to CSV file: {fmetric_name}")

        latex_table = gen_latex_table(average_metric_data)
        with open(flatex_table, "w") as f:
            f.write(latex_table)
        print(f"Latex table saved to {flatex_table}")

    elif task == "sr_x4":
        from models.network_dpsr import MSRResNet_prior as dpsr
        from models.network_rrdb import RRDB as rrdb
        from models.network_unetmoex1 import (
            Autoencoder,
            AutoencoderConfig,
            EncoderConfig,
            MoEConfig,
        )

        json_moex1 = """
        {
            "netG": {
                "net_type": "unet_moex1",
                "kernel": 16,
                "sharpening_factor": 1.3,
                "model_channels": 64,
                "num_res_blocks": 8,
                "attention_resolutions": [16,8,4],
                "dropout": 0.2,
                "num_groups": 8,
                "num_heads": 32,
                "num_head_channels": 32,
                "use_new_attention_order": true,
                "use_checkpoint": true,
                "resblock_updown": false,
                "channel_mult": [1,2,4,8],
                "resample_2d": false,
                "pool": "attention",
                "activation": "GELU",
                "resizer_num_layers": 2,
                "resizer_avg_pool": false,
                "scale": 2,
                "n_channels": 1
            }
        }
        """

        netG_moex1 = json.loads(json_moex1)["netG"]

        z = 2 * netG_moex1["kernel"] + 4 * netG_moex1["kernel"] + netG_moex1["kernel"]

        encoder_cfg = EncoderConfig(
            model_channels=netG_moex1["model_channels"],
            num_res_blocks=netG_moex1["num_res_blocks"],
            attention_resolutions=netG_moex1["attention_resolutions"],
            dropout=netG_moex1["dropout"],
            num_groups=netG_moex1["num_groups"],
            scale_factor=netG_moex1["scale"],
            num_heads=netG_moex1["num_heads"],
            num_head_channels=netG_moex1["num_head_channels"],
            use_new_attention_order=netG_moex1["use_new_attention_order"],
            use_checkpoint=netG_moex1["use_checkpoint"],
            resblock_updown=netG_moex1["resblock_updown"],
            channel_mult=netG_moex1["channel_mult"],
            resample_2d=netG_moex1["resample_2d"],
            pool=netG_moex1["pool"],
            activation=netG_moex1["activation"],
        )

        decoder_cfg = MoEConfig(
            kernel=netG_moex1["kernel"],
            sharpening_factor=netG_moex1["sharpening_factor"],
        )

        autoenocer_cfg = AutoencoderConfig(
            EncoderConfig=encoder_cfg,
            DecoderConfig=decoder_cfg,
            d_in=netG_moex1["n_channels"],
            d_out=z,
            phw=opt["phw"],
            overlap=opt["overlap"],
        )

        model_moex1 = Autoencoder(cfg=autoenocer_cfg)

        model_moex1.load_state_dict(
            torch.load(opt["pretrained_models"]["moex1_x2"], weights_only=True),
            strict=True,
        )
        model_moex1.eval()
        for k, v in model_moex1.named_parameters():
            v.requires_grad = False
        model_moex1 = model_moex1.to(device)

        json_dpsr = """
            {
            "netG": {
                "net_type": "dpsr",
                "in_nc": 1,
                "out_nc": 1,
                "nc": 96,
                "nb": 16,
                "gc": 32,
                "ng": 2,
                "reduction": 16,
                "act_mode": "R",
                "upsample_mode": "pixelshuffle",
                "downsample_mode": "strideconv",
                "init_type": "orthogonal",
                "init_bn_type": "uniform",
                "init_gain": 0.2,
                "scale": 2,
                "n_channels": 1,
                "ang_res": 5,
                "phw": 16,
                "overlap": 10
                }
            }
            """

        netG_dpsr = json.loads(json_dpsr)["netG"]

        model_dpsr = dpsr(
            in_nc=netG_dpsr["in_nc"],
            out_nc=netG_dpsr["out_nc"],
            nc=netG_dpsr["nc"],
            nb=netG_dpsr["nb"],
            upscale=netG_dpsr["scale"],
            act_mode=netG_dpsr["act_mode"],
            upsample_mode=netG_dpsr["upsample_mode"],
        )

        model_dpsr.load_state_dict(
            torch.load(opt["pretrained_models"]["dpsr_X2"], weights_only=True),
            strict=True,
        )
        model_dpsr.eval()
        for k, v in model_dpsr.named_parameters():
            v.requires_grad = False
        model_dpsr = model_dpsr.to(device)

        json_rrdb = """
        {
            "netG": {
                "net_type": "rrdb",
                "in_nc": 1,
                "out_nc": 1,
                "nc": 64,
                "nb": 23,
                "gc": 32,
                "ng": 2,
                "reduction": 16,
                "act_mode": "R",
                "upsample_mode": "upconv",
                "downsample_mode": "strideconv",
                "init_type": "orthogonal",
                "init_bn_type": "uniform",
                "init_gain": 0.2,
                "scale": 2,
                "n_channels": 1,
                "ang_res": 5
            }
        }
        """
        netG_rrdb = json.loads(json_rrdb)["netG"]

        model_esrgan = rrdb(
            in_nc=netG_rrdb["in_nc"],
            out_nc=netG_rrdb["out_nc"],
            nc=netG_rrdb["nc"],
            nb=netG_rrdb["nb"],
            gc=netG_rrdb["gc"],
            upscale=netG_rrdb["scale"],
            act_mode=netG_rrdb["act_mode"],
            upsample_mode=netG_rrdb["upsample_mode"],
        )

        model_esrgan.load_state_dict(
            torch.load(opt["pretrained_models"]["esrgan_x2"], weights_only=True),
            strict=True,
        )
        model_esrgan.eval()
        for k, v in model_esrgan.named_parameters():
            v.requires_grad = False
        model_esrgan = model_esrgan.to(device)

        # titles = [
        #     "High Resolution",
        #     "Low Resolution Crop",
        #     "High Resolution Crop",
        #     "N-SMoE",
        #     "DPSR",
        # ]

        model_cfg = "sam2_hiera_l.yaml"

        sam2 = build_sam2(
            model_cfg,
            opt["pretrained_models"]["sam2"],
            device="cuda",
            apply_postprocessing=True,
        )

        # mask_generator = SAM2AutomaticMaskGenerator(
        #     model=sam2,
        #     points_per_side=256,  # Very high density for the finest details
        #     points_per_batch=128,  # More points per batch for thorough segmentation
        #     pred_iou_thresh=0.7,  # Balanced IoU threshold for quality masks
        #     stability_score_thresh=0.95,  # High stability score threshold for the most stable masks
        #     stability_score_offset=1.0,
        #     mask_threshold=0.0,
        #     box_nms_thresh=0.7,
        #     crop_n_layers=4,  # More layers for multi-level cropping
        #     crop_nms_thresh=0.7,
        #     crop_overlap_ratio=512 / 1500,
        #     crop_n_points_downscale_factor=2,  # Adjusted for better point distribution
        #     min_mask_region_area=20,  # Small region processing to remove artifacts
        #     output_mode="binary_mask",
        #     use_m2m=True,  # Enable M2M refinement
        #     multimask_output=True,
        # )

        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=128,
            points_per_batch=128,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25.0,
            use_m2m=True,
        )

        avg_psnr = 0.0
        idx = 0

        psnr_moex_list: list[torch.float] = []
        psnr_dpsr_list: list[torch.float] = []
        psnr_esrgan_list = []
        psnr_bicubic_list = []

        ssim_moex_list: list[torch.float] = []
        ssim_dpsr_list: list[torch.float] = []
        ssim_esrgan_list: list[torch.float] = []
        ssim_bicubic_list: list[torch.float] = []

        lpips_moex_list: list[torch.float] = []
        lpips_dpsr_list: list[torch.float] = []
        lpips_esrgan_list: list[torch.float] = []
        lpips_bicubic_list: list[torch.float] = []

        dists_moex_list: list[torch.float] = []
        dists_dpsr_list: list[torch.float] = []
        dists_esrgan_list: list[torch.float] = []
        dists_bicubic_list: list[torch.float] = []

        H_img_size = opt["datasets"]["test"]["H_size"]
        scale: str = f'x{opt["scale"]}'

        timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        methods: List[str] = ["Bicubic", "DPSR", "ESRGAN", "N-SMoE"]
        csv_dir = os.path.join(opt["path"]["root"], dataset_name)
        util.mkdir(csv_dir)
        fmetric_name = os.path.join(
            csv_dir,
            degrdation
            + "_"
            + dataset_name
            + "_"
            + timestamp.replace(" ", "_").replace(":", "-"),
        )
        for test_data in test_loader:
            if test_data is None:
                continue

            idx += 1
            image_name_ext = os.path.basename(test_data["L_path"][0])
            img_name, ext = os.path.splitext(image_name_ext)

            img_dir = os.path.join(opt["path"]["images"], img_name)
            util.mkdir(img_dir)

            fname = os.path.join(
                img_dir,
                f"{img_name}_{degrdation}_{dataset_name}_{timestamp.replace(' ', '_').replace(':', '-')}",
            )
            figure_path = f"{fname}.pdf"
            seg_figure_path = os.path.join(
                img_dir,
                f"seg-{img_name}_{degrdation}_{dataset_name}_{timestamp.replace(' ', '_').replace(':', '-')}.pdf",
            )
            with torch.no_grad():
                E_img_moex1 = model_moex1(
                    test_data["L_p"].to(device), test_data["L"].size()
                )

                E_img_dpsr = model_dpsr(test_data["L"].to(device))
                E_img_esrgan = model_esrgan(test_data["L"].to(device))
                E_bicubic = default_resizer(test_data["L"], test_data["H"].size()[2:])
            # gt_img = (test_data["H"].mul(255.0).clamp(0, 255).to(torch.uint8)).to(device)
            # E_img_moex_t = E_img_moex1.mul(255.0).clamp(0, 255).to(torch.uint8)
            # E_img_dpsr_t = E_img_dpsr.mul(255.0).clamp(0, 255).to(torch.uint8)
            # E_img_esrgan_t = E_img_esrgan.mul(255.0).clamp(0, 255).to(torch.uint8)

            gt_img = (test_data["H"].clamp(0, 1).to(torch.float)).to(device)
            E_img_moex_t = E_img_moex1.clamp(0, 1).to(torch.float)
            E_img_dpsr_t = E_img_dpsr.clamp(0, 1).to(torch.float)
            E_img_esrgan_t = E_img_esrgan.clamp(0, 1).to(torch.float)
            E_bicubic_t = E_bicubic.clamp(0, 1).to(torch.float).to(device)

            psnr_moex1 = piq.psnr(E_img_moex_t, gt_img, data_range=1).float()
            psnr_dpsr = piq.psnr(E_img_dpsr_t, gt_img, data_range=1).float()
            psnr_esrgan = piq.psnr(E_img_esrgan_t, gt_img, data_range=1).float()
            psnr_bicubic = piq.psnr(E_bicubic_t, gt_img, data_range=1).float()

            ssim_moex1 = piq.ssim(E_img_moex_t, gt_img, data_range=1, reduction="mean")
            ssim_dpsr = piq.ssim(E_img_dpsr_t, gt_img, data_range=1, reduction="mean")
            ssim_esrgan = piq.ssim(
                E_img_esrgan_t, gt_img, data_range=1, reduction="mean"
            )
            ssim_bicubic = piq.ssim(E_bicubic_t, gt_img, data_range=1, reduction="mean")

            lpips_moex1 = piq.LPIPS()(E_img_moex_t, gt_img).item()
            lpips_dpsr = piq.LPIPS()(E_img_dpsr_t, gt_img).item()
            lpips_esrgan = piq.LPIPS()(E_img_esrgan_t, gt_img).item()
            lpips_bicubic = piq.LPIPS()(E_bicubic_t, gt_img).item()

            dists_moex1 = piq.DISTS()(E_img_moex_t, gt_img).item()
            dists_dpsr = piq.DISTS()(E_img_dpsr_t, gt_img).item()
            dists_esrgan = piq.DISTS()(E_img_esrgan_t, gt_img).item()
            dists_bicubic = piq.DISTS()(E_bicubic_t, gt_img).item()

            brisque_moex1 = piq.brisque(E_img_moex_t, data_range=1.0, reduction="none")
            brisque_dpsr = piq.brisque(E_img_dpsr_t, data_range=1.0, reduction="none")
            brisque_esrgan = piq.brisque(
                E_img_esrgan_t, data_range=1.0, reduction="none"
            )
            brisque_bicubic = piq.brisque(E_bicubic_t, data_range=1.0, reduction="none")

            print(
                f"PSNR N-SMoE: {psnr_moex1}, PSNR DPSR: {psnr_dpsr}, PSNR ESRGAN: {psnr_esrgan}, PSNR Bicubic: {psnr_bicubic}",
            )

            print(
                f"SSIM N-SMoE: {ssim_moex1}, SSIM DPSR: {ssim_dpsr}, SSIM ESRGAN: {ssim_esrgan}, SSIM Bicubic: {ssim_bicubic}"
            )

            print(
                f"LPIPS N-SMoE: {lpips_moex1}, LPIPS DPSR: {lpips_dpsr}, LPIPS ESRGAN: {lpips_esrgan}, LPIPS Bicubic: {lpips_bicubic}"
            )

            print(
                f"DISTS N-SMoE: {dists_moex1}, DISTS DPSR: {dists_dpsr}, DISTS ESRGAN: {dists_esrgan}, DISTS Bicubic: {dists_bicubic}"
            )

            print(
                f"Brisque N-SMoE: {brisque_moex1}, Brisque DPSR: {brisque_dpsr}, Brisque ESRGAN: {brisque_esrgan}, Brisque Bicubic: {brisque_bicubic}"
            )

            psnr_moex_list.append(psnr_moex1)
            psnr_dpsr_list.append(psnr_dpsr)
            psnr_esrgan_list.append(psnr_esrgan)
            psnr_bicubic_list.append(psnr_bicubic)

            ssim_moex_list.append(ssim_moex1)
            ssim_dpsr_list.append(ssim_dpsr)
            ssim_esrgan_list.append(ssim_esrgan)
            ssim_bicubic_list.append(ssim_bicubic)

            lpips_moex_list.append(lpips_moex1)
            lpips_dpsr_list.append(lpips_dpsr)
            lpips_esrgan_list.append(lpips_esrgan)
            lpips_bicubic_list.append(lpips_bicubic)

            dists_moex_list.append(dists_moex1)
            dists_dpsr_list.append(dists_dpsr)
            dists_esrgan_list.append(dists_esrgan)
            dists_bicubic_list.append(dists_bicubic)

            E_img_moex1 = util.tensor2uint(E_img_moex1)
            E_img_dpsr = util._tensor2uint(E_img_dpsr)
            E_img_esrgan = util._tensor2uint(E_img_esrgan)
            E_bicubic = util._tensor2uint(E_bicubic)

            L_crop_img = util.tensor2uint(test_data["L"])
            H_crop_img = util.tensor2uint(test_data["H"])

            img_H = util.tensor2uint(test_data["O"])
            # img_H = util.imread_uint(test_data["H_path"][0], n_channels=1)
            img_H = util.modcrop(img_H, border)

            images: dict[str, Any] = {
                "H_img": img_H,
                "H_img_size": H_img_size,
                "L_crop_img": L_crop_img,
                "H_crop_img": H_crop_img,
                "E_Bicubic_img": E_bicubic,
                "E_SMoE_img": E_img_moex1,
                "E_DPSR_img": E_img_dpsr,
                "E_ESRGAN_img": E_img_esrgan,
                "Degradation_Model": degrdation,
                "scale": scale,
            }

            scipy.io.savemat(f"{fname}.mat", images)

            # visualize_data([L_crop_img, H_crop_img, E_img_moex1], titles)

            titles: list[str] = [
                "HR",
                "Noisy LR Crop",
                "Ground Truth Crop",
                "Bicubic",
                "N-SMoE",
                "DPSR",
                "ESRGAN",
            ]

            visualize_with_segmentation(
                [
                    img_H,
                    L_crop_img,
                    H_crop_img,
                    E_bicubic,
                    E_img_moex1,
                    E_img_dpsr,
                    E_img_esrgan,
                ],
                titles,
                mask_generator,
                cmap="gray",
                save_path=seg_figure_path,
                visualize=opt["visualize"],
            )

            visualize_data(
                [
                    L_crop_img,
                    H_crop_img,
                    E_bicubic,
                    E_img_moex1,
                    E_img_dpsr,
                    E_img_esrgan,
                ],
                titles[1:],
                cmap="gray",
                save_path=figure_path,
                visualize=opt["visualize"],
            )

            current_psnr = util.calculate_psnr(E_img_moex1, H_crop_img, border=border)
            logger.info(
                "{:->4d}--> {:>10s} | {:<4.2f}dB".format(
                    idx, image_name_ext, current_psnr
                )
            )

            avg_psnr += current_psnr

        avg_psnr_moex = torch.tensor(psnr_moex_list).mean().float()
        avg_psnr_dpsr = torch.tensor(psnr_dpsr_list).mean().float()
        avg_psnr_esrgan = torch.tensor(psnr_esrgan_list).mean().float()

        avg_ssim_moex = torch.tensor(ssim_moex_list).mean().float()
        avg_ssim_dpsr = torch.tensor(ssim_dpsr_list).mean().float()
        avg_ssim_esrgan = torch.tensor(ssim_esrgan_list).mean().float()

        avg_lpips_moex = torch.tensor(lpips_moex_list).mean().float()
        avg_lpips_dpsr = torch.tensor(lpips_dpsr_list).mean().float()
        avg_lpips_esrgan = torch.tensor(lpips_esrgan_list).mean().float()

        avg_dists_moex = torch.tensor(dists_moex_list).mean().float()
        avg_dists_dpsr = torch.tensor(dists_dpsr_list).mean().float()
        avg_dists_esrgan = torch.tensor(dists_esrgan_list).mean().float()

        print(f"Average PSNR N-SMoE: {avg_psnr_moex}")
        print(f"Average PSNR DPSR: {avg_psnr_dpsr}")
        print(f"Average PSNR ESRGAN: {avg_psnr_esrgan}")

        print(f"Average SSIM N-SMoE: {avg_ssim_moex}")
        print(f"Average SSIM DPSR: {avg_ssim_dpsr}")
        print(f"Average SSIM ESRGAN: {avg_ssim_esrgan}")

        print(f"Average LPIPS N-SMoE: {avg_lpips_moex}")
        print(f"Average LPIPS DPSR: {avg_lpips_dpsr}")
        print(f"Average LPIPS ESRGAN: {avg_lpips_esrgan}")

        print(f"Average DISTS N-SMoE: {avg_dists_moex}")
        print(f"Average DISTS DPSR: {avg_dists_dpsr}")
        print(f"Average DISTS ESRGAN: {avg_dists_esrgan}")

        psnr_values: List[torch.Tensor] = [
            avg_psnr_dpsr,
            avg_psnr_esrgan,
            avg_psnr_moex,
        ]

        ssim_values: List[torch.Tensor] = [
            avg_ssim_dpsr,
            avg_ssim_esrgan,
            avg_ssim_moex,
        ]

        lpips_values: List[torch.Tensor] = [
            avg_lpips_dpsr,
            avg_lpips_esrgan,
            avg_lpips_moex,
        ]

        dists_values: List[torch.Tensor] = [
            avg_dists_dpsr,
            avg_dists_esrgan,
            avg_dists_moex,
        ]

        diff_psnr_values: List[torch.Tensor] = [
            psnr_values[-1] - psnr for psnr in psnr_values[:-1]
        ]
        diff_ssim_values: List[torch.Tensor] = [
            ssim_values[-1] - ssim for ssim in ssim_values[:-1]
        ]

        with open((fmetric_name + "_metrics.csv"), "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow(
                [
                    "Dataset",
                    "Degradation",
                    "Scale",
                    "Image_Size",
                    "Method",
                    "PSNR",
                    "SSIM",
                    "LPIPS",
                    "DISTS",
                    "Diff_PSNR",
                    "Diff_SSIM",
                ]
            )

            for i, method in enumerate(methods):
                csvwriter.writerow(
                    [
                        dataset_name,
                        degrdation,
                        scale,
                        H_img_size,
                        method,
                        psnr_values[i].item(),
                        ssim_values[i].item(),
                        lpips_values[i].item(),
                        dists_values[i].item(),
                        (
                            diff_psnr_values[i].item()
                            if i < len(diff_psnr_values)
                            else "N/A"
                        ),
                        (
                            diff_ssim_values[i].item()
                            if i < len(diff_ssim_values)
                            else "N/A"
                        ),
                    ]
                )
            print(f"Results saved to CSV file: {fmetric_name}_metrics.csv")

    elif task == "sharpening":
        import matlab.engine

        eng = matlab.engine.start_matlab()

        matlab_func_dir = os.path.join(os.path.dirname(__file__), "matlab")

        eng.addpath(matlab_func_dir, nargout=0)

        def calculate_sharpness_index(image):
            image_np = image.squeeze().cpu().numpy()
            image_matlab = matlab.double(image_np.tolist())

            si = eng.sharpness_index(image_matlab)

            return si

        from models.network_unetmoex1 import Autoencoder as ae1
        from models.network_unetmoex1 import AutoencoderConfig as ae1_cfg
        from models.network_unetmoex1 import EncoderConfig as enc1_cfg
        from models.network_unetmoex1 import MoEConfig as moe1_cfg

        json_moex1 = """
        {
            "netG": {
                "net_type": "unet_moex1",
                "kernel": 16,
                "sharpening_factor": 1.3,
                "model_channels": 64,
                "num_res_blocks": 8,
                "attention_resolutions": [16,8,4],
                "dropout": 0.2,
                "num_groups": 8,
                "num_heads": 32,
                "num_head_channels": 32,
                "use_new_attention_order": true,
                "use_checkpoint": true,
                "resblock_updown": false,
                "channel_mult": [1,2,4,8],
                "resample_2d": false,
                "pool": "attention",
                "activation": "GELU",
                "resizer_num_layers": 2,
                "resizer_avg_pool": false,
                "scale": 2,
                "n_channels": 1
            }
        }
        """

        netG_moex1 = json.loads(json_moex1)["netG"]

        z = 2 * netG_moex1["kernel"] + 4 * netG_moex1["kernel"] + netG_moex1["kernel"]

        encoder_cfg = enc1_cfg(
            model_channels=netG_moex1["model_channels"],
            num_res_blocks=netG_moex1["num_res_blocks"],
            attention_resolutions=netG_moex1["attention_resolutions"],
            dropout=netG_moex1["dropout"],
            num_groups=netG_moex1["num_groups"],
            scale_factor=netG_moex1["scale"],
            num_heads=netG_moex1["num_heads"],
            num_head_channels=netG_moex1["num_head_channels"],
            use_new_attention_order=netG_moex1["use_new_attention_order"],
            use_checkpoint=netG_moex1["use_checkpoint"],
            resblock_updown=netG_moex1["resblock_updown"],
            channel_mult=netG_moex1["channel_mult"],
            resample_2d=netG_moex1["resample_2d"],
            pool=netG_moex1["pool"],
            activation=netG_moex1["activation"],
        )

        sharpening_factors = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

        timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        idx = 0
        for test_data in test_loader:
            if test_data is None:
                continue

            idx += 1
            image_name_ext = os.path.basename(test_data["L_path"][0])
            img_name, ext = os.path.splitext(image_name_ext)

            img_dir = os.path.join(opt["path"]["images"], img_name)
            util.mkdir(img_dir)

            fname = os.path.join(
                img_dir,
                f"{img_name}_{degrdation}_{dataset_name}_sharpening_{timestamp.replace(' ', '_').replace(':', '-')}",
            )
            si_figure_path = f"{fname}.pdf"

            img_L = test_data["L"].to(device)
            img_H = test_data["H"].to(device)
            img_L_p = test_data["L_p"].to(device)
            img_L_size = test_data["L"].size()
            sharpened_images = {}
            metrics = {}

            img_H = img_H.clamp(0, 1).to(torch.float).to(device)

            for factor in sharpening_factors:

                decoder_cfg = moe1_cfg(
                    kernel=netG_moex1["kernel"],
                    sharpening_factor=factor,
                )
                autoenocer_cfg = ae1_cfg(
                    EncoderConfig=encoder_cfg,
                    DecoderConfig=decoder_cfg,
                    d_in=netG_moex1["n_channels"],
                    d_out=z,
                    phw=opt["phw"],
                    overlap=opt["overlap"],
                )

                model_moex1 = ae1(cfg=autoenocer_cfg)
                model_moex1.load_state_dict(
                    torch.load(opt["pretrained_models"]["moex1_x2"], weights_only=True),
                    strict=True,
                )
                model_moex1.eval()
                for k, v in model_moex1.named_parameters():
                    v.requires_grad = False
                model_moex1 = model_moex1.to(device)

                with torch.no_grad():
                    E_img_moex1 = model_moex1(img_L_p, img_L_size)

                E_img_moex_t = E_img_moex1.clamp(0, 1).to(torch.float)
                sharpened_images[factor] = E_img_moex_t.squeeze().cpu().numpy()

                si_moex1 = calculate_sharpness_index(E_img_moex_t)
                psnr_moex1 = piq.psnr(E_img_moex_t, img_H, data_range=1).float()
                ssim_moex1 = piq.ssim(
                    E_img_moex_t, img_H, data_range=1, reduction="mean"
                )

                metrics[factor] = {
                    "PSNR": psnr_moex1.item(),
                    "SSIM": ssim_moex1.item(),
                    "SI": si_moex1,
                }

                print(
                    f"Image {idx}, Sharpening factor {factor}: PSNR = {psnr_moex1:.2f}, SSIM = {ssim_moex1:.4f}, SI = {si_moex1:.4f}"
                )

            visualize_sharpening_results(
                img_L.cpu().numpy().squeeze(),
                img_H.cpu().numpy().squeeze(),
                sharpened_images,
                metrics,
                save_path=si_figure_path,
                visualize=opt["visualize"],
            )
        eng.quit()

    elif task == "upsampling":
        print("Upsampling task")


if __name__ == "__main__":
    main()
