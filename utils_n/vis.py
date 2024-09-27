from typing import Dict, List, Any

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

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["text.usetex"] = True


# def visualize_with_segmentation(
#     images: List[np.ndarray],
#     titles: List[str],
#     mask_generator: SAM2AutomaticMaskGenerator,
#     cmap: str = "gray",
#     save_path: str = None,
#     visualize: bool = False,
#     error_map: bool = False,
# ):

#     def calculate_metrics(gt_mask, pred_mask):
#         gt_seg = gt_mask[0]["segmentation"]
#         pred_seg = pred_mask[0]["segmentation"]

#         vi_split, vi_merge = variation_of_information(gt_seg, pred_seg)
#         vi_score = vi_split + vi_merge
#         are_score, _, _ = adapted_rand_error(gt_seg, pred_seg)

#         return vi_score, are_score

#     def show_anns(anns, borders=True):
#         if len(anns) == 0:
#             return
#         sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
#         ax = plt.gca()
#         ax.set_autoscale_on(False)
#         img = np.ones(
#             (
#                 sorted_anns[0]["segmentation"].shape[0],
#                 sorted_anns[0]["segmentation"].shape[1],
#                 4,
#             )
#         )
#         img[:, :, 3] = 0
#         for ann in sorted_anns:
#             m = ann["segmentation"]
#             color_mask = np.concatenate([np.random.random(3), [0.5]])
#             img[m] = color_mask
#             if borders:
#                 contours, _ = cv2.findContours(
#                     m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
#                 )
#                 contours = [
#                     cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
#                     for contour in contours
#                 ]
#                 cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
#         ax.imshow(img)

#     num_images = len(images)
#     num_titles = len(titles)
#     assert num_images == num_titles, "Number of images must match number of titles."

#     ncols = num_images + 1
#     nrows = 3
#     width_ratios = [2, 2] + [1] * (ncols - 2)
#     total_ratio = sum(width_ratios)

#     unit_width = 19 / 12

#     fig_width = unit_width * total_ratio

#     fig_height = 3.5

#     fig = plt.figure(figsize=(fig_width, fig_height))
#     gs = GridSpec(
#         nrows=nrows,
#         ncols=ncols,
#         height_ratios=[2, 2, 0.5],
#         width_ratios=width_ratios,
#         # width_ratios=[2, 2, 1, 1, 1, 1, 1, 1],
#         hspace=0.01,  # Reducing space between rows
#         wspace=0.01,  # Reducing space between columns
#         figure=fig,
#     )

#     annotated_mask = mask_generator.generate(
#         np.repeat(images[0][:, :, np.newaxis], 3, axis=-1)
#     )
#     ax_annotated = fig.add_subplot(gs[0:2, 0])
#     ax_annotated.imshow(images[0], cmap=cmap)
#     show_anns(annotated_mask)
#     ax_annotated.axis("off")
#     ax_annotated.set_title("Annotated Segmentation", fontsize=12, weight="bold")

#     ax_img_hr = fig.add_subplot(gs[0:2, 1])
#     ax_img_hr.imshow(images[0], cmap=cmap)

#     # ax_img_hr.axis("off")
#     # ax_img_hr.set_title(titles[0], fontsize=12, weight="bold")

#     ground_truth_index = 2
#     ground_truth_crop = images[ground_truth_index]
#     hr_image = images[0]

#     if len(hr_image.shape) == 3 and hr_image.shape[2] == 3:
#         gray_hr = cv2.cvtColor(hr_image, cv2.COLOR_RGB2GRAY)
#     else:
#         gray_hr = hr_image.copy()

#     if len(ground_truth_crop.shape) == 3 and ground_truth_crop.shape[2] == 3:
#         gray_crop = cv2.cvtColor(ground_truth_crop, cv2.COLOR_RGB2GRAY)
#     else:
#         gray_crop = ground_truth_crop.copy()

#     res = cv2.matchTemplate(gray_hr, gray_crop, cv2.TM_CCOEFF_NORMED)
#     _, max_val, _, max_loc = cv2.minMaxLoc(res)

#     x, y = max_loc
#     crop_height, crop_width = ground_truth_crop.shape[:2]

#     rect = patches.Rectangle(
#         (x, y), crop_width, crop_height, linewidth=2, edgecolor="r", facecolor="none"
#     )
#     ax_img_hr.add_patch(rect)

#     ax_img_hr.axis("off")
#     ax_img_hr.set_title(titles[0], fontsize=12, weight="bold")

#     gt_mask = None
#     vi_scores = {}
#     are_scores = {}

#     for i in range(1, len(images)):
#         ax_img = fig.add_subplot(gs[0, i + 1])
#         ax_img.imshow(images[i], cmap=cmap)
#         ax_img.axis("off")

#         mask = mask_generator.generate(np.repeat(images[i][:, :, None], 3, axis=-1))
#         ax_seg = fig.add_subplot(gs[1, i + 1])
#         ax_seg.imshow(images[i], cmap=cmap)
#         show_anns(mask)
#         ax_seg.axis("off")
#         ax_title = fig.add_subplot(gs[2, i + 1])

#         if i == ground_truth_index:
#             gt_mask = mask

#         if i > ground_truth_index:
#             vi_score, are_score = calculate_metrics(gt_mask, mask)
#             vi_scores[i] = vi_score
#             are_scores[i] = are_score

#     sorted_vi = sorted(vi_scores.items(), key=lambda x: x[1])
#     sorted_are = sorted(are_scores.items(), key=lambda x: x[1])

#     for i in range(1, len(images)):
#         ax_title = fig.add_subplot(gs[2, i + 1])
#         if i > ground_truth_index:
#             vi_score = vi_scores[i]
#             are_score = are_scores[i]

#             vi_text = f"VoI: {vi_score:.4f}"
#             are_text = f"ARE: {are_score:.4f}"

#             if i == sorted_vi[0][0]:
#                 vi_text = r"\textbf{VoI: %.4f}" % vi_score
#             elif i == sorted_vi[1][0]:
#                 vi_text = r"\underline{VoI: %.4f}" % vi_score

#             if i == sorted_are[0][0]:
#                 are_text = r"\textbf{ARE: %.4f}" % are_score
#             elif i == sorted_are[1][0]:
#                 are_text = r"\underline{ARE: %.4f}" % are_score

#             display_title = f"{titles[i]}\n{vi_text}\n{are_text}"

#             ax_title.text(
#                 0.5,
#                 0.0,
#                 display_title,
#                 va="center",
#                 ha="center",
#                 transform=ax_title.transAxes,
#             )
#         else:
#             display_title = titles[i]
#             ax_title.text(
#                 0.5,
#                 0.5,
#                 display_title,
#                 weight="bold",
#                 va="center",
#                 ha="center",
#                 transform=ax_title.transAxes,
#             )

#     plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)
#     plt.subplots_adjust(
#         left=0.12, bottom=0.12, right=0.89, top=0.89, wspace=0, hspace=0
#     )

#     for ax in fig.get_axes():
#         ax.axis("off")

#     if save_path:
#         plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=600)
#     if visualize:
#         plt.show()

def visualize_with_segmentation(
    images: Dict[str, Dict[str, Any]],
    mask_generator: SAM2AutomaticMaskGenerator,
    hr_key: str = "H_img",
    ref_key: str = "H_crop_img",
    lrcrop_key: str = "L_crop_img",
    cmap: str = "gray",
    save_path: str = None,
    visualize: bool = False
    error_map: bool = False) -> None:

    def calculate_metrics(gt_mask, pred_mask):
        gt_seg = gt_mask[0]["segmentation"]
        pred_seg = pred_mask[0]["segmentation"]

        vi_split, vi_merge = variation_of_information(gt_seg, pred_seg)
        vi_score = vi_split + vi_merge
        are_score, _, _ = adapted_rand_error(gt_seg, pred_seg)

        return vi_score, are_score

    def show_anns(anns, borders=True):
        if not anns or not all('segmentation' in ann for ann in anns):
            return
        sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
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

    # Extract titles from images dictionary
    titles = [v['title'] for k, v in images.items()]
    num_images = len(images)
    assert num_images == len(titles), "Number of images must match number of titles."

    # Identify reconstructed image keys
    recon_keys = [k for k in images.keys() if k not in [hr_key, ref_key, lrcrop_key]]
    num_recon = len(recon_keys)

    # Set up figure layout
    ncols = 2 + num_recon
    nrows = 3
    width_ratios = [2, 2] + [1] * num_recon
    height_ratios = [2, 2, 0.5]

    unit_width = 19 / 12
    fig_width = unit_width * sum(width_ratios)
    fig_height = 3.5

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = GridSpec(
        nrows=nrows,
        ncols=ncols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.01,
        wspace=0.01,
        figure=fig,
    )

    error_cmap = LinearSegmentedColormap.from_list(
        "custom_diverging", ["navy", "blue", "cyan", "limegreen", "yellow", "red"], N=256
    )

    # Annotated HR Image
    annotated_mask = mask_generator.generate(
        np.repeat(images[hr_key]['image'][:, :, np.newaxis], 3, axis=-1)
    )
    ax_annotated = fig.add_subplot(gs[0:2, 0])
    ax_annotated.imshow(images[hr_key]['image'], cmap=cmap)
    show_anns(annotated_mask)
    ax_annotated.axis("off")
    ax_annotated.set_title(titles[0], fontsize=12, weight="bold")

    # HR Image with Ground Truth Crop
    ax_img_hr = fig.add_subplot(gs[0:2, 1])
    ax_img_hr.imshow(images[hr_key]['image'], cmap=cmap)

    ground_truth_crop = images[ref_key]['image']
    hr_image = images[hr_key]['image']

    if len(hr_image.shape) == 3 and hr_image.shape[2] == 3:
        gray_hr = cv2.cvtColor(hr_image, cv2.COLOR_RGB2GRAY)
    else:
        gray_hr = hr_image.copy()

    if len(ground_truth_crop.shape) == 3 and ground_truth_crop.shape[2] == 3:
        gray_crop = cv2.cvtColor(ground_truth_crop, cv2.COLOR_RGB2GRAY)
    else:
        gray_crop = ground_truth_crop.copy()

    res = cv2.matchTemplate(gray_hr, gray_crop, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    x, y = max_loc
    crop_height, crop_width = ground_truth_crop.shape[:2]

    rect = patches.Rectangle(
        (x, y), crop_width, crop_height, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax_img_hr.add_patch(rect)

    ax_img_hr.axis("off")
    ax_img_hr.set_title(titles[1], fontsize=12, weight="bold")

    # Generate ground truth mask
    gt_mask = mask_generator.generate(
        np.repeat(images[ref_key]['image'][:, :, np.newaxis], 3, axis=-1)
    )

    vi_scores = {}
    are_scores = {}

    for i, key in enumerate(recon_keys, start=2):
        item = images[key]
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(item['image'], cmap=cmap)
        ax_img.axis("off")

        mask = mask_generator.generate(np.repeat(item['image'][:, :, None], 3, axis=-1))
        ax_seg = fig.add_subplot(gs[1, i])
        ax_seg.imshow(item['image'], cmap=cmap)
        show_anns(mask)
        ax_seg.axis("off")

        vi_score, are_score = calculate_metrics(gt_mask, mask)
        vi_scores[key] = vi_score
        are_scores[key] = are_score

    sorted_vi = sorted(vi_scores.items(), key=lambda x: x[1])
    sorted_are = sorted(are_scores.items(), key=lambda x: x[1])

    for i, key in enumerate(recon_keys, start=2):
        item = images[key]
        ax_title = fig.add_subplot(gs[2, i])
        ax_title.axis("off")

        if key in vi_scores and key in are_scores:
            vi_score = vi_scores[key]
            are_score = are_scores[key]

            vi_text = f"VoI: {vi_score:.4f}"
            are_text = f"ARE: {are_score:.4f}"

            if key == sorted_vi[0][0]:
                vi_text = r"\textbf{" + vi_text + "}"
            if len(sorted_vi) > 1 and key == sorted_vi[1][0]:
                vi_text = r"\underline{" + vi_text + "}"

            if key == sorted_are[0][0]:
                are_text = r"\textbf{" + are_text + "}"
            if len(sorted_are) > 1 and key == sorted_are[1][0]:
                are_text = r"\underline{" + are_text + "}"

            ax_title.text(
                0.5,
                0.7,
                vi_text,
                ha="center", va="center",
                transform=ax_title.transAxes,
                fontsize=10
            )
            ax_title.text(
                0.5,
                0.5,
                are_text,
                ha="center", va="center",
                transform=ax_title.transAxes,
                fontsize=10
            )
            ax_title.text(
                0.5,
                0.3,
                item['title'],
                ha="center", va="center",
                transform=ax_title.transAxes,
                fontsize=10,
            )
        else:
            display_title = item['title']
            ax_title.text(
                0.5,
                0.5,
                display_title,
                weight="bold",
                va="center",
                ha="center",
                transform=ax_title.transAxes,
                fontsize=10,
            )

    for ax in fig.get_axes():
        ax.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=600)
    if visualize:
        plt.show()



# def visualize_with_error_map(
#     images: List[np.ndarray],
#     titles: List[str],
#     cmap: str = "gray",
#     save_path: str = None,
#     visualize: bool = True,
# ) -> None:

#     def create_error_cmap():
#         colors = ["navy", "blue", "cyan", "limegreen", "yellow", "red"]
#         return LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)

#     def calculate_error_map(gt_image, reconstructed_image):
#         return (gt_image.astype(float) - reconstructed_image.astype(float)) ** 2

#     num_images = len(images)
#     num_titles = len(titles)
#     assert num_images == num_titles, "Number of images must match number of titles."

#     if num_images < 2:
#         raise ValueError(
#             "At least two images are required: reference and at least one reconstructed image."
#         )
#     ncols = num_images
#     nrows = 3
#     width_ratios = [2] + [1] * (ncols - 1)
#     total_ratio = sum(width_ratios)

#     unit_width = 19 / 12

#     fig_width = unit_width * total_ratio

#     fig_height = 3.5

#     fig = plt.figure(figsize=(fig_width, fig_height))

#     gs = GridSpec(
#         nrows=nrows,
#         ncols=ncols,
#         height_ratios=[2, 2, 0.5],
#         width_ratios=width_ratios,
#         hspace=0.01,
#         wspace=0.01,
#         figure=fig,
#     )

#     gt_index = 2
#     ref_img = images[gt_index].squeeze()

#     psnr_values = {}
#     ssim_values = {}
#     mse_values = {}

#     error_cmap = create_error_cmap()

#     ax_img_hr = fig.add_subplot(gs[0:2, 0])
#     ax_img_hr.imshow(images[0], cmap=cmap)

#     hr_image = images[0]

#     if len(hr_image.shape) == 3 and hr_image.shape[2] == 3:
#         gray_hr = cv2.cvtColor(hr_image, cv2.COLOR_RGB2GRAY)
#     else:
#         gray_hr = hr_image.copy()

#     if len(ref_img.shape) == 3 and ref_img.shape[2] == 3:
#         gray_crop = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
#     else:
#         gray_crop = ref_img.copy()

#     res = cv2.matchTemplate(gray_hr, gray_crop, cv2.TM_CCOEFF_NORMED)
#     _, max_val, _, max_loc = cv2.minMaxLoc(res)

#     x, y = max_loc
#     crop_height, crop_width = ref_img.shape[:2]

#     rect = patches.Rectangle(
#         (x, y), crop_width, crop_height, linewidth=2, edgecolor="r", facecolor="none"
#     )
#     ax_img_hr.add_patch(rect)

#     ax_img_hr.axis("off")
#     ax_img_hr.set_title(titles[0], fontsize=12, fontweight="bold")

#     max_error = 0

#     for i in range(1, len(images)):
#         img = images[i]
#         title = titles[i]

#         ax_img = fig.add_subplot(gs[0, i])
#         ax_img.imshow(img, cmap=cmap)
#         ax_img.axis("off")

#         if i > 2:
#             error_map = calculate_error_map(ref_img, img)
#             error_map_normalized = (error_map - error_map.min()) / (
#                 error_map.max() - error_map.min()
#             )
#             max_error = max(max_error, error_map.max())

#             ax_error_map = fig.add_subplot(gs[1, i])
#             im = ax_error_map.imshow(
#                 error_map_normalized, cmap=error_cmap, vmin=0, vmax=1
#             )
#             ax_error_map.axis("off")

#             try:
#                 current_psnr = psnr(ref_img, img)
#                 current_ssim = ssim(ref_img, img, multichannel=True)
#                 current_mse = mse(ref_img, img)
#                 psnr_values[title] = current_psnr
#                 ssim_values[title] = current_ssim
#                 mse_values[title] = current_mse
#             except Exception as e:
#                 print(f"Error calculating PSNR/SSIM/MSE for {title}: {str(e)}")

#     sorted_psnr = sorted(psnr_values.items(), key=lambda x: x[1], reverse=True)
#     sorted_ssim = sorted(ssim_values.items(), key=lambda x: x[1], reverse=True)
#     sorted_mse = sorted(mse_values.items(), key=lambda x: x[1])

#     for i in range(1, len(images)):
#         title = titles[i]
#         ax_title = fig.add_subplot(gs[2, i])
#         ax_title.axis("off")

#         if i > 2 and title in psnr_values and title in ssim_values:
#             psnr_text = f"PSNR: {psnr_values[title]:.2f}"
#             ssim_text = f"SSIM: {ssim_values[title]:.4f}"
#             mse_text = f"MSE: {mse_values[title]:.4f}"

#             if title == sorted_psnr[0][0]:
#                 psnr_text = r"\textbf{" + psnr_text + "}"
#             elif title == sorted_psnr[1][0]:
#                 psnr_text = r"\underline{" + psnr_text + "}"

#             if title == sorted_ssim[0][0]:
#                 ssim_text = r"\textbf{" + ssim_text + "}"
#             elif title == sorted_ssim[1][0]:
#                 ssim_text = r"\underline{" + ssim_text + "}"

#             display_title = f"{title}\n${psnr_text}$ dB\n${ssim_text}$\n${mse_text}$"
#             ax_title.text(
#                 0.5,
#                 0.0,
#                 display_title,
#                 ha="center",
#                 va="center",
#                 transform=ax_title.transAxes,
#                 fontsize=10,
#             )
#         else:
#             display_title = title

#             ax_title.text(
#                 0.5,
#                 4.5,
#                 display_title,
#                 ha="center",
#                 va="center",
#                 transform=ax_title.transAxes,
#                 fontsize=10,
#             )

#     plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)
#     plt.subplots_adjust(
#         left=0.12, bottom=0.12, right=0.88, top=0.88, wspace=0, hspace=0
#     )

#     if save_path:
#         plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=600)
#     if visualize:
#         plt.show()


def visualize_with_error_map(
    images: Dict[str, Dict[str, Any]],
    hr_key: str = "H_img",
    ref_key: str = "H_crop_img",
    lrcrop_key: str = "L_crop_img",
    cmap: str = "gray",
    save_path: str = None,
    visualize: bool = True,
) -> None:
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

    ref_image = images[ref_key]["image"]
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

    unit_width = 20 / 13
    fig_width = unit_width * sum(width_ratios)
    fig_height = 3.5

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
    ax_hr.set_title(rf"\textbf{{{hr_title}}}", fontsize=12)

    ax_lrcrop = fig.add_subplot(gs[0, 1])
    ax_lrcrop.imshow(lrcrop_image, cmap=cmap)
    ax_lrcrop.axis("off")
    ax_lrcrop.set_title(rf"\textbf{{{lrcrop_title}}}", fontsize=10)

    ax_ref = fig.add_subplot(gs[0, 2])
    ax_ref.imshow(ref_image, cmap=cmap)
    ax_ref.axis("off")
    ax_ref.set_title(rf"\textbf{{{ref_title}}}", fontsize=10)

    gray_hr = (
        cv2.cvtColor(hr_image, cv2.COLOR_RGB2GRAY)
        if len(hr_image.shape) == 3 and hr_image.shape[2] == 3
        else hr_image.copy()
    )
    gray_ref = (
        cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
        if len(ref_image.shape) == 3 and ref_image.shape[2] == 3
        else ref_image.copy()
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
    mse_values = {}

    for idx, (key, item) in enumerate(recon_items.items(), start=3):
        recon_image = item["image"]
        recon_title = item["title"]

        ax_recon = fig.add_subplot(gs[0, idx])
        ax_recon.imshow(recon_image, cmap=cmap)
        ax_recon.axis("off")
        ax_recon.set_title(rf"\textbf{{{recon_title}}}", fontsize=10)

        error_map = calculate_error_map(ref_image, recon_image)
        error_map_normalized = (error_map - error_map.min()) / (
            error_map.max() - error_map.min()
            if error_map.max() - error_map.min() != 0
            else 1
        )

        ax_error = fig.add_subplot(gs[1, idx])
        ax_error.imshow(error_map_normalized, cmap=error_cmap, vmin=0, vmax=1)
        ax_error.axis("off")

        try:
            if ref_image.shape != recon_image.shape:
                raise ValueError(
                    f"Shape mismatch between reference image '{ref_key}' {ref_image.shape} "
                    f"and reconstructed image '{key}' {recon_image.shape}."
                )
            current_psnr = psnr(
                ref_image, recon_image, data_range=ref_image.max() - ref_image.min()
            )
            current_ssim = ssim(ref_image, recon_image, multichannel=True)
            current_mse = mse(ref_image, recon_image)

            psnr_values[recon_title] = current_psnr
            ssim_values[recon_title] = current_ssim
            mse_values[recon_title] = current_mse
        except Exception as e:
            print(f"Error calculating PSNR/SSIM/MSE for '{recon_title}': {str(e)}")

    sorted_psnr = sorted(psnr_values.items(), key=lambda x: x[1], reverse=True)
    sorted_ssim = sorted(ssim_values.items(), key=lambda x: x[1], reverse=True)
    sorted_mse = sorted(mse_values.items(), key=lambda x: x[1])

    for idx, (key, item) in enumerate(recon_items.items(), start=3):
        recon_title = item["title"]
        ax_title = fig.add_subplot(gs[2, idx])
        ax_title.axis("off")

        if (
            recon_title in psnr_values
            and recon_title in ssim_values
            and recon_title in mse_values
        ):
            psnr_val = psnr_values[recon_title]
            ssim_val = ssim_values[recon_title]
            mse_val = mse_values[recon_title]

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

            psnr_display = format_metric(recon_title, sorted_psnr, psnr_val, "PSNR")
            ssim_display = format_metric(recon_title, sorted_ssim, ssim_val, "SSIM")
            mse_display = format_metric(recon_title, sorted_mse, mse_val, "MSE")

            display_title = f"${psnr_display}$\n${ssim_display}$\n${mse_display}$"

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
            ax_title.text(
                0.4,
                0.4,
                recon_title,
                ha="center",
                va="center",
                transform=ax_title.transAxes,
                fontsize=10,
            )

    plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)
    plt.subplots_adjust(
        left=0.12, bottom=0.12, right=0.88, top=0.88, wspace=0, hspace=0
    )

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=600)
    if visualize:
        plt.show()


def visualize_data(
    images: List[np.ndarray],
    titles: List[str],
    cmap: str = "gray",
    save_path: str = None,
    visualize: bool = True,
) -> None:
    num_images = len(images)

    width_ratios = [1] * num_images
    total_width_ratio = sum(width_ratios)

    # desired_num_images = 9
    # desired_fig_width = 19  # inches for 9 images
    # unit_width = desired_fig_width / desired_num_images

    unit_width = 20 / 8

    fig_width = unit_width * total_width_ratio
    fig_height = 8

    ncols = num_images
    nrows = 5

    # fig = plt.figure(figsize=(19, 7.8), constrained_layout=True)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)

    gs = GridSpec(
        nrows=nrows, ncols=ncols, figure=fig, height_ratios=[3, 0.5, 0.5, 1, 2]
    )

    axes_colors = ["darkslategray", "olive", "steelblue", "darkred", "slategray"]
    ref_index = 1  # Index of "Ground Truth"
    low_res_index = 0  # Index of "Noisy Low-Resolution"
    ref_img = images[ref_index].squeeze()

    psnr_values = {}
    ssim_values = {}

    for i, (img, title) in enumerate(zip(images, titles)):
        ax_img = fig.add_subplot(gs[0, i])
        if img is not None and img.size > 0:
            ax_img.imshow(img, cmap=cmap, aspect="auto")
        else:
            print(f"Warning: Invalid image data for {title}")
        ax_img.axis("on")
        for spine in ax_img.spines.values():
            spine.set_color(axes_colors[i % len(axes_colors)])

        if i != ref_index and i != low_res_index:
            current_psnr = psnr(ref_img, img)
            current_ssim = ssim(ref_img, img, multichannel=True)
            psnr_values[titles[i]] = current_psnr
            ssim_values[titles[i]] = current_ssim
            title += f"\n$PSNR: {current_psnr:.2f}$ dB"
            title += f"\n$SSIM: {current_ssim:.4f}$"

        ax_img.set_title(
            title, fontsize=12, family="Times New Roman", fontweight="bold"
        )

        freq = fftshift(fft2(img))
        freq_magnitude = np.log(np.abs(freq) + 1)

        ax_x_spectrum = fig.add_subplot(gs[1, i])
        ax_x_spectrum.plot(np.sum(freq_magnitude, axis=0), color="blue")
        ax_x_spectrum.set_title(
            r"$X-\mathrm{Spectrum}$",
            fontsize=12,
            family="Times New Roman",
            fontweight="bold",
        )
        ax_x_spectrum.set_xlabel(
            r"$\mathrm{Frequency\ (pixels)}$", fontsize=11, family="Times New Roman"
        )

        ax_y_spectrum = fig.add_subplot(gs[2, i])
        ax_y_spectrum.plot(np.sum(freq_magnitude, axis=1), color="blue")
        ax_y_spectrum.set_title(
            r"$Y-\mathrm{Spectrum}$",
            fontsize=12,
            family="Times New Roman",
            fontweight="bold",
        )
        ax_y_spectrum.set_xlabel(
            r"$\mathrm{Frequency\ (pixels)}$", fontsize=11, family="Times New Roman"
        )

        ax_2d_spectrum = fig.add_subplot(gs[3, i])
        ax_2d_spectrum.imshow(freq_magnitude, cmap="gray")
        ax_2d_spectrum.set_title(
            r"$2D\ \mathrm{Spectrum}$",
            fontsize=12,
            family="Times New Roman",
            fontweight="bold",
        )
        ax_2d_spectrum.axis("on")

    sorted_psnr = sorted(psnr_values.items(), key=lambda x: x[1], reverse=True)
    sorted_ssim = sorted(ssim_values.items(), key=lambda x: x[1], reverse=True)

    for i, (img, title) in enumerate(zip(images, titles)):
        if i != ref_index and i != low_res_index:
            current_psnr = psnr_values[title]
            current_ssim = ssim_values[title]

            psnr_text = f"$PSNR: {current_psnr:.2f}$"
            ssim_text = f"$SSIM: {current_ssim:.4f}$"

            if title == sorted_psnr[0][0]:
                psnr_text = r"\textbf{" + psnr_text + "}"
            elif title == sorted_psnr[1][0]:
                psnr_text = r"\underline{" + psnr_text + "}"

            if title == sorted_ssim[0][0]:
                ssim_text = r"\textbf{" + ssim_text + "}"
            elif title == sorted_ssim[1][0]:
                ssim_text = r"\underline{" + ssim_text + "}"

            title += f"\n${psnr_text}$ dB\n${ssim_text}$"

    plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=600)
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

    fig = plt.figure(figsize=(16, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
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
