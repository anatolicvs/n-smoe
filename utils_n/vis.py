from typing import Any, Dict, List
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


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

    fig = plt.figure(figsize=(19, 3.5))
    ncols = len(images) + 1
    gs = GridSpec(
        3,
        ncols,
        height_ratios=[2, 2, 0.5],  # Adjusting the height ratios for better space usage
        width_ratios=[2, 2] + [1] * (ncols - 2),
        # width_ratios=[2, 2, 1, 1, 1, 1, 1, 1],
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
    plt.subplots_adjust(
        left=0.129, bottom=0.11, right=0.89, top=0.874, wspace=0, hspace=0
    )

    for ax in fig.get_axes():
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=600)
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

    fig = plt.figure(figsize=(15, 3.41))
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
    fig = plt.figure(figsize=(19, 7.8), constrained_layout=True)
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
