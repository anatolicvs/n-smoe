def gen_latex_table(average_metric_data):
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
    latex_str += (
        " ".join(
            [f"\\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}" for i in range(len(datasets))]
        )
        + "\n"
    )
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
    latex_str += (
        " ".join(
            [
                f"\\cmidrule(lr){{{3 + 4 * i}-{4 + 4 * i}}} \\cmidrule(lr){{{5 + 4 * i}-{6 + 4 * i}}}"
                for i in range(len(datasets))
            ]
        )
        + "\n"
    )
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
