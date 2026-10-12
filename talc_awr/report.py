from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import pair
from .io import load_prediction, read_json, read_manifest, write_csv, write_json

COLORS = ["#4779a8", "#51999e", "#6d986b", "#777c84", "#a08d71", "#807fa6"]


def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def report(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for folder in args.evaluations:
        record = read_json(Path(folder) / "summary.json")
        if "metrics" not in record:
            raise ValueError("Report inputs must be evaluate-matrix directories")
        groups[record.get("experiment", record["variant"])].append((Path(folder), record))
    rows, curves = [], {}
    for variant, records in groups.items():
        seeds = [record["seed"] for _, record in records]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seed for variant {variant}; report experiments separately")
        metrics = {**records[0][1]["metrics"], **records[0][1]["macro_task_mean"]}
        row = {"experiment": variant, "variant": records[0][1]["variant"], "runs": len(records)}
        for key in metrics:
            values = [
                record["metrics"].get(key, record["macro_task_mean"].get(key))
                for _, record in records
            ]
            if any(value is None for value in values):
                raise ValueError(f"Missing {key} for some runs of {variant}")
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_sample_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
        stage_curves = []
        for folder, _ in records:
            matrix_path = folder / "matrix.json"
            if matrix_path.is_file():
                matrix_record = read_json(matrix_path)
                matrix = np.array(
                    [[np.nan if v is None else v for v in row] for row in matrix_record["psnr"]],
                    dtype=float,
                )
                stage_curves.append([float(matrix[i, : i + 1].mean()) for i in range(len(matrix))])
                figure, ax = plt.subplots(figsize=(5.5, 4.2))
                masked = np.ma.masked_invalid(matrix)
                plot = ax.imshow(masked, cmap="Blues")
                ax.set_xticks(range(len(matrix)), matrix_record["tasks"], rotation=35, ha="right")
                ax.set_yticks(range(len(matrix)), range(1, len(matrix) + 1))
                ax.set_ylabel("Training stage")
                for i in range(len(matrix)):
                    for j in range(i + 1):
                        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
                figure.colorbar(plot, ax=ax, label="PSNR (dB)")
                save_figure(
                    figure,
                    output / f"matrix_{variant}_seed{read_json(folder / 'summary.json')['seed']}",
                )
        if stage_curves:
            lengths = {len(curve) for curve in stage_curves}
            if len(lengths) != 1:
                raise ValueError("Do not combine different stage counts in one variant")
            curves[variant] = np.asarray(stage_curves)
    write_csv(output / "comparison.csv", rows)
    write_json(output / "comparison.json", rows)
    if curves:
        figure, ax = plt.subplots(figsize=(7, 4))
        for index, (variant, values) in enumerate(curves.items()):
            x, mean = np.arange(1, values.shape[1] + 1), values.mean(0)
            color = COLORS[index % len(COLORS)]
            ax.plot(x, mean, marker="o", label=variant, color=color)
            if len(values) > 1:
                deviation = values.std(0, ddof=1)
                ax.fill_between(x, mean - deviation, mean + deviation, alpha=0.15, color=color)
        ax.set(xlabel="Training stage", ylabel="Seen-task mean PSNR (dB)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        save_figure(figure, output / "continual_stage_psnr")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in zip(axes, ["AP", "F"], ["Average PSNR (dB)", "Forgetting (dB)"]):
        valid = [row for row in rows if f"{key}_mean" in row]
        ax.bar(
            range(len(valid)),
            [row[f"{key}_mean"] for row in valid],
            color=COLORS[0],
            yerr=[row[f"{key}_sample_sd"] or 0 for row in valid],
            capsize=3,
        )
        ax.set_xticks(
            range(len(valid)), [row["experiment"] for row in valid], rotation=45, ha="right"
        )
        ax.set_ylabel(label)
    save_figure(figure, output / "continual_comparison")


def plot_diagnostics(args):
    folder, output = Path(args.diagnostics), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    record = read_json(folder / "summary.json")
    confusion = np.asarray(record["single_weather"]["confusion_row_normalized"])
    figure, ax = plt.subplots(figsize=(5, 4))
    plot = ax.imshow(confusion, vmin=0, vmax=1, cmap="Blues")
    ax.set(xlabel="Predicted task", ylabel="True task")
    for i in range(len(confusion)):
        for j in range(len(confusion)):
            ax.text(j, i, f"{100 * confusion[i, j]:.1f}", ha="center", va="center")
    figure.colorbar(plot, ax=ax, label="Fraction")
    save_figure(figure, output / "router_confusion")
    figure, ax = plt.subplots(figsize=(7, 4))
    names = ["top1", "top2", "true_pair", "wrong_pair"]
    x = np.arange(len(names))
    for index, (mixture, values) in enumerate(record["mixtures"].items()):
        ax.bar(
            x + (index - 0.5) * 0.35,
            [values[n]["psnr"] for n in names],
            width=0.35,
            color=COLORS[index],
            label=mixture,
        )
    ax.set_xticks(x, names)
    ax.set_ylabel("PSNR (dB)")
    ax.legend()
    save_figure(figure, output / "routing_policies")


def plot_perception(args):
    record = read_json(args.results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    conditions = list(record["delta_condition_miou_pp"])
    figure, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(conditions))
    for index, mode in enumerate(["input", "restored"]):
        ax.bar(
            x + (index - 0.5) * 0.35,
            [100 * record["modes"][mode]["per_condition"][c]["miou"] for c in conditions],
            width=0.35,
            label=mode,
            color=COLORS[index],
        )
    ax.set_xticks(x, conditions)
    ax.set_ylabel("mIoU (%)")
    ax.legend()
    save_figure(figure, output / "perception_miou")
    names = [k for k, v in record["delta_pooled_class_iou_pp"].items() if v is not None]
    figure, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        range(len(names)), [record["delta_pooled_class_iou_pp"][k] for k in names], color=COLORS[1]
    )
    ax.set_xticks(range(len(names)), names, rotation=50, ha="right")
    ax.set_ylabel("Pooled class IoU gain (pp)")
    ax.axhline(0, color=COLORS[3], linewidth=0.7)
    save_figure(figure, output / "perception_class_iou")


def qualitative(args):
    records = read_manifest(args.manifest)[: args.limit]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.names and len(args.names) != len(args.predictions):
        raise ValueError("--names must match the number of --predictions directories")
    names = args.names or [Path(path).name for path in args.predictions]
    for record in records:
        image, target = pair(record)
        panels = [("Input", image)]
        panels.extend(
            (name, load_prediction(folder, record["id"]))
            for name, folder in zip(names, args.predictions)
        )
        if target is not None:
            panels.append(("Target", target))
        if any(panel.shape != image.shape for _, panel in panels):
            raise ValueError("Qualitative images must have matching geometry")
        figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3), squeeze=False)
        for ax, (name, panel) in zip(axes[0], panels):
            ax.imshow(panel.permute(1, 2, 0).numpy())
            ax.set_title(name)
            ax.axis("off")
        save_figure(figure, output / record["id"])


def plot_comparison(args):
    record = read_json(args.comparison)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    defaults = {
        "restoration": ["quality.psnr", "continual.F", "quality.lpips"],
        "segmentation": ["restored.macro_condition_miou", "delta_macro_miou_pp"],
        "detection": [
            "macro_condition_mean.restored.AP_50_95",
            "macro_condition_mean.restored.AP_small",
        ],
        "routing": [
            "single_weather.macro_accuracy",
            "mixtures.mixed.top2.psnr",
            "blur.entropy_auroc",
        ],
        "profile": ["parameters", "active_conv_linear_gflops", "fps"],
    }
    metrics = args.metrics or defaults[record["kind"]]
    supported = [
        key for key in metrics if any(key in values for values in record["groups"].values())
    ]
    if not supported:
        raise ValueError("The comparison has no requested metrics")
    figure, axes = plt.subplots(1, len(supported), figsize=(5 * len(supported), 4.5), squeeze=False)
    for ax, key in zip(axes[0], supported):
        methods = [method for method, values in record["groups"].items() if key in values]
        values = [record["groups"][method][key] for method in methods]
        ax.bar(
            range(len(methods)),
            [v["mean"] for v in values],
            yerr=[v["sample_sd"] or 0 for v in values],
            capsize=3,
            color=COLORS[0],
        )
        ax.set_xticks(range(len(methods)), methods, rotation=40, ha="right")
        ax.set_ylabel(key)
    save_figure(figure, output / f"{record['kind']}_comparison")
    if args.restoration_comparison:
        quality = read_json(args.restoration_comparison)
        if record["kind"] != "profile" or quality["kind"] != "restoration":
            raise ValueError("Tradeoff plots require profile and restoration comparisons")
        names = [
            name
            for name in record["groups"]
            if name in quality["groups"] and "continual.F" in quality["groups"][name]
        ]
        if not names:
            raise ValueError("Tradeoff files have no matching experiment names")
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for index, name in enumerate(names):
            cost, scores = record["groups"][name], quality["groups"][name]
            points = [
                (cost["parameters"]["mean"] / 1e6, scores["continual.F"]["mean"]),
                (cost["fps"]["mean"], scores["quality.psnr"]["mean"]),
            ]
            for ax, (x, y) in zip(axes, points):
                ax.scatter(x, y, color=COLORS[index % len(COLORS)])
                ax.annotate(name, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
        axes[0].set(xlabel="Stored parameters (M)", ylabel="Forgetting (dB)")
        axes[1].set(xlabel="FPS", ylabel="Mean PSNR (dB)")
        save_figure(figure, output / "efficiency_tradeoff")
