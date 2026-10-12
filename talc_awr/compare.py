import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import read_json, write_csv, write_json


def scalar_metrics(values, prefix=""):
    result = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(scalar_metrics(value, name))
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            result[name] = float(value)
    return result


def result_metrics(record, kind):
    if kind == "restoration":
        values = {
            "continual": record.get("metrics", {}),
            "quality": record["macro_task_mean"],
            "per_task": record["per_task"],
        }
        if "router" in record:
            values["router"] = {
                key: record["router"][key] for key in ["macro_accuracy", "ece_10_bins"]
            }
        if "router_chance_accuracy" in record:
            values["router_chance_accuracy"] = record["router_chance_accuracy"]
        return scalar_metrics(values)
    if kind == "segmentation":
        values = {
            "delta_macro_miou_pp": record["delta_macro_miou_pp"],
            "delta_condition_miou_pp": record["delta_condition_miou_pp"],
            "delta_pooled_class_iou_pp": record["delta_pooled_class_iou_pp"],
        }
        for mode, data in record["modes"].items():
            values[mode] = {
                "macro_condition_miou": data["macro_condition_miou"],
                "pooled_miou": data["pooled_archive"]["miou"],
                "per_condition": {
                    key: value["miou"] for key, value in data["per_condition"].items()
                },
                "pooled_class_iou": dict(
                    zip(record["classes"], data["pooled_archive"]["class_iou"])
                ),
            }
        return scalar_metrics(values)
    if kind == "detection":
        return scalar_metrics(
            {
                "conditions": record["conditions"],
                "macro_condition_mean": record["macro_condition_mean"],
            }
        )
    if kind == "routing":
        return scalar_metrics(
            {
                "single_weather": {
                    key: record["single_weather"][key] for key in ["macro_accuracy", "ece_10_bins"]
                },
                "mixtures": record["mixtures"],
                "blur": record["blur"],
            }
        )
    keys = [
        "parameters",
        "active_model_parameters",
        "parameter_storage_fp32_mb",
        "paired_memory_mib",
        "active_conv_linear_gflops",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "fps",
    ]
    return {key: float(record[key]) for key in keys}


def result_identity(path, record, kind):
    if kind == "restoration":
        with (path.parent / "per_image.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        return {"images": sorted((row["id"], row["task"], row["degradation"]) for row in rows)}
    if kind == "segmentation":
        return {
            "images": sorted(record["image_ids"]),
            "segmenter": record["segmenter"],
            "segmenter_weights": record["segmenter_weights"],
            "short_side": record["short_side"],
            "precision": record["precision"],
        }
    if kind == "detection":
        return {
            "images": sorted(record["image_ids"]),
            "detector_weights": record["detector_weights"],
            "score_threshold": record["score_threshold"],
            "nms_threshold": record["nms_threshold"],
        }
    if kind == "profile":
        return {
            "device": record["device"],
            "timing_image": record["timing_image"],
            "input_shape": record["input_shape"],
            "precision": record["precision"],
            "device_setup": record["device_setup"],
            "frames_per_repeat": record["frames_per_repeat"],
            "repeats": record["repeats"],
            "flops_convention": record["flops_convention"],
        }
    return {"tasks": record["tasks"]}


def compare(args):
    if args.names is not None and len(args.names) != len(args.results):
        raise ValueError("--names must provide one name for each result file")
    groups, identities = defaultdict(list), {}
    for index, path in enumerate(map(Path, args.results)):
        record = read_json(path)
        default_name = (
            record.get("method")
            or record.get("experiment")
            or record.get("variant")
            or record.get("restoration", {}).get("variant")
            or path.parent.name
        )
        name = args.names[index] if args.names else default_name
        identity = result_identity(path, record, args.kind)
        if name in identities and identities[name] != identity:
            raise ValueError(f"Different data or evaluation settings in group {name}")
        identities[name] = identity
        seed = record.get("seed")
        if any(previous["seed"] == seed for previous in groups[name]):
            raise ValueError(f"Duplicate or unspecified seed in group {name}")
        groups[name].append({"seed": seed, "metrics": result_metrics(record, args.kind)})
    summaries = {}
    for name, runs in groups.items():
        keys = sorted(set.intersection(*(set(run["metrics"]) for run in runs)))
        if not keys:
            raise ValueError(f"No supported metrics in group {name}")
        summaries[name] = {}
        for key in keys:
            values = [run["metrics"][key] for run in runs]
            summaries[name][key] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "runs": len(values),
            }
    gaps = {}
    if args.reference:
        if args.reference not in groups:
            raise ValueError("--reference must identify one result group")
        reference = groups[args.reference]
        for name, runs in groups.items():
            if name == args.reference:
                continue
            if identities[name] != identities[args.reference]:
                raise ValueError(
                    "Reference comparisons require identical data and evaluation settings"
                )
            reference_seeds = {run["seed"] for run in reference}
            method_seeds = {run["seed"] for run in runs}
            paired = reference_seeds == method_seeds
            keys = set(summaries[name]) & set(summaries[args.reference])
            gaps[name] = {}
            for key in sorted(keys):
                if paired:
                    method_runs = {run["seed"]: run for run in runs}
                    values = [
                        run["metrics"][key] - method_runs[run["seed"]]["metrics"][key]
                        for run in reference
                    ]
                    gaps[name][key] = {
                        "reference_minus_method": float(np.mean(values)),
                        "paired_sample_sd": float(np.std(values, ddof=1))
                        if len(values) > 1
                        else None,
                        "paired_runs": len(values),
                    }
                else:
                    gaps[name][key] = {
                        "reference_minus_method": summaries[args.reference][key]["mean"]
                        - summaries[name][key]["mean"],
                        "paired_sample_sd": None,
                        "paired_runs": 0,
                    }
    output = Path(args.output)
    write_json(
        output / "comparison.json",
        {"kind": args.kind, "groups": summaries, "reference": args.reference, "gaps": gaps},
    )
    rows = [
        {"method": method, "metric": key, **value}
        for method, metrics in summaries.items()
        for key, value in metrics.items()
    ]
    write_csv(output / "comparison.csv", rows)
    if gaps:
        write_csv(
            output / "reference_gaps.csv",
            [
                {"method": method, "metric": key, **value}
                for method, metrics in gaps.items()
                for key, value in metrics.items()
            ],
        )
