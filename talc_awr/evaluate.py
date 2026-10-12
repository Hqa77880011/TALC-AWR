import json
from pathlib import Path

import numpy as np
import torch

from .data import pair
from .io import (
    load_checkpoint,
    load_prediction,
    read_manifest,
    read_manifests,
    save_prediction,
    write_csv,
    write_json,
)
from .memory import memory_bytes
from .metrics import RestorationMetrics, auroc, continual_metrics, entropy, router_metrics
from .model import build_model


def loaded_model(path, device):
    checkpoint = load_checkpoint(path)
    model = build_model(checkpoint["config"], checkpoint["variant"], checkpoint["stage"])
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device).requires_grad_(False)
    return model, checkpoint


def policy_arguments(policy, record, tasks):
    top_k = 1 if policy == "top1" else 2
    forced = None
    if policy in {"true_pair", "wrong_pair"}:
        if len(tasks) != 4 or len(record.get("components", [])) != 2:
            raise ValueError(
                "Forced pair policies require a four-task model and two-component mixtures"
            )
        true_ids = [tasks.index(component) for component in record["components"]]
        selected = true_ids if policy == "true_pair" else [i for i in range(4) if i not in true_ids]
        forced = torch.tensor(selected)
    return top_k, forced


def summarize_rows(rows):
    tasks = sorted({r["task"] for r in rows})
    measures = sorted(
        {
            key
            for row in rows
            for key in row
            if key
            in {
                "psnr",
                "ssim",
                "lpips",
                "input_psnr",
                "input_ssim",
                "input_lpips",
                "delta_psnr",
                "delta_ssim",
                "delta_lpips",
                "top2_both_recalled",
            }
        }
    )
    grouped = {
        task: {
            measure: float(
                np.mean([r[measure] for r in rows if r["task"] == task and measure in r])
            )
            for measure in measures
            if any(r["task"] == task and measure in r for r in rows)
        }
        for task in tasks
    }
    return {
        "per_task": grouped,
        "macro_task_mean": {
            key: float(np.mean([grouped[t][key] for t in tasks if key in grouped[t]]))
            for key in measures
        },
    }


def checkpoint_identity(checkpoint):
    return {
        "variant": checkpoint["variant"],
        "seed": checkpoint["seed"],
        "stage": checkpoint["stage"],
        "tasks": checkpoint["config"]["tasks"][: checkpoint["stage"]],
        "paired_memory_mib": memory_bytes(checkpoint["memory"]) / 2**20,
    }


def final_router_summary(rows, checkpoint):
    if not checkpoint["variant"].startswith("talc"):
        return {}
    tasks = checkpoint["config"]["tasks"][: checkpoint["stage"]]
    if checkpoint["variant"] == "talc_uniform":
        return {"routing_policy": "uniform_all_experts", "router_chance_accuracy": 1 / len(tasks)}
    selected = [row for row in rows if row["task"] in tasks]
    if not selected:
        return {}
    return {
        "router": router_metrics(
            [[row[f"p_{task}"] for task in tasks] for row in selected],
            [tasks.index(row["task"]) for row in selected],
        )
    }


@torch.inference_mode()
def score_records(model, checkpoint, records, metrics, device, policy="top2", save_directory=None):
    rows = []
    tasks = checkpoint["config"]["tasks"][: checkpoint["stage"]]
    for record in records:
        image, target = pair(record)
        if target is None:
            raise ValueError("Full-reference metrics require aligned clean targets")
        image, target = image.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
        top_k, forced = policy_arguments(policy, record, tasks)
        result = model(image, top_k=top_k, forced=forced)
        row = {
            "id": record["id"],
            "task": record["task"],
            "scene": record.get("scene", record["id"]),
            "degradation": json.dumps(record.get("degradations", []), sort_keys=True),
        }
        restored = metrics(result["image"], target)
        degraded = metrics(image, target)
        row.update(restored)
        row.update({f"input_{key}": value for key, value in degraded.items()})
        row.update({f"delta_{key}": value - degraded[key] for key, value in restored.items()})
        p = result["probabilities"]
        if p is not None and model.router is not None:
            p = p[0].cpu().numpy()
            row.update({f"p_{task}": float(value) for task, value in zip(tasks, p)})
            row["entropy"] = float(entropy(p[None])[0]) if len(p) > 1 else 0.0
            if record.get("components"):
                true_ids = {tasks.index(component) for component in record["components"]}
                row["top2_both_recalled"] = int(true_ids.issubset(set(np.argsort(p)[-2:])))
        rows.append(row)
        if save_directory is not None:
            save_prediction(save_directory, record["id"], result["image"][0])
    summary = summarize_rows(rows)
    summary.update(final_router_summary(rows, checkpoint))
    return rows, summary


def evaluate(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, checkpoint = loaded_model(args.checkpoint, args.device)
    metrics = RestorationMetrics(args.device, lpips_enabled=not args.no_lpips)
    records = read_manifests(args.manifests)
    policy = args.policy or ("top1" if checkpoint["variant"] == "talc_top1" else "top2")
    rows, summary = score_records(
        model,
        checkpoint,
        records,
        metrics,
        args.device,
        policy,
        output / "images" if args.save_images else None,
    )
    write_csv(output / "per_image.csv", rows)
    write_json(
        output / "summary.json",
        {
            "checkpoint": str(args.checkpoint),
            **checkpoint_identity(checkpoint),
            "policy": policy,
            **summary,
        },
    )
    print(summary["macro_task_mean"])


def evaluate_matrix(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.run).glob("stage*.pt"), key=lambda p: int(p.stem[5:]))
    if not paths:
        raise FileNotFoundError("No stage checkpoints in --run")
    final = load_checkpoint(paths[-1])
    tasks = final["config"]["tasks"][: final["stage"]]
    metrics = RestorationMetrics(args.device, lpips_enabled=not args.no_lpips)
    if final["variant"] == "joint":
        model, checkpoint = loaded_model(paths[-1], args.device)
        records = [
            r for task in tasks for r in read_manifest(Path(args.data) / f"test_{task}.jsonl")
        ]
        rows, summary = score_records(model, checkpoint, records, metrics, args.device)
        write_csv(output / "per_image.csv", rows)
        write_json(
            output / "summary.json",
            {
                "variant": "joint",
                "seed": final["seed"],
                "tasks": tasks,
                "experiment": args.name or "joint",
                "metrics": {"AP": summary["macro_task_mean"]["psnr"]},
                **summary,
            },
        )
        return
    if [int(p.stem[5:]) for p in paths] != list(range(1, len(tasks) + 1)):
        raise ValueError("Matrix evaluation needs every stage checkpoint from 1 through T")
    matrix = np.full((len(tasks), len(tasks)), np.nan)
    final_rows = []
    for stage, path in enumerate(paths, start=1):
        model, checkpoint = loaded_model(path, args.device)
        if (
            checkpoint["variant"] != final["variant"]
            or checkpoint["seed"] != final["seed"]
            or checkpoint["config"] != final["config"]
            or checkpoint["stage"] != stage
        ):
            raise ValueError(
                "Stage checkpoints must belong to the same method, seed and configuration"
            )
        rows = []
        for task_index, task in enumerate(tasks[:stage]):
            if final["variant"] == "separate":
                model, checkpoint = loaded_model(paths[task_index], args.device)
            task_rows, summary = score_records(
                model,
                checkpoint,
                read_manifest(Path(args.data) / f"test_{task}.jsonl"),
                metrics,
                args.device,
                "top1" if final["variant"] == "talc_top1" else "top2",
            )
            matrix[stage - 1, task_index] = summary["macro_task_mean"]["psnr"]
            rows.extend(task_rows)
        write_csv(
            output / ("per_image.csv" if stage == len(tasks) else f"stage{stage}_per_image.csv"),
            rows,
        )
        if stage == len(tasks):
            final_rows = rows
    serial_matrix = [[float(v) if np.isfinite(v) else None for v in row] for row in matrix]
    write_json(output / "matrix.json", {"tasks": tasks, "psnr": serial_matrix})
    summary = summarize_rows(final_rows)
    summary.update(final_router_summary(final_rows, final))
    write_json(
        output / "summary.json",
        {
            **checkpoint_identity(final),
            "experiment": args.name or final["variant"],
            "metrics": continual_metrics(matrix),
            **summary,
        },
    )
    print(continual_metrics(matrix))


def diagnostics(args):
    model, checkpoint = loaded_model(args.checkpoint, args.device)
    tasks = checkpoint["config"]["tasks"][: checkpoint["stage"]]
    if model.kind != "talc" or len(tasks) != 4 or model.router is None:
        raise ValueError("Routing diagnostics require a learned four-task TALC router")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = RestorationMetrics(args.device, lpips_enabled=not args.no_lpips)
    known = [r for task in tasks for r in read_manifest(Path(args.data) / f"test_{task}.jsonl")]
    known_rows, known_summary = score_records(model, checkpoint, known, metrics, args.device)
    write_csv(output / "known_per_image.csv", known_rows)
    report = {
        **checkpoint_identity(checkpoint),
        "single_weather": known_summary["router"],
        "mixtures": {},
    }
    for mixture in ["mixed", "mixed_shift"]:
        records = read_manifest(Path(args.data) / f"test_{mixture}.jsonl")
        report["mixtures"][mixture] = {}
        for policy in ["top1", "top2", "true_pair", "wrong_pair"]:
            rows, summary = score_records(model, checkpoint, records, metrics, args.device, policy)
            write_csv(output / f"{mixture}_{policy}_per_image.csv", rows)
            report["mixtures"][mixture][policy] = {
                **summary["macro_task_mean"],
                "per_task": summary["per_task"],
            }
        reference = report["mixtures"][mixture]["true_pair"]["psnr"]
        for item in report["mixtures"][mixture].values():
            item["true_pair_psnr_gap"] = reference - item["psnr"]
    blur_rows, blur_summary = score_records(
        model, checkpoint, read_manifest(Path(args.data) / "test_blur.jsonl"), metrics, args.device
    )
    write_csv(output / "blur_per_image.csv", blur_rows)
    balanced = []
    for task in tasks:
        task_rows = sorted([r for r in known_rows if r["task"] == task], key=lambda r: r["id"])
        if len(task_rows) < args.known_per_task:
            raise ValueError("Not enough known-weather images for the requested balanced control")
        balanced.extend(task_rows[: args.known_per_task])
    report["blur"] = {
        **blur_summary["macro_task_mean"],
        "entropy_auroc": auroc([r["entropy"] for r in blur_rows], [r["entropy"] for r in balanced]),
    }
    write_json(output / "summary.json", report)


@torch.inference_mode()
def infer(args):
    model, checkpoint = loaded_model(args.checkpoint, args.device)
    policy = args.policy or ("top1" if checkpoint["variant"] == "talc_top1" else "top2")
    output = Path(args.output)
    if args.manifest:
        records = read_manifest(args.manifest)
    else:
        source = Path(args.input)
        files = (
            [source]
            if source.is_file()
            else sorted(
                p
                for p in source.rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )
        )
        if not files:
            raise FileNotFoundError("No input images")
        records = [
            {
                "id": p.stem
                if source.is_file()
                else str(p.relative_to(source).with_suffix(""))
                .replace("\\", "__")
                .replace("/", "__"),
                "input": str(p),
            }
            for p in files
        ]
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("Input filenames map to duplicate output IDs; use a manifest")
    for record in records:
        image, _ = pair(record)
        result = model(image.unsqueeze(0).to(args.device), top_k=1 if policy == "top1" else 2)
        save_prediction(output, record["id"], result["image"][0], args.format)


@torch.inference_mode()
def score_predictions(args):
    manifests = args.manifests or [args.manifest]
    records = read_manifests(manifests)
    metrics = RestorationMetrics(args.device, lpips_enabled=not args.no_lpips)
    rows = []
    for record in records:
        image, target = pair(record)
        if target is None:
            raise ValueError("Prediction scoring requires paired targets")
        prediction = load_prediction(args.predictions, record["id"])
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction dimensions differ from target: {record['id']}")
        target = target.unsqueeze(0).to(args.device)
        restored = metrics(prediction.unsqueeze(0).to(args.device), target)
        degraded = metrics(image.unsqueeze(0).to(args.device), target)
        rows.append(
            {
                "id": record["id"],
                "task": record["task"],
                "scene": record.get("scene", record["id"]),
                "degradation": json.dumps(record.get("degradations", []), sort_keys=True),
                **restored,
                **{f"input_{k}": v for k, v in degraded.items()},
                **{f"delta_{k}": v - degraded[k] for k, v in restored.items()},
            }
        )
    write_csv(Path(args.output) / "per_image.csv", rows)
    write_json(
        Path(args.output) / "summary.json",
        {"method": args.name, "seed": args.seed, "manifests": manifests, **summarize_rows(rows)},
    )
