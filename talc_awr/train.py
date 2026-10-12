import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import batch, pair
from .io import load_checkpoint, load_config, read_json, read_manifest, seed_everything, write_json
from .losses import distillation_loss, restoration_loss, routing_loss
from .memory import allocations, memory_bytes, replay_batch, update_memory
from .metrics import psnr
from .model import build_model

VARIANTS = [
    "talc",
    "finetune",
    "ewc",
    "lwf",
    "replay",
    "derpp",
    "derpp_capacity",
    "joint",
    "separate",
    "talc_uniform",
    "talc_shared",
    "talc_no_replay",
    "talc_no_orth",
    "talc_top1",
    "talc_unfreeze",
    "talc_no_distill",
    "talc_unfreeze_no_distill",
]


def uses_replay(variant):
    return variant in {"replay", "derpp", "derpp_capacity"} or (
        variant.startswith("talc") and variant != "talc_no_replay"
    )


@torch.inference_mode()
def validate(model, records, device, top_k):
    model.eval()
    task_scores = []
    for task_records in records:
        scores = []
        for record in task_records:
            image, target = pair(record)
            output = model(image.unsqueeze(0).to(device), top_k=top_k)["image"]
            scores.append(psnr(output, target.unsqueeze(0).to(device)).item())
        task_scores.append(float(np.mean(scores)))
    model.train()
    return float(np.mean(task_scores)), task_scores


def fisher_diagonal(model, records, config, rng, device):
    model.eval()
    parameters = dict(model.named_parameters())
    fisher = {name: torch.zeros_like(p, device=device) for name, p in parameters.items()}
    sample_count = config["training"]["fisher_samples"]
    if sample_count < 1:
        raise ValueError("fisher_samples must be positive")
    for _ in range(sample_count):
        image, target, _ = batch(
            records,
            1,
            config["training"]["crop_size"],
            rng,
            device,
            {task: i for i, task in enumerate(config["tasks"])},
        )
        model.zero_grad(set_to_none=True)
        loss = restoration_loss(model(image)["image"], target, config["loss"]["ssim"])
        loss.backward()
        for name, parameter in parameters.items():
            if parameter.grad is not None:
                fisher[name] += parameter.grad.detach().square() / sample_count
    model.zero_grad(set_to_none=True)
    return fisher


def save_stage(
    path,
    model,
    config,
    variant,
    stage,
    seed,
    memory,
    fisher,
    anchor,
    step,
    scores,
    current_rng,
    replay_rng,
):
    checkpoint = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": config,
        "variant": variant,
        "stage": stage,
        "seed": seed,
        "memory": memory,
        "fisher": {k: v.cpu() for k, v in fisher.items()},
        "anchor": {k: v.cpu() for k, v in anchor.items()},
        "selected_step": step,
        "validation_psnr": scores,
        "rng": {
            "current": current_rng.bit_generator.state,
            "replay": replay_rng.bit_generator.state,
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    torch.save(checkpoint, path)


def train(args):
    config = load_config(args.config)
    if args.tasks:
        config["tasks"] = args.tasks.split(",")
        if len(set(config["tasks"])) != len(config["tasks"]):
            raise ValueError("Task order must not contain duplicate tasks")
    for key in ["steps", "batch_size", "validation_interval"]:
        value = getattr(args, key, None)
        if value is not None:
            if value < 1:
                raise ValueError(f"{key} must be positive")
            config["training"][key] = value
    if args.pairs_per_task is not None:
        if args.pairs_per_task < 0:
            raise ValueError("pairs_per_task cannot be negative")
        config["memory"]["pairs_per_task"] = args.pairs_per_task
    if args.total_pairs is not None:
        if args.total_pairs < 0:
            raise ValueError("total_pairs cannot be negative")
        config["memory"]["total_pairs"] = args.total_pairs
    variant = args.variant
    memory_enabled = uses_replay(variant) and (
        (config["memory"]["total_pairs"] or 0) > 0
        if config["memory"]["total_pairs"] is not None
        else config["memory"]["pairs_per_task"] > 0
    )
    if memory_enabled and not args.replay_plan:
        raise ValueError("Replay methods require --replay-plan from select-memory")
    if memory_enabled:
        plan = read_json(args.replay_plan)
        if plan["task_order"] != config["tasks"]:
            raise ValueError("Replay plan must use this run's task order and first-task reference")
        for stage, task in enumerate(config["tasks"], start=1):
            if len(plan["tasks"].get(task, [])) < allocations(config, stage)[-1]:
                raise ValueError(f"Replay plan has insufficient ordered candidates for {task}")
    if args.resume and variant == "joint":
        raise ValueError(
            "Joint training has one completed checkpoint and cannot resume a task stream"
        )
    if args.stage_limit is not None and variant == "joint":
        raise ValueError(
            "--stage-limit applies to sequential methods; joint training sees every task"
        )
    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("stage*.pt")):
        raise FileExistsError("Use a new output directory or resume into a new directory")
    model, memory, fisher, anchor = None, {}, {}, {}
    current_rng = np.random.default_rng(90000 + args.seed)
    replay_rng = np.random.default_rng(91000 + args.seed)
    start = 1
    if args.resume:
        previous = load_checkpoint(args.resume)
        if (
            previous["variant"] != variant
            or previous["config"] != config
            or previous["seed"] != args.seed
        ):
            raise ValueError("Resume variant, configuration and seed must match")
        start = previous["stage"] + 1
        model = build_model(config, variant, previous["stage"]).to(device)
        model.load_state_dict(previous["model"])
        memory = previous["memory"]
        fisher = {k: v.to(device) for k, v in previous["fisher"].items()}
        anchor = {k: v.to(device) for k, v in previous["anchor"].items()}
        current_rng.bit_generator.state = previous["rng"]["current"]
        replay_rng.bit_generator.state = previous["rng"]["replay"]
        torch.set_rng_state(previous["rng"]["torch"])
        if device.type == "cuda" and previous["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(previous["rng"]["cuda"])
    limit = min(args.stage_limit or len(config["tasks"]), len(config["tasks"]))
    if start > limit:
        raise ValueError("No remaining stage to train")
    write_json(output / "config.json", {"config": config, "variant": variant, "seed": args.seed})
    task_indices = {task: i for i, task in enumerate(config["tasks"])}
    training = config["training"]
    loss_config = config["loss"]
    amp = training["amp"] and device.type == "cuda"
    top_k = 1 if variant == "talc_top1" else 2
    stages = [len(config["tasks"])] if variant == "joint" else range(start, limit + 1)
    for stage in stages:
        task = config["tasks"][stage - 1]
        records = read_manifest(Path(args.data) / f"train_{task}.jsonl")
        if variant == "joint":
            records = [
                record
                for task_name in config["tasks"]
                for record in read_manifest(Path(args.data) / f"train_{task_name}.jsonl")
            ]
        teacher = None
        if model is not None and variant not in {"joint", "separate", "finetune", "ewc", "replay"}:
            teacher = copy.deepcopy(model).eval().requires_grad_(False)
        if model is None or variant == "separate":
            model = build_model(config, variant, stage).to(device)
        else:
            model.add_task()
            model.to(device)
        model.configure_trainable(
            unfreeze_old=variant in {"talc_unfreeze", "talc_unfreeze_no_distill"}
        )
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=training["lr"],
            weight_decay=training["weight_decay"],
        )
        steps = training["steps"] * (len(config["tasks"]) if variant == "joint" else 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, steps, eta_min=training["min_lr"]
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        validation_tasks = [task] if variant == "separate" else config["tasks"][:stage]
        validation_records = [
            read_manifest(Path(args.data) / f"val_{t}.jsonl") for t in validation_tasks
        ]
        best_score, best_step, best_scores = -math.inf, 0, []
        best_path = output / f"stage{stage}.pt"
        with (output / f"stage{stage}_loss.jsonl").open("w", encoding="utf-8") as log:
            for step in range(1, steps + 1):
                image, target, labels = batch(
                    records,
                    training["batch_size"],
                    training["crop_size"],
                    current_rng,
                    device,
                    task_indices,
                )
                replay_data = (
                    replay_batch(
                        memory, training["replay_batch_size"], replay_rng, device, task_indices
                    )
                    if memory_enabled
                    else None
                )
                orth_enabled = variant.startswith("talc") and variant not in {
                    "talc_no_orth",
                    "talc_shared",
                }
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    current = model(image, top_k=top_k, orth=orth_enabled)
                    loss = restoration_loss(current["image"], target, loss_config["ssim"])
                    replay = None
                    replay_labels = None
                    terms = {"current": loss.detach().float().item()}
                    if replay_data is not None:
                        replay_x, replay_y, replay_labels = replay_data
                        replay = model(replay_x, top_k=top_k)
                        replay_loss = restoration_loss(
                            replay["image"], replay_y, loss_config["ssim"]
                        )
                        replay_weight = (
                            0.5 if variant in {"derpp", "derpp_capacity"} else loss_config["replay"]
                        )
                        loss = loss + replay_weight * replay_loss
                        terms["replay"] = replay_loss.detach().float().item()
                        if teacher is not None and variant not in {
                            "talc_no_distill",
                            "talc_unfreeze_no_distill",
                        }:
                            with torch.no_grad():
                                old = teacher(replay_x, top_k=top_k)
                            if variant in {"derpp", "derpp_capacity"}:
                                distillation = F.l1_loss(
                                    replay["image"].float(), old["image"].float()
                                )
                            else:
                                distillation = distillation_loss(
                                    replay, old, loss_config["feature"]
                                )
                            distillation_weight = (
                                0.5
                                if variant in {"derpp", "derpp_capacity"}
                                else loss_config["distillation"]
                            )
                            loss = loss + distillation_weight * distillation
                            terms["distillation"] = distillation.detach().float().item()
                    if variant == "lwf" and teacher is not None:
                        with torch.no_grad():
                            old = teacher(image)
                        loss = loss + F.l1_loss(current["image"].float(), old["image"].float())
                    if variant.startswith("talc"):
                        route = routing_loss(current, replay, labels, replay_labels)
                        loss = loss + loss_config["route"] * route
                        loss = loss + loss_config["orthogonality"] * current["orthogonality"]
                        terms["route"] = route.detach().float().item()
                        terms["orthogonality"] = current["orthogonality"].detach().float().item()
                    if variant == "ewc" and fisher:
                        penalty = sum(
                            (fisher[name] * (parameter - anchor[name]).square()).sum()
                            for name, parameter in model.named_parameters()
                        )
                        loss = loss + training["ewc_weight"] * penalty / 2
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at stage {stage}, step {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                if step == 1 or step % 100 == 0:
                    log.write(
                        json.dumps({"step": step, "loss": loss.detach().float().item(), **terms})
                        + "\n"
                    )
                    log.flush()
                if step % training["validation_interval"] == 0 or step == steps:
                    score, scores = validate(model, validation_records, device, top_k)
                    print(
                        f"{variant} seed={args.seed} stage={stage} step={step} val_psnr={score:.4f}",
                        flush=True,
                    )
                    if score > best_score:
                        best_score, best_step, best_scores = score, step, scores
                        save_stage(
                            best_path,
                            model,
                            config,
                            variant,
                            stage,
                            args.seed,
                            memory,
                            fisher,
                            anchor,
                            step,
                            scores,
                            current_rng,
                            replay_rng,
                        )
        model.load_state_dict(load_checkpoint(best_path)["model"])
        if variant == "ewc":
            new_fisher = fisher_diagonal(
                model, records, config, np.random.default_rng([args.seed, stage, 991]), device
            )
            fisher = {
                name: value + training["fisher_decay"] * fisher.get(name, 0)
                for name, value in new_fisher.items()
            }
            anchor = {name: p.detach().clone() for name, p in model.named_parameters()}
        if memory_enabled:
            memory = update_memory(memory, records, config, stage, args.replay_plan)
        save_stage(
            best_path,
            model,
            config,
            variant,
            stage,
            args.seed,
            memory,
            fisher,
            anchor,
            best_step,
            best_scores,
            current_rng,
            replay_rng,
        )
        write_json(
            output / f"stage{stage}_summary.json",
            {
                "stage": stage,
                "task": task,
                "selected_step": best_step,
                "validation_psnr": best_scores,
                "paired_memory_bytes": memory_bytes(memory),
                "paired_memory_mib": memory_bytes(memory) / 2**20,
                "parameters": sum(p.numel() for p in model.parameters()),
            },
        )
