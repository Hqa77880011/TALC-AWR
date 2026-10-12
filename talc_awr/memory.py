import numpy as np
import torch
from torch.nn import functional as F

from .data import crop_pair, pair
from .io import load_checkpoint, load_config, read_json, read_manifest, write_json
from .model import build_model


def candidate_patch(record, selection, size):
    image, target = crop_pair(
        *pair(record), size, np.random.default_rng(selection["crop_seed"]), flip=False
    )[:2]
    return torch.floor(torch.stack([image, target]).clamp(0, 1) * 255 + 0.5).to(torch.uint8)


@torch.inference_mode()
def select_memory(args):
    config = load_config(args.config)
    if args.tasks:
        config["tasks"] = args.tasks.split(",")
        if len(set(config["tasks"])) != len(config["tasks"]):
            raise ValueError("Selection tasks must be unique")
    checkpoint = load_checkpoint(args.reference)
    if checkpoint["stage"] != 1 or not checkpoint["variant"].startswith("talc"):
        raise ValueError("The shared embedding reference must be a stage-one TALC checkpoint")
    if checkpoint["seed"] != 17 or checkpoint["config"]["tasks"][0] != config["tasks"][0]:
        raise ValueError("Each order needs its own seed-17 first-task TALC reference")
    model = build_model(checkpoint["config"], checkpoint["variant"], 1)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(args.device).requires_grad_(False)
    plan = {
        "reference": str(args.reference),
        "seed": args.seed,
        "patch_size": config["training"]["replay_size"],
        "task_order": config["tasks"],
        "tasks": {},
    }
    for task in config["tasks"]:
        records = read_manifest(f"{args.data}/train_{task}.jsonl")
        candidates, embeddings = [], []
        for record in sorted(records, key=lambda r: (r["scene"], r["variant"])):
            image, target = pair(record)
            for index in range(config["memory"]["candidates_per_image"]):
                seed = (
                    4000073001
                    + 10**7 * record["task_id"]
                    + 100 * record["base_index"]
                    + 10 * record["variant"]
                    + index
                )
                x, _, _ = crop_pair(
                    image, target, plan["patch_size"], np.random.default_rng(seed), flip=False
                )
                x = torch.floor(x.clamp(0, 1) * 255 + 0.5) / 255
                embedding = F.normalize(
                    model.probe(x.unsqueeze(0).to(args.device)).float(), dim=1, eps=1e-6
                )
                embeddings.append(embedding[0].cpu())
                candidates.append(
                    {
                        "id": record["id"],
                        "candidate": index,
                        "crop_seed": seed,
                        "scene": record["scene"],
                        "variant": record["variant"],
                    }
                )
        values = torch.stack(embeddings).double()
        mean, total = values.mean(0), torch.zeros(values.shape[1], dtype=torch.float64)
        used = torch.zeros(len(values), dtype=torch.bool)
        selected = []
        if args.max_pairs > len(values):
            raise ValueError(f"Not enough candidates for {task}: requested {args.max_pairs}")
        for count in range(args.max_pairs):
            distance = ((total + values) / (count + 1) - mean).square().sum(1)
            distance[used] = float("inf")
            index = int(distance.argmin())
            selected.append(candidates[index])
            total += values[index]
            used[index] = True
        plan["tasks"][task] = selected
        print(f"Selected {len(selected)} ordered patches for {task}")
    write_json(args.output, plan)


def allocations(config, stage):
    total = config["memory"]["total_pairs"]
    if total is None:
        return [config["memory"]["pairs_per_task"]] * stage
    total = min(total, config["memory"]["pairs_per_task"] * stage)
    quotient, remainder = divmod(total, stage)
    return [quotient + int(i < remainder) for i in range(stage)]


def update_memory(memory, records, config, stage, plan_path):
    sizes = allocations(config, stage)
    tasks = config["tasks"][:stage]
    updated = {}
    for index, task in enumerate(tasks[:-1]):
        if task in memory:
            item = memory[task]
            updated[task] = {
                "pairs": item["pairs"][: sizes[index]].clone(),
                "ids": item["ids"][: sizes[index]],
            }
    count = sizes[-1]
    if count:
        plan = read_json(plan_path)
        if plan["patch_size"] != config["training"]["replay_size"]:
            raise ValueError("Replay plan and training config have different patch sizes")
        selections = plan["tasks"][tasks[-1]][:count]
        if len(selections) != count:
            raise ValueError("Replay plan has fewer patches than the configured budget")
        lookup = {record["id"]: record for record in records}
        pairs, identifiers = [], []
        for selection in selections:
            pairs.append(candidate_patch(lookup[selection["id"]], selection, plan["patch_size"]))
            identifiers.append(selection)
        updated[tasks[-1]] = {"pairs": torch.stack(pairs), "ids": identifiers}
    return updated


def replay_batch(memory, batch_size, rng, device, task_indices):
    if not memory:
        return None
    tasks = [task for task in memory if len(memory[task]["pairs"])]
    if not tasks:
        return None
    selected, labels = [], []
    for _ in range(batch_size):
        task = tasks[int(rng.integers(len(tasks)))]
        patches = memory[task]["pairs"]
        patch = patches[int(rng.integers(len(patches)))].float() / 255
        if rng.random() < 0.5:
            patch = patch.flip(-1)
        selected.append(patch)
        labels.append(task_indices[task])
    selected = torch.stack(selected).to(device)
    return selected[:, 0], selected[:, 1], torch.tensor(labels, device=device, dtype=torch.long)


def memory_bytes(memory):
    return sum(item["pairs"].numel() * item["pairs"].element_size() for item in memory.values())
