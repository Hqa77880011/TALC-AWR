import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_manifest(path):
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError(f"Manifest must contain unique, nonempty records: {path}")
    return records


def write_manifest(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    tasks = config["tasks"]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Tasks must be nonempty and unique")
    training = config["training"]
    if training["steps"] < 1 or training["validation_interval"] < 1:
        raise ValueError("Training steps and validation interval must be positive")
    if (
        min(
            training["batch_size"],
            training["replay_batch_size"],
            training["crop_size"],
            training["replay_size"],
        )
        < 1
    ):
        raise ValueError("Batch and crop sizes must be positive")
    memory = config["memory"]
    if memory["pairs_per_task"] < 0 or (
        memory["total_pairs"] is not None and memory["total_pairs"] < 0
    ):
        raise ValueError("Memory budgets cannot be negative")
    return config


def rgb_tensor(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def save_rgb(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(path)


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)
