from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage, signal

from .io import read_json, read_manifest, rgb_tensor, save_rgb, write_json, write_manifest

WEATHERS = ["rain", "fog", "snow", "low_light"]
TRAIN_IDS = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def weather_parameters(weather, rng, high=False):
    if weather == "rain":
        return {
            "weather": weather,
            "length": 30 if high else int(rng.choice([15, 25, 35])),
            "angle": int(rng.choice([-15, 0, 15])),
            "opacity": 0.32 if high else float(rng.choice([0.12, 0.20, 0.28])),
        }
    if weather == "fog":
        return {
            "weather": weather,
            "beta": 0.028 if high else float(rng.choice([0.008, 0.016, 0.024])),
            "airlight": 0.9 if high else float(rng.choice([0.75, 0.85, 0.95])),
        }
    if weather == "snow":
        return {
            "weather": weather,
            "particles": 1400 if high else int(rng.choice([400, 800, 1200])),
            "radius": int(rng.choice([1, 2, 3])),
            "opacity": 0.75 if high else float(rng.choice([0.3, 0.5, 0.7])),
        }
    if weather == "low_light":
        return {
            "weather": weather,
            "exposure": 0.35 if high else float(rng.choice([0.4, 0.55, 0.7])),
            "gamma": 2.8 if high else float(rng.choice([1.8, 2.2, 2.6])),
            "sigma": 0.018 if high else float(rng.choice([0.005, 0.01, 0.015])),
        }
    raise ValueError(f"Unknown weather: {weather}")


def streaks(height, width, length, angle, rng):
    size = 2 * length + 1
    kernel = Image.new("F", (size, size), 0)
    draw = ImageDraw.Draw(kernel)
    radians = np.deg2rad(angle)
    dx, dy = np.sin(radians) * length / 2, np.cos(radians) * length / 2
    draw.line((length - dx, length - dy, length + dx, length + dy), fill=1, width=1)
    kernel = np.asarray(kernel, dtype=np.float32).copy()
    kernel /= max(kernel.max(), 1)
    layer = np.zeros((height, width), dtype=np.float32)
    for scale, density in [(1, 0.0016), (2, 0.0012), (4, 0.0008)]:
        h, w = max(1, height // scale), max(1, width // scale)
        impulses = (rng.random((h, w)) < density).astype(np.float32)
        band = signal.fftconvolve(impulses, kernel, mode="same")
        band = np.asarray(
            Image.fromarray(band.astype(np.float32)).resize(
                (width, height), Image.Resampling.BILINEAR
            )
        )
        layer += band / scale
    return np.clip(layer, 0, 1)


def degrade(image, parameters, rng, depth=None):
    h, w = image.shape[:2]
    weather = parameters["weather"]
    if weather == "rain":
        alpha = parameters["opacity"] * streaks(
            h, w, parameters["length"], parameters["angle"], rng
        )
        result = image * (1 - alpha[..., None]) + alpha[..., None]
    elif weather == "fog":
        if depth is None or depth.shape != (h, w):
            raise ValueError("Fog requires a depth map aligned with the clean image")
        transmission = np.exp(-parameters["beta"] * depth)[..., None]
        result = image * transmission + parameters["airlight"] * (1 - transmission)
    elif weather == "snow":
        layer = Image.new("F", (w, h), 0)
        draw = ImageDraw.Draw(layer)
        radius = parameters["radius"]
        for x, y in zip(
            rng.integers(w, size=parameters["particles"]),
            rng.integers(h, size=parameters["particles"]),
        ):
            draw.ellipse(
                (int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius), fill=1
            )
        mask = ndimage.gaussian_filter(np.asarray(layer), sigma=max(0.5, radius / 2))
        mask += 0.2 * streaks(h, w, 12, 0, rng)
        alpha = np.clip(mask, 0, 1) * parameters["opacity"]
        result = image * (1 - alpha[..., None]) + alpha[..., None]
    elif weather == "low_light":
        signal_level = (image * parameters["exposure"]) ** parameters["gamma"]
        result = rng.poisson(signal_level * 255).astype(np.float32) / 255
        result += rng.normal(0, parameters["sigma"], image.shape).astype(np.float32)
    elif weather == "blur":
        result = ndimage.gaussian_filter(image, sigma=(parameters["sigma"], parameters["sigma"], 0))
    else:
        raise ValueError(f"Unsupported degradation: {weather}")
    return np.clip(result, 0, 1).astype(np.float32)


def pair(record):
    target = rgb_tensor(record["target"]) if record.get("target") else None
    if record.get("input"):
        image = rgb_tensor(record["input"])
    else:
        if target is None:
            raise ValueError("Synthetic records require a clean target")
        array = target.permute(1, 2, 0).numpy()
        depth = np.load(record["depth"]) if record.get("depth") else None
        rng = np.random.default_rng(record["seed"])
        for parameters in record["degradations"]:
            array = degrade(array, parameters, rng, depth)
        image = torch.from_numpy(array.copy()).permute(2, 0, 1)
    if target is not None and image.shape != target.shape:
        raise ValueError(f"Unaligned pair: {record['id']}")
    return image, target


def crop_pair(image, target, size, rng, flip=True):
    if target is None:
        raise ValueError("Training requires paired clean targets")
    h, w = image.shape[-2:]
    if min(h, w) < size:
        padding = (0, max(0, size - w), 0, max(0, size - h))
        image = torch.nn.functional.pad(image, padding, mode="replicate")
        target = torch.nn.functional.pad(target, padding, mode="replicate")
        h, w = image.shape[-2:]
    y, x = int(rng.integers(h - size + 1)), int(rng.integers(w - size + 1))
    image, target = image[:, y : y + size, x : x + size], target[:, y : y + size, x : x + size]
    if flip and rng.random() < 0.5:
        image, target = image.flip(-1), target.flip(-1)
    return image, target, [y, x]


def batch(records, batch_size, crop_size, rng, device, task_indices):
    images, targets, labels = [], [], []
    for index in rng.integers(len(records), size=batch_size):
        record = records[int(index)]
        image, target = crop_pair(*pair(record), crop_size, rng)[:2]
        images.append(image)
        targets.append(target)
        labels.append(task_indices[record["task"]])
    return (
        torch.stack(images).to(device),
        torch.stack(targets).to(device),
        torch.tensor(labels, device=device, dtype=torch.long),
    )


def depth_from_disparity(disparity_path, camera_path, semantic_path):
    raw = np.asarray(Image.open(disparity_path), dtype=np.float32)
    camera = read_json(camera_path)
    disparity = (raw - 1) / 256
    valid = raw > 1
    depth = np.zeros_like(raw)
    depth[valid] = np.clip(
        camera["intrinsic"]["fx"] * camera["extrinsic"]["baseline"] / disparity[valid], 0.1, 200
    )
    if not valid.any():
        raise ValueError(f"No valid disparity: {disparity_path}")
    fallback = float(np.median(depth[valid]))
    labels = np.asarray(Image.open(semantic_path))
    for semantic_id in np.unique(labels):
        components, _ = ndimage.label(labels == semantic_id)
        boxes = ndimage.find_objects(components)
        for component, box in enumerate(boxes, start=1):
            if box is None:
                continue
            mask = components[box] == component
            observed = mask & valid[box]
            median = float(np.median(depth[box][observed])) if observed.any() else fallback
            region = depth[box]
            region[mask & ~valid[box]] = median
    return depth.astype(np.float32)


def stratified_split(paths, validation_size, rng):
    if not 0 < validation_size < len(paths):
        raise ValueError("Validation size must be smaller than the official training partition")
    cities = sorted({p.parent.name for p in paths})
    groups = [[p for p in paths if p.parent.name == city] for city in cities]
    exact = np.array([len(g) for g in groups]) * validation_size / len(paths)
    counts = np.floor(exact).astype(int)
    order = sorted(range(len(cities)), key=lambda i: (-(exact[i] - counts[i]), cities[i]))
    for index in order[: validation_size - counts.sum()]:
        counts[index] += 1
    validation = set()
    for group, count in zip(groups, counts):
        indices = rng.permutation(len(group))[:count]
        validation.update(group[int(i)] for i in indices)
    return [p for p in paths if p not in validation], sorted(validation)


def prepare_cityscapes(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    train = sorted((root / "leftImg8bit/train").glob("*/*_leftImg8bit.png"))
    test = sorted((root / "leftImg8bit/val").glob("*/*_leftImg8bit.png"))
    if not train or not test:
        raise FileNotFoundError("Expected leftImg8bit/train and leftImg8bit/val below --root")
    rng = np.random.default_rng(args.seed)
    train, validation = stratified_split(train, args.validation_size, rng)
    splits = {"train": train, "val": validation, "test": test}
    output.mkdir(parents=True, exist_ok=True)
    base_records = {}
    for split, paths in splits.items():
        entries = []
        for path in paths:
            stem = path.name.removesuffix("_leftImg8bit.png")
            official_split = path.parent.parent.name
            suffix = Path(official_split) / path.parent.name
            semantic = root / "gtFine" / suffix / f"{stem}_gtFine_labelIds.png"
            instance = root / "gtFine" / suffix / f"{stem}_gtFine_instanceIds.png"
            disparity = root / "disparity" / suffix / f"{stem}_disparity.png"
            camera = root / "camera" / suffix / f"{stem}_camera.json"
            for required in [semantic, instance, disparity, camera]:
                if not required.is_file():
                    raise FileNotFoundError(required)
            depth_path = output / "depth" / f"{stem}.npy"
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(depth_path, depth_from_disparity(disparity, camera, semantic))
            entries.append(
                {
                    "id": stem,
                    "target": str(path),
                    "depth": str(depth_path),
                    "mask": str(semantic),
                    "label_encoding": "cityscapes",
                    "instances": str(instance),
                }
            )
        base_records[split] = entries
    tasks = WEATHERS + [f"{w}_high" for w in WEATHERS]
    for split, bases in base_records.items():
        for task in tasks:
            records = []
            for base in bases:
                for variant in range(args.variants if split == "train" else 1):
                    seed = int(rng.integers(0, 2**63))
                    weather = task.removesuffix("_high")
                    parameters = weather_parameters(
                        weather, np.random.default_rng(seed), task.endswith("_high")
                    )
                    records.append(
                        {
                            **base,
                            "id": f"{base['id']}_{task}_{variant}",
                            "task": task,
                            "seed": seed,
                            "degradations": [parameters],
                        }
                    )
            write_manifest(output / f"{split}_{task}.jsonl", records)
    for name, high in [("mixed", False), ("mixed_shift", True), ("blur", False)]:
        records = []
        for index, base in enumerate(base_records["test"]):
            components = ["rain", "fog"] if index < len(test) // 2 else ["snow", "low_light"]
            seed = int(rng.integers(0, 2**63))
            local_rng = np.random.default_rng(seed)
            parameters = (
                [{"weather": "blur", "sigma": 1.5}]
                if name == "blur"
                else [weather_parameters(w, local_rng, high) for w in components]
            )
            records.append(
                {
                    **base,
                    "id": f"{base['id']}_{name}",
                    "task": name,
                    "components": [] if name == "blur" else components,
                    "seed": seed,
                    "degradations": parameters,
                }
            )
        write_manifest(output / f"test_{name}.jsonl", records)
    write_json(output / "splits.json", {s: [r["id"] for r in rs] for s, rs in base_records.items()})
    print(
        f"Prepared {len(train)}/{len(validation)}/{len(test)} clean train/val/test scenes in {output}"
    )


def prepare_pairs(args):
    input_root, target_root = Path(args.inputs).resolve(), Path(args.targets).resolve()
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def indexed(root, suffix):
        result = {}
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in extensions:
                relative = path.relative_to(root)
                stem = relative.stem.removesuffix(suffix) if suffix else relative.stem
                key = str(relative.parent / stem).replace("\\", "/")
                if key in result:
                    raise ValueError(f"Duplicate pair key {key} in {root}")
                result[key] = path
        return result

    inputs, targets = (
        indexed(input_root, args.input_suffix),
        indexed(target_root, args.target_suffix),
    )
    if not inputs or inputs.keys() != targets.keys():
        raise ValueError(
            "Input and target trees must have matching relative stems after suffix removal"
        )
    records = [
        {
            "id": key.replace("/", "__"),
            "input": str(inputs[key]),
            "target": str(targets[key]),
            "task": args.task,
            "scene": key,
        }
        for key in sorted(inputs)
    ]
    write_manifest(args.output, records)
    print(f"Prepared {len(records)} paired images")


def prepare_acdc(args):
    root = Path(args.root).resolve()
    records = []
    for condition in ["rain", "fog", "snow", "night"]:
        for split in ["train", "val"]:
            folder = root / "rgb_anon" / condition / split
            for path in sorted(folder.rglob("*_rgb_anon.png")):
                relative = path.relative_to(folder)
                mask = (
                    root
                    / "gt"
                    / condition
                    / split
                    / relative.parent
                    / relative.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
                )
                if not mask.is_file():
                    raise FileNotFoundError(mask)
                records.append(
                    {
                        "id": f"{condition}_{path.stem}",
                        "input": str(path),
                        "task": condition,
                        "condition": condition,
                        "mask": str(mask),
                        "label_encoding": "train_ids",
                    }
                )
    if not records:
        raise FileNotFoundError("No ACDC labelled train/val RGB images found")
    write_manifest(args.output, records)
    print(f"Prepared {len(records)} ACDC frames")


def semantic_labels(record):
    labels = np.asarray(Image.open(record["mask"]), dtype=np.int64)
    if record["label_encoding"] == "cityscapes":
        mapped = np.full_like(labels, 255)
        for train_id, label_id in enumerate(TRAIN_IDS):
            mapped[labels == label_id] = train_id
        labels = mapped
    if np.any((labels != 255) & ((labels < 0) | (labels >= 19))):
        raise ValueError(f"Semantic mask has invalid train IDs: {record['mask']}")
    return labels


def prepare_detection(args):
    records = read_manifest(args.manifest)
    instance_classes = [24, 25, 26, 27, 28, 31, 32, 33]
    categories = [
        {"id": i + 1, "name": CLASS_NAMES[TRAIN_IDS.index(c)]}
        for i, c in enumerate(instance_classes)
    ]
    images, annotations = [], []
    for image_id, record in enumerate(records, start=1):
        ids = np.asarray(Image.open(record["instances"]), dtype=np.int64)
        height, width = ids.shape
        images.append({"id": image_id, "file_name": record["id"], "height": height, "width": width})
        for instance_id in np.unique(ids):
            class_id = instance_id // 1000 if instance_id >= 1000 else instance_id
            if class_id not in instance_classes:
                continue
            y, x = np.nonzero(ids == instance_id)
            x0, y0, x1, y1 = int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": instance_classes.index(class_id) + 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": int(len(x)),
                    "iscrowd": int(instance_id < 1000),
                }
            )
    write_json(
        args.output,
        {"info": {}, "images": images, "annotations": annotations, "categories": categories},
    )
    print(f"Converted {len(annotations)} instance boxes")


def export_data(args):
    records = read_manifest(args.manifest)
    output = Path(args.output)
    for record in records:
        image, target = pair(record)
        save_rgb(output / "inputs" / f"{record['id']}.png", image)
        if target is not None:
            save_rgb(output / "targets" / f"{record['id']}.png", target)
