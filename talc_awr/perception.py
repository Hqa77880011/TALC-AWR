import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import CLASS_NAMES, pair, semantic_labels
from .evaluate import loaded_model
from .io import load_checkpoint, read_json, read_manifest, rgb_tensor, write_json
from .metrics import confusion_iou


def segmentation_model(args):
    if args.segmenter == "segformer":
        from transformers import SegformerForSemanticSegmentation

        model = SegformerForSemanticSegmentation.from_pretrained(args.segmenter_weights)
        if model.config.num_labels != 19:
            raise ValueError("SegFormer checkpoint must have 19 Cityscapes classes")
    else:
        if not args.deeplab_source:
            raise ValueError(
                "DeepLabV3+ requires --deeplab-source pointing to DeepLabV3Plus-Pytorch"
            )
        source = Path(args.deeplab_source).resolve()
        if not (source / "network/modeling.py").is_file():
            raise FileNotFoundError(source / "network/modeling.py")
        sys.path.insert(0, str(source))
        from network import modeling

        model = modeling.deeplabv3plus_resnet50(num_classes=19, output_stride=16)
        checkpoint = load_checkpoint(args.segmenter_weights)
        model.load_state_dict(checkpoint["model_state"])
    return model.eval().to(args.device).requires_grad_(False)


@torch.inference_mode()
def semantic_prediction(model, image, args, label_shape):
    height, width = image.shape[-2:]
    scale = args.short_side / min(height, width)
    new_h, new_w = round(height * scale), round(width * scale)
    resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    normalized = (resized - mean) / std
    normalized = F.pad(normalized, (0, (-new_w) % 32, 0, (-new_h) % 32))
    result = model(normalized)
    logits = result.logits if args.segmenter == "segformer" else result
    logits = F.interpolate(logits, size=normalized.shape[-2:], mode="bilinear", align_corners=False)
    logits = logits[..., :new_h, :new_w]
    logits = F.interpolate(logits, size=label_shape, mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


def restored_image(record, image, restorer, args):
    if restorer is not None:
        return restorer(image)["image"]
    prediction = (
        rgb_tensor(Path(args.predictions) / f"{record['id']}.png").unsqueeze(0).to(args.device)
    )
    if prediction.shape != image.shape:
        raise ValueError(f"Restored and input image dimensions differ: {record['id']}")
    return prediction


@torch.inference_mode()
def segmentation(args):
    records = read_manifest(args.manifest)
    segmenter = segmentation_model(args)
    restorer = loaded_model(args.checkpoint, args.device)[0] if args.checkpoint else None
    counts = {mode: {} for mode in ["input", "restored"]}
    for record in records:
        label = semantic_labels(record)
        condition = record.get("condition", record["task"])
        image, _ = pair(record)
        image = image.unsqueeze(0).to(args.device)
        restored = restored_image(record, image, restorer, args)
        for mode, current in [("input", image), ("restored", restored)]:
            prediction = semantic_prediction(segmenter, current, args, label.shape)
            valid = label != 255
            confusion = np.bincount(
                (19 * label[valid] + prediction[valid]).ravel(), minlength=19**2
            ).reshape(19, 19)
            counts[mode].setdefault(condition, np.zeros((19, 19), dtype=np.int64))
            counts[mode][condition] += confusion
    report = {
        "segmenter": args.segmenter,
        "segmenter_weights": args.segmenter_weights,
        "short_side": args.short_side,
        "classes": CLASS_NAMES,
        "modes": {},
    }
    for mode, conditions in counts.items():
        pooled = sum(conditions.values())
        per_condition = {name: confusion_iou(value) for name, value in conditions.items()}
        report["modes"][mode] = {
            "per_condition": per_condition,
            "pooled_archive": confusion_iou(pooled),
            "macro_condition_miou": float(np.mean([v["miou"] for v in per_condition.values()])),
            "confusion_counts": {k: v.tolist() for k, v in conditions.items()},
        }
    report["delta_macro_miou_pp"] = 100 * (
        report["modes"]["restored"]["macro_condition_miou"]
        - report["modes"]["input"]["macro_condition_miou"]
    )
    report["delta_condition_miou_pp"] = {
        condition: 100
        * (
            report["modes"]["restored"]["per_condition"][condition]["miou"]
            - report["modes"]["input"]["per_condition"][condition]["miou"]
        )
        for condition in counts["input"]
    }
    input_iou = report["modes"]["input"]["pooled_archive"]["class_iou"]
    restored_iou = report["modes"]["restored"]["pooled_archive"]["class_iou"]
    report["delta_pooled_class_iou_pp"] = {
        name: 100 * (b - a) if a is not None and b is not None else None
        for name, a, b in zip(CLASS_NAMES, input_iou, restored_iou)
    }
    write_json(Path(args.output) / "segmentation.json", report)
    print(f"Macro-condition mIoU gain: {report['delta_macro_miou_pp']:.3f} pp")


def detector_model(args):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=9,
        box_score_thresh=args.score_threshold,
        box_nms_thresh=args.nms_threshold,
    )
    state = load_checkpoint(args.detector_weights)
    model.load_state_dict(state["model"])
    return model.eval().to(args.device).requires_grad_(False)


def coco_report(coco, detections, image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if detections:
        predicted = coco.loadRes(detections)
    else:
        predicted = COCO()
        predicted.dataset = {
            "images": coco.dataset["images"],
            "categories": coco.dataset["categories"],
            "annotations": [],
        }
        predicted.createIndex()
    evaluation = COCOeval(coco, predicted, "bbox")
    evaluation.params.imgIds = image_ids
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    result = {
        "AP_50_95": float(evaluation.stats[0]),
        "AP_50": float(evaluation.stats[1]),
        "AP_small": float(evaluation.stats[3]),
    }
    precision = evaluation.eval["precision"]
    for class_id, name in [(1, "person"), (3, "car")]:
        index = list(evaluation.params.catIds).index(class_id)
        values = precision[:, :, index, 0, -1]
        values = values[values >= 0]
        result[f"AP_{name}"] = float(values.mean()) if len(values) else None
    return {key: value if value is None or value >= 0 else None for key, value in result.items()}


@torch.inference_mode()
def detection(args):
    from pycocotools.coco import COCO

    records = read_manifest(args.manifest)
    coco = COCO(args.annotations)
    lookup = {image["file_name"]: image["id"] for image in coco.dataset["images"]}
    if {r["id"] for r in records} != set(lookup):
        raise ValueError("COCO annotations and manifest must contain exactly the same image IDs")
    detector = detector_model(args)
    restorer = loaded_model(args.checkpoint, args.device)[0] if args.checkpoint else None
    predictions = {mode: [] for mode in ["input", "restored"]}
    for record in records:
        image, _ = pair(record)
        image = image.unsqueeze(0).to(args.device)
        restored = restored_image(record, image, restorer, args)
        for mode, tensor in [("input", image), ("restored", restored)]:
            result = detector([tensor[0]])[0]
            for box, label, score in zip(result["boxes"], result["labels"], result["scores"]):
                x0, y0, x1, y1 = box.cpu().tolist()
                predictions[mode].append(
                    {
                        "image_id": lookup[record["id"]],
                        "category_id": int(label),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": float(score),
                    }
                )
    conditions = sorted({r["task"] for r in records})
    report = {
        "detector_weights": args.detector_weights,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "conditions": {},
    }
    for condition in conditions:
        ids = [lookup[r["id"]] for r in records if r["task"] == condition]
        report["conditions"][condition] = {
            mode: coco_report(coco, values, ids) for mode, values in predictions.items()
        }
    for condition in conditions:
        values = report["conditions"][condition]
        values["delta_pp"] = {
            key: 100 * (values["restored"][key] - value)
            if value is not None and values["restored"][key] is not None
            else None
            for key, value in values["input"].items()
        }
    output = Path(args.output)
    for mode, values in predictions.items():
        write_json(output / f"{mode}_detections.json", values)
    write_json(output / "detection.json", report)


def train_detector(args):
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from .io import seed_everything

    seed_everything(args.seed)
    records = read_manifest(args.manifest)
    if any(r.get("degradations") or r.get("input") for r in records):
        records = [{**r, "input": r["target"], "degradations": []} for r in records]
    annotations = read_json(args.annotations)
    identifiers = {image["file_name"]: image["id"] for image in annotations["images"]}
    if {r["id"] for r in records} != set(identifiers):
        raise ValueError("Detector training manifest and annotations do not match")
    unique_scenes = {}
    for record in records:
        unique_scenes.setdefault(record["target"], record)
    records = list(unique_scenes.values())
    boxes_by_image = {identifier: [] for identifier in identifiers.values()}
    for annotation in annotations["annotations"]:
        if not annotation["iscrowd"]:
            boxes_by_image[annotation["image_id"]].append(annotation)
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 9)
    model.to(args.device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        losses = []
        for index in rng.permutation(len(records)):
            record = records[int(index)]
            image = rgb_tensor(record["target"]).to(args.device)
            annotations_for_image = boxes_by_image[identifiers[record["id"]]]
            boxes, labels = [], []
            for annotation in annotations_for_image:
                x, y, width, height = annotation["bbox"]
                boxes.append([x, y, x + width, y + height])
                labels.append(annotation["category_id"])
            target = {
                "boxes": torch.tensor(boxes, device=args.device, dtype=torch.float32).reshape(
                    -1, 4
                ),
                "labels": torch.tensor(labels, device=args.device, dtype=torch.int64),
            }
            optimizer.zero_grad(set_to_none=True)
            loss = sum(model([image], [target]).values())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite detector loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"Detector epoch={epoch + 1} loss={np.mean(losses):.5f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "epochs": args.epochs,
            "seed": args.seed,
            "classes": 8,
        },
        output,
    )
