import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import CLASS_NAMES, pair, semantic_labels
from .evaluate import checkpoint_identity, loaded_model
from .io import (
    load_checkpoint,
    load_prediction,
    read_json,
    read_manifest,
    read_manifests,
    rgb_tensor,
    write_json,
)
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

        model = modeling.deeplabv3plus_resnet50(
            num_classes=19, output_stride=16, pretrained_backbone=False
        )
        checkpoint = load_checkpoint(args.segmenter_weights)
        model.load_state_dict(checkpoint["model_state"])
    return model.eval().to(args.device).requires_grad_(False)


@torch.inference_mode()
def semantic_prediction(model, image, args, label_shape):
    height, width = image.shape[-2:]
    new_h, new_w = height, width
    resized = image.float()
    if args.short_side is not None:
        scale = args.short_side / min(height, width)
        new_h, new_w = round(height * scale), round(width * scale)
        resized = F.interpolate(resized, size=(new_h, new_w), mode="bilinear", align_corners=False)
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    normalized = (resized - mean) / std
    normalized = F.pad(normalized, (0, (-new_w) % 32, 0, (-new_h) % 32))
    with torch.autocast(
        device_type=image.device.type,
        enabled=image.device.type == "cuda" and args.precision == "fp16",
    ):
        if args.segmenter == "segformer":
            logits = model(normalized).logits
        else:
            logits = model.classifier(model.backbone(normalized))
    logits = logits.float()
    logits = F.interpolate(logits, size=normalized.shape[-2:], mode="bilinear", align_corners=False)
    logits = logits[..., :new_h, :new_w]
    logits = F.interpolate(logits, size=label_shape, mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


def restored_image(record, image, restorer, args):
    if restorer is not None:
        return restorer(image)["image"]
    prediction = load_prediction(args.predictions, record["id"]).unsqueeze(0).to(args.device)
    if prediction.shape != image.shape:
        raise ValueError(f"Restored and input image dimensions differ: {record['id']}")
    return prediction


@torch.inference_mode()
def segmentation(args):
    records = read_manifest(args.manifest)
    segmenter = segmentation_model(args)
    restorer, restoration_checkpoint = (
        loaded_model(args.checkpoint, args.device) if args.checkpoint else (None, None)
    )
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
        "precision": "fp16"
        if torch.device(args.device).type == "cuda" and args.precision == "fp16"
        else "fp32",
        "image_ids": [record["id"] for record in records],
        "restoration": checkpoint_identity(restoration_checkpoint)
        if restoration_checkpoint
        else {"predictions": args.predictions},
        "seed": restoration_checkpoint["seed"] if restoration_checkpoint else args.seed,
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


def build_detector(score_threshold=0.001, nms_threshold=0.5, pretrained=False):
    from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.ops.misc import FrozenBatchNorm2d

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=None, norm_layer=FrozenBatchNorm2d, trainable_layers=3
    )
    model = FasterRCNN(
        backbone,
        num_classes=91 if pretrained else 9,
        min_size=800,
        max_size=1333,
        box_score_thresh=score_threshold,
        box_nms_thresh=nms_threshold,
        box_detections_per_img=100,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=1000,
    )
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = 0.0
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
        model.roi_heads.box_predictor = FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, 9
        )
    return model


def detector_model(args):
    model = build_detector(args.score_threshold, args.nms_threshold)
    state = load_checkpoint(args.detector_weights)
    model.load_state_dict(state["model"])
    return model.eval().to(args.device).requires_grad_(False)


def filtered_detections(result, image_id, ignore_mask):
    height, width = ignore_mask.shape
    detections = []
    for box, label, score in zip(
        result["boxes"].cpu(), result["labels"].cpu(), result["scores"].cpu()
    ):
        x0, y0, x1, y1 = box.tolist()
        x = min(width - 1, max(0, int((x0 + x1) / 2)))
        y = min(height - 1, max(0, int((y0 + y1) / 2)))
        if not ignore_mask[y, x]:
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": float(score),
                }
            )
    return detections


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
    evaluation.params.iouThrs = np.linspace(0.5, 0.95, 10)
    evaluation.params.recThrs = np.linspace(0, 1, 101)
    evaluation.params.maxDets = [1, 10, 100]
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

    records = read_manifests(args.manifests or [args.manifest])
    coco = COCO(args.annotations)
    lookup = {image["file_name"]: image["id"] for image in coco.dataset["images"]}
    if {r["id"] for r in records} != set(lookup):
        raise ValueError("COCO annotations and manifest must contain exactly the same image IDs")
    detector = detector_model(args)
    restorer, restoration_checkpoint = (
        loaded_model(args.checkpoint, args.device) if args.checkpoint else (None, None)
    )
    predictions = {mode: [] for mode in ["input", "restored"]}
    for record in records:
        image, _ = pair(record)
        image = image.unsqueeze(0).to(args.device)
        restored = restored_image(record, image, restorer, args)
        ignore_mask = semantic_labels(record) == 255
        for mode, tensor in [("input", image), ("restored", restored)]:
            result = detector([tensor[0]])[0]
            predictions[mode].extend(filtered_detections(result, lookup[record["id"]], ignore_mask))
    conditions = sorted({r["task"] for r in records})
    report = {
        "detector_weights": args.detector_weights,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "image_ids": [record["id"] for record in records],
        "restoration": checkpoint_identity(restoration_checkpoint)
        if restoration_checkpoint
        else {"predictions": args.predictions},
        "seed": restoration_checkpoint["seed"] if restoration_checkpoint else args.seed,
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
    report["macro_condition_mean"] = {}
    for mode in ["input", "restored", "delta_pp"]:
        report["macro_condition_mean"][mode] = {
            key: float(
                np.mean(
                    [
                        report["conditions"][c][mode][key]
                        for c in conditions
                        if report["conditions"][c][mode][key] is not None
                    ]
                )
            )
            if any(report["conditions"][c][mode][key] is not None for c in conditions)
            else None
            for key in report["conditions"][conditions[0]][mode]
        }
    output = Path(args.output)
    for mode, values in predictions.items():
        write_json(output / f"{mode}_detections.json", values)
    write_json(output / "detection.json", report)


def train_detector(args):
    from pycocotools.coco import COCO
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
    validation_records = read_manifest(args.validation_manifest)
    validation_coco = COCO(args.validation_annotations)
    validation_ids = {
        image["file_name"]: image["id"] for image in validation_coco.dataset["images"]
    }
    if {r["id"] for r in validation_records} != set(validation_ids):
        raise ValueError("Detector validation manifest and annotations do not match")
    if {r["target"] for r in records} & {r["target"] for r in validation_records}:
        raise ValueError("Detector training and internal-validation scenes overlap")
    model = build_detector(pretrained=True)
    model.to(args.device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_ap, history = -1.0, []
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
        model.eval()
        validation_predictions = []
        with torch.inference_mode():
            for record in validation_records:
                image = rgb_tensor(record["target"]).to(args.device)
                result = model([image])[0]
                validation_predictions.extend(
                    filtered_detections(
                        result, validation_ids[record["id"]], semantic_labels(record) == 255
                    )
                )
        score = coco_report(validation_coco, validation_predictions, list(validation_ids.values()))[
            "AP_50_95"
        ]
        if score is None:
            raise ValueError("Detector validation set has no evaluable instance classes")
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_AP": score})
        if score > best_ap:
            best_ap = score
            torch.save(
                {
                    "model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "selected_epoch": epoch + 1,
                    "seed": args.seed,
                    "classes": 8,
                    "validation_AP": score,
                    "normalization": "FrozenBatchNorm2d_eps0",
                },
                output,
            )
        model.train()
    write_json(output.with_suffix(".json"), {"history": history, "selected_validation_AP": best_ap})
