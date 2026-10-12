import math

import numpy as np
import torch
from torch.nn import functional as F


def ssim(prediction, target):
    prediction, target = prediction.float(), target.float()
    size = min(11, *prediction.shape[-2:])
    if size % 2 == 0:
        size -= 1
    axis = torch.arange(size, device=prediction.device, dtype=torch.float32) - size // 2
    kernel = torch.exp(-axis.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel).expand(prediction.shape[1], 1, size, size)

    def smooth(x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            padding = size // 2
            mode = "reflect" if min(x.shape[-2:]) > padding else "replicate"
            padded = F.pad(x.float(), (padding, padding, padding, padding), mode=mode)
            return F.conv2d(padded, window, groups=x.shape[1])

    mu_x, mu_y = smooth(prediction), smooth(target)
    var_x = (smooth(prediction.square()) - mu_x.square()).clamp_min(0)
    var_y = (smooth(target.square()) - mu_y.square()).clamp_min(0)
    covariance = smooth(prediction * target) - mu_x * mu_y
    score = (
        (2 * mu_x * mu_y + 0.01**2)
        * (2 * covariance + 0.03**2)
        / ((mu_x.square() + mu_y.square() + 0.01**2) * (var_x + var_y + 0.03**2))
    )
    return score.mean((1, 2, 3))


def psnr(prediction, target):
    mse = (prediction.float() - target.float()).square().mean((1, 2, 3))
    return -10 * torch.log10(mse.clamp_min(1e-12))


class RestorationMetrics:
    def __init__(self, device, lpips_enabled=True):
        self.lpips = None
        if lpips_enabled:
            import lpips

            self.lpips = lpips.LPIPS(net="alex", version="0.1", spatial=False).eval().to(device)
            self.lpips.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, prediction, target):
        result = {"psnr": psnr(prediction, target).item(), "ssim": ssim(prediction, target).item()}
        if self.lpips is not None:
            result["lpips"] = self.lpips(2 * prediction - 1, 2 * target - 1, normalize=False).item()
        return result


def continual_metrics(matrix):
    r = np.asarray(matrix, dtype=float)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or not len(r):
        raise ValueError("Task matrix must be nonempty and square")
    if not np.isfinite(r[np.tril_indices(len(r))]).all():
        raise ValueError("All observed stage/task scores are required")
    n = len(r)
    forgetting = [max(r[j : n - 1, j]) - r[-1, j] for j in range(n - 1)]
    bwt = [r[-1, j] - r[j, j] for j in range(n - 1)]
    result = {
        "AP": float(r[-1].mean()),
        "F": float(np.mean(forgetting)) if n > 1 else 0.0,
        "BWT": float(np.mean(bwt)) if n > 1 else 0.0,
        "last_task_psnr": float(r[-1, -1]),
    }
    if n >= 8:
        result["AP_original_four"] = float(r[-1, :4].mean())
        result["original_four_loss"] = float(r[3, :4].mean() - r[-1, :4].mean())
        result["AP_new_severity"] = float(r[-1, 4:].mean())
    return result


def router_metrics(probabilities, labels):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    prediction = p.argmax(1)
    confusion = np.zeros((p.shape[1], p.shape[1]), dtype=np.int64)
    np.add.at(confusion, (y, prediction), 1)
    counts = confusion.sum(1)
    supported = counts > 0
    accuracy = np.diag(confusion)[supported] / counts[supported]
    confidence = p.max(1)
    correct = prediction == y
    ece = 0.0
    bins = np.minimum((confidence * 10).astype(int), 9)
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "macro_accuracy": float(accuracy.mean()),
        "ece_10_bins": float(ece),
        "confusion_counts": confusion.tolist(),
        "confusion_row_normalized": (confusion / counts.clip(1)[:, None]).tolist(),
    }


def entropy(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.shape[1] < 2:
        raise ValueError("Entropy diagnostics require at least two observed tasks")
    return -(p * np.log(p.clip(1e-12))).sum(1) / math.log(p.shape[1])


def auroc(positive, negative):
    from scipy.stats import rankdata

    positive, negative = np.asarray(positive), np.asarray(negative)
    if not len(positive) or not len(negative):
        raise ValueError("AUROC requires both positive and negative samples")
    ranks = rankdata(np.concatenate([positive, negative]), method="average")
    n, m = len(positive), len(negative)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def confusion_iou(confusion):
    c = np.asarray(confusion, dtype=np.float64)
    intersection = np.diag(c)
    union = c.sum(0) + c.sum(1) - intersection
    present = union > 0
    iou = np.divide(intersection, union, out=np.zeros_like(union), where=present)
    return {
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "class_iou": [float(v) if ok else None for v, ok in zip(iou, present)],
        "intersection": intersection.tolist(),
        "union": union.tolist(),
    }
