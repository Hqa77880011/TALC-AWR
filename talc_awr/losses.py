import torch
from torch.nn import functional as F

from .metrics import ssim


def restoration_loss(prediction, target, ssim_weight):
    error = prediction.float() - target.float()
    return torch.sqrt(error.square() + 1e-6).mean() + ssim_weight * (
        1 - ssim(prediction, target).mean()
    )


def distillation_loss(student, teacher, feature_weight):
    result = F.l1_loss(student["image"].float(), teacher["image"].float())
    for current, previous in zip(student["features"], teacher["features"]):
        result = result + feature_weight * F.l1_loss(current.float(), previous.float())
    return result


def routing_loss(current, replay, current_labels, replay_labels):
    if current["logits"] is None:
        return current["image"].new_zeros(())
    logits, labels = [current["logits"]], [current_labels]
    if replay is not None:
        logits.append(replay["logits"])
        labels.append(replay_labels)
    return F.cross_entropy(torch.cat(logits), torch.cat(labels))
