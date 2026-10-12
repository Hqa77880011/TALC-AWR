import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluate import loaded_model
from .io import write_json
from .memory import memory_bytes


@torch.inference_mode()
def profile(args):
    device = torch.device(args.device)
    model, checkpoint = loaded_model(args.checkpoint, device)
    image = torch.rand(1, 3, args.height, args.width, device=device)
    amp = args.precision == "fp16"
    if amp and device.type != "cuda":
        raise ValueError("FP16 profiling requires CUDA")
    operations = {"conv_macs": 0, "linear_macs": 0}

    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            operations["conv_macs"] += (
                output.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size)
            )
        elif isinstance(module, nn.Linear):
            operations["linear_macs"] += output.numel() * module.in_features

    hooks = [
        module.register_forward_hook(count)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    with torch.autocast(device_type=device.type, enabled=amp):
        model(image, top_k=args.top_k)
    for hook in hooks:
        hook.remove()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device.type, enabled=amp):
        for _ in range(args.warmup):
            model(image, top_k=args.top_k)
        sync()
        latencies = []
        for _ in range(args.frames):
            sync()
            start = time.perf_counter()
            model(image, top_k=args.top_k)
            sync()
            latencies.append((time.perf_counter() - start) * 1000)
    parameters = sum(p.numel() for p in model.parameters())
    macs = int(sum(operations.values()))
    report = {
        "parameters": parameters,
        "parameter_storage_fp32_mb": parameters * 4 / 1e6,
        "paired_memory_mib": memory_bytes(checkpoint["memory"]) / 2**20,
        "active_conv_linear_macs": macs,
        "active_conv_linear_gflops": 2 * macs / 1e9,
        "flops_convention": "2 FLOPs per convolution/linear MAC; elementwise operations excluded",
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "fps": 1000 / float(np.mean(latencies)),
        "latency_ms": latencies,
        "precision": args.precision,
        "input_shape": list(image.shape),
        "top_k": args.top_k,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "device_setup": args.device_setup,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 1e6
        if device.type == "cuda"
        else None,
    }
    write_json(Path(args.output) / "profile.json", report)
    print({key: report[key] for key in ["parameters", "active_conv_linear_gflops", "fps"]})
