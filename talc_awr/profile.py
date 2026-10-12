import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import pair
from .evaluate import loaded_model
from .io import load_checkpoint, read_manifest, seed_everything, write_json
from .memory import memory_bytes


@torch.inference_mode()
def profile(args):
    device = torch.device(args.device)
    seed_everything(17)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, checkpoint = loaded_model(args.checkpoint, device)
    records = sorted(read_manifest(args.manifest), key=lambda record: record["id"])
    record = records[0]
    stored_parameters = sum(p.numel() for p in model.parameters())
    if checkpoint["variant"] == "separate":
        tasks = checkpoint["config"]["tasks"][: checkpoint["stage"]]
        directory = Path(args.checkpoint).parent
        bank = [
            load_checkpoint(directory / f"stage{stage}.pt") for stage in range(1, len(tasks) + 1)
        ]
        if any(
            item["variant"] != "separate" or item["seed"] != checkpoint["seed"] for item in bank
        ):
            raise ValueError(
                "Separate-expert profiling requires one consistent completed model bank"
            )
        stored_parameters = sum(value.numel() for item in bank for value in item["model"].values())
        if record["task"] not in tasks:
            raise ValueError("Separate experts require a known task for the timing image")
        model, _ = loaded_model(directory / f"stage{tasks.index(record['task']) + 1}.pt", device)
    image = pair(record)[0].unsqueeze(0).to(device)
    image = F.interpolate(
        image, size=(args.height, args.width), mode="bilinear", align_corners=False
    ).contiguous()
    top_k = args.top_k or model.default_top_k
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
        model(image, top_k=top_k)
    for hook in hooks:
        hook.remove()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    telemetry_process, telemetry_stream = None, None
    if args.telemetry:
        if device.type != "cuda":
            raise ValueError("GPU telemetry requires a CUDA profiling device")
        if shutil.which("tegrastats"):
            command = ["tegrastats", "--interval", "100"]
        elif shutil.which("nvidia-smi"):
            command = [
                "nvidia-smi",
                "--query-gpu=timestamp,index,name,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu",
                "--format=csv",
                "--loop-ms=100",
            ]
        else:
            raise FileNotFoundError("Install nvidia-smi or tegrastats for --telemetry")
        telemetry_stream = (output / "telemetry.log").open("w", encoding="utf-8")
        telemetry_process = subprocess.Popen(
            command,
            stdout=telemetry_stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    latencies, repeat_fps = [], []
    try:
        with torch.autocast(device_type=device.type, enabled=amp):
            for _ in range(args.repeats):
                for _ in range(args.warmup):
                    model(image, top_k=top_k)
                sync()
                repeat = []
                for _ in range(args.frames):
                    sync()
                    start = time.perf_counter()
                    model(image, top_k=top_k)
                    sync()
                    repeat.append((time.perf_counter() - start) * 1000)
                latencies.extend(repeat)
                repeat_fps.append(1000 / float(np.mean(repeat)))
        if telemetry_process is not None and telemetry_process.poll() is not None:
            raise RuntimeError("Telemetry process stopped; inspect telemetry.log")
    finally:
        if telemetry_process is not None:
            if telemetry_process.poll() is None:
                telemetry_process.terminate()
                telemetry_process.wait(timeout=10)
            telemetry_stream.close()
    macs = int(sum(operations.values()))
    report = {
        "variant": checkpoint["variant"],
        "seed": checkpoint["seed"],
        "parameters": stored_parameters,
        "active_model_parameters": sum(p.numel() for p in model.parameters()),
        "parameter_storage_fp32_mb": stored_parameters * 4 / 1e6,
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
        "top_k": top_k,
        "timing_image": record["id"],
        "manifest": args.manifest,
        "repeats": args.repeats,
        "warmup_per_repeat": args.warmup,
        "frames_per_repeat": args.frames,
        "repeat_fps": repeat_fps,
        "cpu_threads": 1,
        "tf32": False,
        "deterministic_cudnn": True,
        "telemetry_interval_ms": 100 if args.telemetry else None,
        "ambient_temperature_c": args.ambient_temperature,
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
    write_json(output / "profile.json", report)
    print({key: report[key] for key in ["parameters", "active_conv_linear_gflops", "fps"]})
