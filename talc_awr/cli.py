import argparse

from .compare import compare
from .data import export_data, prepare_acdc, prepare_cityscapes, prepare_detection, prepare_pairs
from .evaluate import diagnostics, evaluate, evaluate_matrix, infer, score_predictions
from .memory import select_memory
from .perception import detection, segmentation, train_detector
from .profile import profile
from .report import plot_comparison, plot_diagnostics, plot_perception, qualitative, report
from .sweep import sweep
from .train import VARIANTS, train


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def common(parser, output=True):
    parser.add_argument(
        "--device", default="cuda", help="PyTorch device, such as cuda, cuda:1 or cpu"
    )
    if output:
        parser.add_argument("--output", required=True)


def restoration_metrics(parser):
    parser.add_argument("--no-lpips", action="store_true", help="Compute only PSNR and SSIM")


def main():
    parser = argparse.ArgumentParser(prog="talc", description="TALC-AWR implementation")
    commands = parser.add_subparsers(dest="command", required=True)
    city = commands.add_parser(
        "prepare-cityscapes", help="Prepare splits, depth and weather manifests"
    )
    city.add_argument("--root", required=True)
    city.add_argument("--output", default="data/weather")
    city.add_argument("--seed", type=int, default=2025, help="City-stratified split seed")
    city.add_argument("--degradation-root", type=int, default=41001)
    city.add_argument("--validation-size", type=positive_int, default=500)
    city.add_argument("--variants", type=positive_int, default=4)
    city.set_defaults(function=prepare_cityscapes)
    pairs = commands.add_parser("prepare-pairs", help="Index aligned external input/target folders")
    for key in ["inputs", "targets", "output", "task"]:
        pairs.add_argument(f"--{key}", required=True)
    pairs.add_argument("--input-suffix", default="")
    pairs.add_argument("--target-suffix", default="")
    pairs.set_defaults(function=prepare_pairs)
    export = commands.add_parser(
        "export-data", help="Save manifest images for external model input"
    )
    export.add_argument("--manifest", required=True)
    export.add_argument("--output", required=True)
    export.set_defaults(function=export_data)
    acdc = commands.add_parser("prepare-acdc")
    acdc.add_argument("--root", required=True)
    acdc.add_argument("--output", default="data/acdc.jsonl")
    acdc.set_defaults(function=prepare_acdc)
    boxes = commands.add_parser("prepare-detection")
    detection_sources = boxes.add_mutually_exclusive_group(required=True)
    detection_sources.add_argument("--manifest")
    detection_sources.add_argument("--manifests", nargs="+")
    boxes.add_argument("--output", required=True)
    boxes.set_defaults(function=prepare_detection)
    memory = commands.add_parser("select-memory", help="Create a shared ordered herding plan")
    common(memory)
    memory.add_argument("--config", default="configs/default.yaml")
    memory.add_argument("--data", default="data/weather")
    memory.add_argument("--reference", required=True)
    memory.add_argument("--seed", type=int, choices=[17], default=17, help="Common reference seed")
    memory.add_argument("--tasks", help="Comma-separated tasks to include in the shared plan")
    memory.add_argument("--max-pairs", type=positive_int, default=64)
    memory.set_defaults(function=select_memory)
    training = commands.add_parser("train")
    common(training)
    training.add_argument("--config", default="configs/default.yaml")
    training.add_argument("--data", default="data/weather")
    training.add_argument("--variant", choices=VARIANTS, default="talc")
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--replay-plan")
    training.add_argument(
        "--resume", help="Completed stage checkpoint; continue into a new output directory"
    )
    training.add_argument(
        "--tasks", help="Comma-separated task order, including optional _high tasks"
    )
    for key in ["steps", "batch-size", "validation-interval", "stage-limit"]:
        training.add_argument(f"--{key}", type=positive_int)
    training.add_argument("--pairs-per-task", type=int)
    training.add_argument("--total-pairs", type=int)
    training.set_defaults(function=train)
    evaluation = commands.add_parser("evaluate")
    common(evaluation)
    restoration_metrics(evaluation)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifests", nargs="+", required=True)
    evaluation.add_argument("--policy", choices=["top1", "top2", "true_pair", "wrong_pair"])
    evaluation.add_argument("--save-images", action="store_true")
    evaluation.set_defaults(function=evaluate)
    matrix = commands.add_parser("evaluate-matrix")
    common(matrix)
    restoration_metrics(matrix)
    matrix.add_argument("--run", required=True)
    matrix.add_argument("--data", default="data/weather")
    matrix.add_argument("--name", help="Experiment name used to group runs in reports")
    matrix.set_defaults(function=evaluate_matrix)
    routing = commands.add_parser("diagnostics")
    common(routing)
    restoration_metrics(routing)
    routing.add_argument("--checkpoint", required=True)
    routing.add_argument("--data", default="data/weather")
    routing.add_argument("--known-per-task", type=positive_int, default=125)
    routing.set_defaults(function=diagnostics)
    inference = commands.add_parser("infer")
    common(inference)
    inference.add_argument("--checkpoint", required=True)
    source = inference.add_mutually_exclusive_group(required=True)
    source.add_argument("--input")
    source.add_argument("--manifest")
    inference.add_argument("--policy", choices=["top1", "top2"])
    inference.add_argument("--format", choices=["png", "npy", "both"], default="both")
    inference.set_defaults(function=infer)
    score = commands.add_parser("score-predictions")
    common(score)
    restoration_metrics(score)
    scoring_manifests = score.add_mutually_exclusive_group(required=True)
    scoring_manifests.add_argument("--manifest")
    scoring_manifests.add_argument("--manifests", nargs="+")
    score.add_argument("--seed", type=int)
    for key in ["predictions", "name"]:
        score.add_argument(f"--{key}", required=True)
    score.set_defaults(function=score_predictions)
    segment = commands.add_parser("segmentation")
    common(segment)
    segment.add_argument("--manifest", required=True)
    front = segment.add_mutually_exclusive_group(required=True)
    front.add_argument("--checkpoint")
    front.add_argument("--predictions")
    segment.add_argument("--segmenter", choices=["segformer", "deeplab"], default="segformer")
    segment.add_argument(
        "--segmenter-weights", default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    )
    segment.add_argument("--deeplab-source")
    segment.add_argument(
        "--short-side",
        type=positive_int,
        help="Optional resized evaluation; default is native resolution",
    )
    segment.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    segment.add_argument("--seed", type=int, help="Seed of externally generated predictions")
    segment.set_defaults(function=segmentation)
    detector = commands.add_parser("detection")
    common(detector)
    detection_sources = detector.add_mutually_exclusive_group(required=True)
    detection_sources.add_argument("--manifest")
    detection_sources.add_argument("--manifests", nargs="+")
    detector.add_argument("--annotations", required=True)
    detector.add_argument("--detector-weights", required=True)
    detector.add_argument("--score-threshold", type=float, default=0.001)
    detector.add_argument("--nms-threshold", type=float, default=0.5)
    detector.add_argument("--seed", type=int, help="Seed of externally generated predictions")
    front = detector.add_mutually_exclusive_group(required=True)
    front.add_argument("--checkpoint")
    front.add_argument("--predictions")
    detector.set_defaults(function=detection)
    detector_training = commands.add_parser("train-detector")
    common(detector_training)
    detector_training.add_argument("--manifest", required=True)
    detector_training.add_argument("--annotations", required=True)
    detector_training.add_argument("--validation-manifest", required=True)
    detector_training.add_argument("--validation-annotations", required=True)
    detector_training.add_argument("--epochs", type=positive_int, default=12)
    detector_training.add_argument("--lr", type=float, default=0.005)
    detector_training.add_argument("--seed", type=int, default=17)
    detector_training.set_defaults(function=train_detector)
    timing = commands.add_parser("profile")
    common(timing)
    timing.add_argument("--checkpoint", required=True)
    timing.add_argument("--manifest", default="data/weather/test_rain.jsonl")
    timing.add_argument("--height", type=positive_int, default=512)
    timing.add_argument("--width", type=positive_int, default=1024)
    timing.add_argument("--warmup", type=positive_int, default=100)
    timing.add_argument("--frames", type=positive_int, default=500)
    timing.add_argument("--repeats", type=positive_int, default=3)
    timing.add_argument("--top-k", type=positive_int)
    timing.add_argument("--telemetry", action="store_true")
    timing.add_argument("--ambient-temperature", type=float)
    timing.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    timing.add_argument(
        "--device-setup", default="", help="Power mode, clocks and other device settings"
    )
    timing.set_defaults(function=profile)
    aggregate = commands.add_parser("report")
    aggregate.add_argument("--evaluations", nargs="+", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(function=report)
    comparison = commands.add_parser(
        "compare", help="Compare restoration, perception, routing or device results"
    )
    comparison.add_argument(
        "--kind",
        choices=["restoration", "segmentation", "detection", "routing", "profile"],
        required=True,
    )
    comparison.add_argument("--results", nargs="+", required=True)
    comparison.add_argument("--names", nargs="+")
    comparison.add_argument("--reference")
    comparison.add_argument("--output", required=True)
    comparison.set_defaults(function=compare)
    comparison_plot = commands.add_parser("plot-comparison")
    comparison_plot.add_argument("--comparison", required=True)
    comparison_plot.add_argument("--metrics", nargs="+")
    comparison_plot.add_argument("--restoration-comparison")
    comparison_plot.add_argument("--output", required=True)
    comparison_plot.set_defaults(function=plot_comparison)
    route_plot = commands.add_parser("plot-diagnostics")
    route_plot.add_argument("--diagnostics", required=True)
    route_plot.add_argument("--output", required=True)
    route_plot.set_defaults(function=plot_diagnostics)
    perception_plot = commands.add_parser("plot-perception")
    perception_plot.add_argument("--results", required=True)
    perception_plot.add_argument("--output", required=True)
    perception_plot.set_defaults(function=plot_perception)
    panels = commands.add_parser("qualitative")
    panels.add_argument("--manifest", required=True)
    panels.add_argument("--predictions", nargs="+", required=True)
    panels.add_argument("--names", nargs="+")
    panels.add_argument("--limit", type=positive_int, default=4)
    panels.add_argument("--output", required=True)
    panels.set_defaults(function=qualitative)
    experiments = commands.add_parser("sweep")
    common(experiments)
    experiments.add_argument(
        "--suite",
        choices=["main", "ablations", "capacity", "memory", "orders", "eight"],
        required=True,
    )
    experiments.add_argument("--config", default="configs/default.yaml")
    experiments.add_argument("--data", default="data/weather")
    experiments.add_argument("--replay-plan")
    experiments.add_argument(
        "--order-replay-plans", nargs=3, help="Plans for default, fog-first and snow-first orders"
    )
    experiments.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    experiments.add_argument(
        "--dry-run", action="store_true", help="Print commands without training"
    )
    experiments.set_defaults(function=sweep)
    args = parser.parse_args()
    args.function(args)
