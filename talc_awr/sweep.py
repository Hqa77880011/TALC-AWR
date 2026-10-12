import shlex
import subprocess
import sys
from pathlib import Path

from .data import WEATHERS


def sweep(args):
    if args.suite == "orders" and not args.order_replay_plans:
        raise ValueError(
            "Order sensitivity requires --order-replay-plans for all three first-task references"
        )
    experiments = []
    if args.suite == "main":
        experiments = [
            (variant, variant, {}, None)
            for variant in [
                "finetune",
                "ewc",
                "lwf",
                "replay",
                "derpp",
                "talc",
                "joint",
                "separate",
            ]
        ]
    elif args.suite == "ablations":
        experiments = [
            (variant, variant, {}, None)
            for variant in [
                "talc",
                "talc_uniform",
                "talc_shared",
                "talc_no_replay",
                "talc_no_orth",
                "talc_top1",
            ]
        ]
    elif args.suite == "capacity":
        experiments = [
            (variant, variant, {}, None)
            for variant in [
                "talc",
                "derpp_capacity",
                "talc_unfreeze",
                "talc_no_distill",
                "talc_unfreeze_no_distill",
            ]
        ]
    elif args.suite == "memory":
        experiments = [
            (f"memory{count}", "talc", {"--pairs-per-task": count}, None)
            for count in [0, 8, 16, 32, 64]
        ]
    elif args.suite == "orders":
        experiments = [
            (f"order{index + 1}", "talc", {}, order)
            for index, order in enumerate(
                [
                    WEATHERS,
                    ["fog", "low_light", "rain", "snow"],
                    ["snow", "rain", "low_light", "fog"],
                ]
            )
        ]
    elif args.suite == "eight":
        tasks = WEATHERS + [f"{task}_high" for task in WEATHERS]
        experiments = [
            ("talc_fixed_total", "talc", {"--total-pairs": 128}, tasks),
            ("derpp_fixed_total", "derpp", {"--total-pairs": 128}, tasks),
            ("talc_fixed_per_task", "talc", {}, tasks),
        ]
    for label, variant, options, tasks in experiments:
        for seed in args.seeds:
            output = Path(args.output) / args.suite / label / f"seed{seed}"
            command = [
                sys.executable,
                "-m",
                "talc_awr",
                "train",
                "--config",
                args.config,
                "--data",
                args.data,
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--output",
                str(output),
            ]
            replay_plan = (
                args.order_replay_plans[int(label[-1]) - 1]
                if args.suite == "orders"
                else args.replay_plan
            )
            if replay_plan:
                command.extend(["--replay-plan", replay_plan])
            for key, value in options.items():
                command.extend([key, str(value)])
            if tasks:
                command.extend(["--tasks", ",".join(tasks)])
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
