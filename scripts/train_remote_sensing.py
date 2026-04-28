import argparse
import subprocess
import sys


REMOTE_SENSING_DATASETS = [
    "rsd46whu",
    "aid",
    "ucmerced",
    "patternnet",
    # "deepglobe",
]


def build_command(args, dataset):
    command = [
        sys.executable,
        "training.py",
        f"params.dataset={dataset}",
        f"params.max_epochs={args.max_epochs}",
        f"model.name={args.model}",
        f"logging.exp_dir={args.exp_dir}",
        f"params.cuda_no={args.cuda_no}",
        f"params.seed={args.seed}",
        f"params.batch_size={args.batch_size}",
        f"params.num_workers={args.num_workers}",
        f"params.slurm_bypass={str(args.slurm_bypass).lower()}",
    ]

    if args.pretrained:
        command.append("model.pretrained=true")

    return command


def main():
    parser = argparse.ArgumentParser(
        description="Train all remote sensing datasets with shared settings."
    )
    parser.add_argument("-m", "--model", type=str, default="resnet50")
    parser.add_argument("-l", "--exp_dir", type=str, required=True)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--cuda_no", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--slurm_bypass", action="store_true")
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=REMOTE_SENSING_DATASETS,
        help="Remote sensing datasets to train. Defaults to all supported ones.",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        print(f"\nTraining dataset: {dataset}")
        command = build_command(args, dataset)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
