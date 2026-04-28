"""
Build infographic figures from Lightning CSV logs under `.logs/`.

Interprets *feature reliance* as the drop in macro test accuracy relative to the
`resize_no_suppression` baseline when controlled suppressions are applied at test time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Human-readable cue families (aligned with paper framing)
PROTOCOL_LABELS = {
    "resize_no_suppression": "Baseline (no suppression)",
    "resize_patch_shuffle": "Local shape — patch shuffle",
    "resize_patch_rotation": "Local shape — patch rotation",
    "resize_patch_shuffle_grayscale": "Shape + color",
    "resize_grayscale": "Color — grayscale mix",
    "resize_channel_shuffle": "Color — channel shuffle",
    "resize_bilateral": "Texture — bilateral",
    "resize_gaussianblur2": "Texture — Gaussian blur",
    "resize_nlmeans": "Texture — non-local means",
    "resize_bilateral_patch_shuffle": "Bilateral + patch shuffle",
    "resize_bilateral_patch_shuffle2": "Bilateral + patch shuffle (fixed)",
    "resize_bilateral_grayscale": "Bilateral + grayscale",
    "resize_wavelet_texture": "Texture — wavelet",
    "resize_wavelet_shape": "Shape — wavelet",
}


def load_hparams(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("cfg", raw)


def read_test_accmac(metrics_csv: Path) -> float | None:
    if not metrics_csv.is_file():
        return None
    df = pd.read_csv(metrics_csv)
    if df.empty or "test_accmac" not in df.columns:
        return None
    # Single test epoch: use last row
    return float(df["test_accmac"].iloc[-1])


def collect_runs(log_root: Path) -> pd.DataFrame:
    rows = []
    for hparams_path in log_root.rglob("hparams.yaml"):
        cfg = load_hparams(hparams_path)
        params = cfg.get("params", {})
        protocol = params.get("protocol_name")
        if not protocol or not str(protocol).startswith("resize_"):
            continue
        dataset = params.get("dataset")
        if not dataset:
            continue
        metrics_path = hparams_path.parent / "metrics.csv"
        acc = read_test_accmac(metrics_path)
        if acc is None:
            continue
        dataaug = cfg.get("dataaug", {})
        rows.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "test_accmac": acc,
                "grid_size": dataaug.get("grid_size"),
                "gray_alpha": dataaug.get("gray_alpha"),
                "bilateral_d": dataaug.get("bilateral_d"),
                "sigma_color": dataaug.get("sigma_color"),
                "gaussian_k": dataaug.get("gaussian_k"),
                "gaussian_sigma": dataaug.get("gaussian_sigma"),
                "nlmeans_h": dataaug.get("nlmeans_h"),
                "log_dir": str(hparams_path.parent),
            }
        )
    return pd.DataFrame(rows)


def baselines(df: pd.DataFrame) -> pd.Series:
    base = df[df["protocol"] == "resize_no_suppression"].groupby("dataset")["test_accmac"].max()
    return base


def summarize_protocols(df: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    """Per dataset & protocol: min/mean/max accuracy and max drop from baseline."""
    out = []
    for (dataset, protocol), g in df.groupby(["dataset", "protocol"]):
        b = base.get(dataset)
        if b is None or pd.isna(b):
            continue
        acc_min = g["test_accmac"].min()
        acc_mean = g["test_accmac"].mean()
        acc_max = g["test_accmac"].max()
        out.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "n_runs": len(g),
                "acc_min": acc_min,
                "acc_mean": acc_mean,
                "acc_max": acc_max,
                "baseline": b,
                "max_drop": b - acc_min,
                "mean_drop": b - acc_mean,
            }
        )
    return pd.DataFrame(out)


def plot_heatmap_max_drop(summary: pd.DataFrame, out_path: Path) -> None:
    pivot = summary.pivot(index="dataset", columns="protocol", values="max_drop")
    # Order columns: baseline first (should be 0), then alphabetical by label
    cols = [c for c in pivot.columns if c != "resize_no_suppression"]
    cols = sorted(cols, key=lambda x: PROTOCOL_LABELS.get(x, x))
    if "resize_no_suppression" in pivot.columns:
        pivot = pivot.drop(columns=["resize_no_suppression"], errors="ignore")
    pivot = pivot[cols]
    col_labels = [PROTOCOL_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 0.45), max(4, len(pivot) * 0.55)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=np.nanmax(pivot.values))
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Feature reliance: max accuracy drop vs. no-suppression baseline\n"
                 "(per protocol sweep, macro accuracy)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Δ accuracy (baseline − min in protocol)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bars_per_dataset(summary: pd.DataFrame, out_path: Path) -> None:
    datasets = sorted(summary["dataset"].unique())
    protocols = sorted(
        [p for p in summary["protocol"].unique() if p != "resize_no_suppression"],
        key=lambda x: PROTOCOL_LABELS.get(x, x),
    )
    x = np.arange(len(protocols))
    width = 0.8 / max(1, len(datasets))

    fig, ax = plt.subplots(figsize=(max(12, len(protocols) * 0.5), 6))
    for i, ds in enumerate(datasets):
        sub = summary[summary["dataset"] == ds].set_index("protocol")
        heights = [sub.loc[p, "max_drop"] if p in sub.index else 0.0 for p in protocols]
        ax.bar(x + (i - len(datasets) / 2) * width + width / 2, heights, width, label=ds)

    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_LABELS.get(p, p) for p in protocols], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Max Δ accuracy vs. baseline")
    ax.set_title("Strongest suppression effect per cue protocol (by dataset)")
    ax.legend(fontsize=8, ncol=2)
    ax.axhline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_patch_shuffle_curves(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df["protocol"] == "resize_patch_shuffle"].dropna(subset=["grid_size"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds in sorted(sub["dataset"].unique()):
        g = sub[sub["dataset"] == ds].groupby("grid_size")["test_accmac"].min()
        ax.plot(g.index.astype(int), g.values, marker="o", label=ds)
    ax.set_xlabel("Patch grid size")
    ax.set_ylabel("Macro accuracy (min over runs at grid size)")
    ax.set_title("Local shape suppression: patch shuffle vs. grid size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_baseline(summary: pd.DataFrame, base: pd.Series, out_path: Path) -> None:
    """Dot plot: mean accuracy per protocol vs baseline per dataset."""
    sub = summary[summary["protocol"] != "resize_no_suppression"].copy()
    if sub.empty:
        return
    datasets = sorted(sub["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        dsub = sub[sub["dataset"] == ds].sort_values("acc_mean")
        y = np.arange(len(dsub))
        b = float(base[ds])
        ax.axvline(b, color="tab:green", linestyle="--", linewidth=1, label="Baseline")
        ax.scatter(dsub["acc_mean"], y, color="tab:blue", s=36, zorder=3)
        ax.scatter(dsub["acc_min"], y, color="tab:red", s=22, alpha=0.7, zorder=2, label="Min (worst)")
        ax.set_yticks(y)
        ax.set_yticklabels([PROTOCOL_LABELS.get(p, p) for p in dsub["protocol"]], fontsize=7)
        ax.set_xlabel("test_accmac")
        ax.set_title(ds)
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].set_ylabel("Suppression protocol")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("Accuracy under suppression (mean and worst run per protocol)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot feature reliance infographics from .logs")
    ap.add_argument(
        "--log-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".logs",
        help="Root directory containing dataset/model/from_scratch/... logs",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "feature_reliance",
        help="Where to write PNG figures",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_runs(args.log_root)
    if df.empty:
        raise SystemExit(f"No reliance runs found under {args.log_root} (expected resize_* protocols).")

    base = baselines(df)
    summary = summarize_protocols(df, base)
    summary.to_csv(args.out_dir / "summary_by_protocol.csv", index=False)

    plot_heatmap_max_drop(summary, args.out_dir / "01_heatmap_max_drop.png")
    plot_bars_per_dataset(summary, args.out_dir / "02_bar_max_drop_by_dataset.png")
    plot_patch_shuffle_curves(df, args.out_dir / "03_patch_shuffle_grid_size.png")
    plot_accuracy_vs_baseline(summary, base, args.out_dir / "04_accuracy_mean_vs_baseline.png")

    print(f"Wrote figures and summary CSV to {args.out_dir}")


if __name__ == "__main__":
    main()
