from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.clocks.train_clock_cnn import ClockCNN, TrainConfig as ClockTrainConfig
from model.speech.train_pitt_transformer import (
    OUTPUT_DIR as SPEECH_OUTPUT_DIR,
    SpeechTransformerClassifier,
    TrainConfig as SpeechTrainConfig,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output_performance" / "model_architecture"
CLOCK_CHECKPOINT = REPO_ROOT / "output_performance" / "clocks" / "nhats_cnn" / "nhats_clock_cnn.pt"
SPEECH_CHECKPOINT = SPEECH_OUTPUT_DIR / "pitt_transformer.pt"

COLORS = {
    "input": "#E8F1FA",
    "conv": "#D9EAD3",
    "transform": "#D9EAD3",
    "pool": "#FFF2CC",
    "dense": "#FCE5CD",
    "output": "#EADCF8",
}


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    legacy_path = output_dir / f"{stem}_visualtorch.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copyfile(png_path, legacy_path)
    return [png_path, svg_path, legacy_path]


def _draw_pipeline(
    title: str,
    subtitle: str,
    labels: Sequence[str],
    kinds: Sequence[str],
) -> plt.Figure:
    if len(labels) != len(kinds):
        raise ValueError("Each architecture block must have a corresponding block kind.")

    fig, ax = plt.subplots(figsize=(15, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(title, x=0.5, y=0.96, fontsize=19, fontweight="bold")
    ax.text(0.5, 0.88, subtitle, ha="center", va="center", fontsize=10.5, color="#4A4A4A")

    margin = 0.035
    gap = 0.018
    box_width = (1 - 2 * margin - gap * (len(labels) - 1)) / len(labels)
    box_height = 0.34
    y = 0.34
    centers: list[tuple[float, float]] = []

    for index, (label, kind) in enumerate(zip(labels, kinds)):
        x = margin + index * (box_width + gap)
        patch = FancyBboxPatch(
            (x, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=1.4,
            edgecolor="#3D4A53",
            facecolor=COLORS[kind],
        )
        ax.add_patch(patch)
        ax.text(
            x + box_width / 2,
            y + box_height / 2,
            label,
            ha="center",
            va="center",
            multialignment="center",
            fontsize=8.4 if len(labels) > 7 else 9.3,
            linespacing=1.35,
        )
        centers.append((x + box_width / 2, y + box_height / 2))

    for left, right in zip(centers, centers[1:]):
        ax.annotate(
            "",
            xy=(right[0] - box_width / 2 - 0.003, right[1]),
            xytext=(left[0] + box_width / 2 + 0.003, left[1]),
            arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": "#3D4A53"},
        )

    legend_items = (
        ("input", "Input"),
        ("conv", "Feature extraction"),
        ("pool", "Pooling"),
        ("dense", "Classifier"),
        ("output", "Output"),
    )
    legend_x = 0.28
    for offset, (kind, text) in enumerate(legend_items):
        x = legend_x + offset * 0.115
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.16),
                0.018,
                0.028,
                boxstyle="round,pad=0.003",
                linewidth=0.8,
                edgecolor="#3D4A53",
                facecolor=COLORS[kind],
            )
        )
        ax.text(x + 0.025, 0.174, text, va="center", fontsize=8.5, color="#4A4A4A")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.04)
    return fig


def _clock_config(checkpoint: dict[str, Any] | None) -> ClockTrainConfig:
    if checkpoint and isinstance(checkpoint.get("config"), dict):
        return ClockTrainConfig(**checkpoint["config"])
    return ClockTrainConfig(
        clock_data_dir="dataset/clocks/ClockData_256",
        nhats_round_14b_sas="dataset/clocks/ClockData/NHATS_Round_14B_SP_File.sas7bdat",
        output_dir=str(REPO_ROOT / "output_performance" / "clocks" / "nhats_cnn"),
    )


def _downsample_size(size: int) -> int:
    return (size + 1) // 2


def render_clock_cnn(output_dir: Path) -> list[Path]:
    checkpoint = _load_checkpoint(CLOCK_CHECKPOINT)
    config = _clock_config(checkpoint)
    model = ClockCNN(dropout=config.dropout)
    if checkpoint and isinstance(checkpoint.get("model_state_dict"), dict):
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sizes = [config.image_size]
    for _ in range(4):
        sizes.append(_downsample_size(sizes[-1]))

    labels = (
        f"Grayscale image\n1 x {sizes[0]} x {sizes[0]}",
        f"Conv 5x5, stride 2\nBatchNorm + ReLU\n16 x {sizes[1]} x {sizes[1]}",
        f"Conv 3x3, stride 2\nBatchNorm + ReLU\n32 x {sizes[2]} x {sizes[2]}",
        f"Conv 3x3, stride 2\nBatchNorm + ReLU\n64 x {sizes[3]} x {sizes[3]}",
        f"Conv 3x3, stride 2\nBatchNorm + ReLU\n128 x {sizes[4]} x {sizes[4]}",
        f"Conv 3x3\nBatchNorm + ReLU\n128 x {sizes[4]} x {sizes[4]}",
        "Adaptive average pool\nFlatten\n128 features",
        f"Dropout {config.dropout:.2f}\nDense 128 -> 64\nReLU + dropout",
        "Dense 64 -> 6\nScore logits\n0, 1, 2, 3, 4, 5",
    )
    kinds = ("input", "conv", "conv", "conv", "conv", "conv", "pool", "dense", "output")
    fig = _draw_pipeline(
        "NHATS Clock-Drawing CNN",
        f"Compact baseline architecture | {_parameter_count(model):,} trainable parameters",
        labels,
        kinds,
    )
    return _save_figure(fig, output_dir, "clock_cnn_architecture")


def _speech_config(checkpoint: dict[str, Any] | None) -> SpeechTrainConfig:
    if checkpoint and isinstance(checkpoint.get("config"), dict):
        return SpeechTrainConfig(**checkpoint["config"])
    return SpeechTrainConfig(
        pitt_dir="dataset/speech/Pitt",
        output_dir=str(SPEECH_OUTPUT_DIR),
    )


def render_speech_transformer(output_dir: Path) -> list[Path]:
    checkpoint = _load_checkpoint(SPEECH_CHECKPOINT)
    config = _speech_config(checkpoint)
    vocab = checkpoint.get("vocab") if checkpoint else None
    vocab_size = len(vocab) if isinstance(vocab, dict) else config.max_vocab

    model = SpeechTransformerClassifier(vocab_size=vocab_size, config=config)
    if checkpoint and isinstance(checkpoint.get("model_state_dict"), dict):
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = (
        f"Token IDs + mask\nlength {config.max_len}",
        f"Token embedding\n{vocab_size:,} x {config.embed_dim}",
        f"Sinusoidal position\nencoding + dropout\n{config.max_len} x {config.embed_dim}",
        f"Transformer encoder x{config.num_layers}\n{config.num_heads} heads | FF {config.ff_dim}\nmasked attention",
        f"Mask-aware mean pool\nLayerNorm\n{config.embed_dim} features",
        f"Dropout {config.dropout:.2f}\nDense {config.embed_dim} -> {config.embed_dim}\nGELU + dropout",
        "Dense -> 2\nControl / Dementia\nclass logits",
    )
    kinds = ("input", "transform", "transform", "transform", "pool", "dense", "output")
    fig = _draw_pipeline(
        "Pitt Speech Transformer Classifier",
        f"Logical model stages | {_parameter_count(model):,} trainable parameters",
        labels,
        kinds,
    )
    return _save_figure(fig, output_dir, "speech_transformer_architecture")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compact PyTorch architecture schematics.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        choices=("all", "clock", "speech"),
        default="all",
        help="Which PyTorch architecture schematic to render.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    if args.model in {"all", "clock"}:
        generated.extend(render_clock_cnn(args.output_dir))
    if args.model in {"all", "speech"}:
        generated.extend(render_speech_transformer(args.output_dir))

    print("Generated compact model architecture figures:")
    for path in generated:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
