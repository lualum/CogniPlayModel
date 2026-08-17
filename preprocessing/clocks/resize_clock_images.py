from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_CLOCK_DATA_DIR = REPO_ROOT / "dataset" / "clocks" / "ClockData"
RESIZED_CLOCK_DATA_DIR = REPO_ROOT / "dataset" / "clocks" / "ClockData_256"
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ResizeConfig:
    input_dir: str
    output_dir: str
    size: int = 256
    output_format: str = "png"
    overwrite: bool = False


def output_path_for(input_path: Path, input_root: Path, output_root: Path, output_format: str) -> Path:
    relative_path = input_path.relative_to(input_root)
    return (output_root / relative_path).with_suffix(f".{output_format.lower()}")


def resize_image(input_path: Path, output_path: Path, size: int, output_format: str, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as raw_image:
        image = raw_image.convert("L")
    image = image.resize((size, size), Image.Resampling.LANCZOS)

    if output_format.lower() == "png":
        image.save(output_path, format="PNG", optimize=True)
    else:
        image.save(output_path)
    return True


def resize_clock_images(config: ResizeConfig) -> dict[str, object]:
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input clock data directory: {input_dir}")

    total = 0
    written = 0
    split_counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        split_dir = input_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing input split directory: {split_dir}")

        split_written = 0
        for input_path in sorted(split_dir.iterdir()):
            if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            total += 1
            output_path = output_path_for(input_path, input_dir, output_dir, config.output_format)
            if resize_image(input_path, output_path, config.size, config.output_format, config.overwrite):
                written += 1
                split_written += 1
        split_counts[split] = split_written

    return {
        "config": asdict(config),
        "input_images": total,
        "written_images": written,
        "written_by_split": split_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize NHATS clock images into a smaller training directory.")
    parser.add_argument("--input-dir", type=Path, default=RAW_CLOCK_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESIZED_CLOCK_DATA_DIR)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output-format", choices=["png", "tif"], default="png")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = resize_clock_images(
        ResizeConfig(
            input_dir=str(args.input_dir),
            output_dir=str(args.output_dir),
            size=args.size,
            output_format=args.output_format,
            overwrite=args.overwrite,
        )
    )
    print("Clock image resize complete")
    print(f"Input images: {result['input_images']}")
    print(f"Written images: {result['written_images']}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
