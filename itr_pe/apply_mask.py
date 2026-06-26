import argparse
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(input_dir: Path, recursive: bool = False):
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            yield path


def fill_binary_holes(foreground: np.ndarray) -> np.ndarray:
    foreground = foreground.astype(bool)
    height, width = foreground.shape
    outside = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def push(row: int, col: int) -> None:
        if row < 0 or row >= height or col < 0 or col >= width:
            return
        if foreground[row, col] or outside[row, col]:
            return
        outside[row, col] = True
        queue.append((row, col))

    for row in range(height):
        push(row, 0)
        push(row, width - 1)
    for col in range(width):
        push(0, col)
        push(height - 1, col)

    while queue:
        row, col = queue.popleft()
        push(row - 1, col)
        push(row + 1, col)
        push(row, col - 1)
        push(row, col + 1)

    holes = ~foreground & ~outside
    return foreground | holes


def mask_to_alpha(mask: Image.Image, threshold: int = 127, fill_holes: bool = True) -> Image.Image:
    alpha_array = np.array(mask, dtype=np.uint8) >= threshold
    if fill_holes:
        alpha_array = fill_binary_holes(alpha_array)
    alpha = (alpha_array.astype(np.uint8) * 255)
    return Image.fromarray(alpha, mode="L")


def apply_alpha_mask(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    threshold: int = 127,
    fill_holes: bool = True,
) -> Path:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    with Image.open(image_path) as image:
        rgba = image.convert("RGBA")

    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")

    if mask.size != rgba.size:
        mask = mask.resize(rgba.size, Image.Resampling.NEAREST)

    alpha = mask_to_alpha(mask, threshold=threshold, fill_holes=fill_holes)
    rgba.putalpha(alpha)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output_path)
    return output_path


def apply_masks(
    input_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    threshold: int = 127,
    fill_holes: bool = True,
    recursive: bool = False,
) -> list[Path]:
    input_dir = input_dir.expanduser().resolve()
    mask_dir = mask_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    saved_paths: list[Path] = []
    missing_masks: list[Path] = []
    images = list(iter_images(input_dir, recursive=recursive))
    if not images:
        raise FileNotFoundError(f"No input images found in: {input_dir}")

    for image_path in tqdm(images, desc="Apply masks"):
        relative_path = image_path.relative_to(input_dir).with_suffix(".png")
        mask_path = mask_dir / relative_path
        output_path = output_dir / relative_path
        if not mask_path.exists():
            missing_masks.append(mask_path)
            continue
        saved_paths.append(
            apply_alpha_mask(
                image_path,
                mask_path,
                output_path,
                threshold=threshold,
                fill_holes=fill_holes,
            )
        )

    for mask_path in missing_masks:
        print(f"Mask not found: {mask_path}")

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply grayscale masks to input images and save RGBA cutouts.")
    parser.add_argument("--input-dir", default=str(BASE_DIR / "input"), help="Directory containing source images.")
    parser.add_argument("--mask-dir", default=str(BASE_DIR / "output_detect"), help="Directory containing PNG masks.")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "masked"), help="Directory for RGBA cutouts.")
    parser.add_argument("--threshold", type=int, default=127, help="Mask threshold for alpha channel.")
    parser.add_argument("--keep-holes", action="store_true", help="Do not fill enclosed transparent holes in the mask.")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively under input-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = apply_masks(
        input_dir=Path(args.input_dir),
        mask_dir=Path(args.mask_dir),
        output_dir=Path(args.output_dir),
        threshold=args.threshold,
        fill_holes=not args.keep_holes,
        recursive=args.recursive,
    )
    print(f"Saved {len(saved_paths)} cutout(s) to: {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
