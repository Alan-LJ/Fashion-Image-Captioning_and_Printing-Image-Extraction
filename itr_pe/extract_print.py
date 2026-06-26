import argparse
from pathlib import Path

try:
    from Inference import BASE_DIR, DEFAULT_MODEL_PATH, positive_int, run_inference
    from apply_mask import apply_masks
except ImportError:
    from .Inference import BASE_DIR, DEFAULT_MODEL_PATH, positive_int, run_inference
    from .apply_mask import apply_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract print masks and RGBA cutouts.")
    parser.add_argument("--input-dir", default=str(BASE_DIR / "input"), help="Directory containing input images.")
    parser.add_argument("--mask-dir", default=str(BASE_DIR / "output_detect"), help="Directory for retained grayscale masks.")
    parser.add_argument("--cutout-dir", default=str(BASE_DIR / "masked"), help="Directory for RGBA print cutouts.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Local IS-Net .pth weight path.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--input-size", type=positive_int, default=1024, help="Square inference size.")
    parser.add_argument("--threshold", type=int, default=127, help="Mask threshold for alpha channel.")
    parser.add_argument("--keep-holes", action="store_true", help="Do not fill enclosed transparent holes in the mask.")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively under input-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_paths = run_inference(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.mask_dir),
        model_path=Path(args.model_path),
        device_name=args.device,
        input_size=args.input_size,
        recursive=args.recursive,
    )
    cutout_paths = apply_masks(
        input_dir=Path(args.input_dir),
        mask_dir=Path(args.mask_dir),
        output_dir=Path(args.cutout_dir),
        threshold=args.threshold,
        fill_holes=not args.keep_holes,
        recursive=args.recursive,
    )

    print(f"Saved {len(mask_paths)} grayscale mask(s) to: {Path(args.mask_dir).expanduser().resolve()}")
    print(f"Saved {len(cutout_paths)} RGBA cutout(s) to: {Path(args.cutout_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
