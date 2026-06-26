import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from models import ISNetDIS
except ImportError:
    from .models import ISNetDIS


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    BASE_DIR
    / "weights"
    / "gpu_itr_98000_traLoss_0.0577_traTarLoss_0.0027_valLoss_0.0726_valTarLoss_0.0047_maxF1_0.9521_mae_0.0024_time_0.027661.pth"
)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def iter_images(input_dir: Path, recursive: bool = False):
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            yield path


def load_model(model_path: Path, device: torch.device) -> ISNetDIS:
    if not model_path.exists():
        raise FileNotFoundError(f"Weight file not found: {model_path}")

    model = ISNetDIS()
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_rgb_image(image_path: Path):
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return np.array(rgb_image), rgb_image.size


def save_mask(mask: torch.Tensor, output_path: Path) -> None:
    mask = mask.squeeze(0).squeeze(0)
    minimum = torch.min(mask)
    maximum = torch.max(mask)
    mask = (mask - minimum) / (maximum - minimum + 1e-8)
    mask_image = (mask.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_image, mode="L").save(output_path)


@torch.inference_mode()
def infer_image(model: ISNetDIS, image_path: Path, output_path: Path, device: torch.device, input_size: int) -> Path:
    image_array, original_size = load_rgb_image(image_path)
    original_width, original_height = original_size

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32)
    image_tensor = F.interpolate(
        image_tensor,
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
    )
    image_tensor = image_tensor / 255.0 - 0.5

    outputs, _ = model(image_tensor)
    mask = F.interpolate(
        outputs[0],
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )
    save_mask(mask, output_path)
    return output_path


def run_inference(
    input_dir: Path,
    output_dir: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    device_name: str = "auto",
    input_size: int = 1024,
    recursive: bool = False,
) -> list[Path]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    model_path = model_path.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = list(iter_images(input_dir, recursive=recursive))
    if not images:
        raise FileNotFoundError(f"No input images found in: {input_dir}")

    device = select_device(device_name)
    model = load_model(model_path, device)

    saved_paths: list[Path] = []
    for image_path in tqdm(images, desc="Mask inference"):
        relative_path = image_path.relative_to(input_dir).with_suffix(".png")
        output_path = output_dir / relative_path
        saved_paths.append(infer_image(model, image_path, output_path, device, input_size))

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local print mask inference with IS-Net.")
    parser.add_argument("--input-dir", default=str(BASE_DIR / "input"), help="Directory containing input images.")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "output_detect"), help="Directory for grayscale masks.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Local IS-Net .pth weight path.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--input-size", type=positive_int, default=1024, help="Square inference size.")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively under input-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = run_inference(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_path=Path(args.model_path),
        device_name=args.device,
        input_size=args.input_size,
        recursive=args.recursive,
    )
    print(f"Saved {len(saved_paths)} mask(s) to: {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
