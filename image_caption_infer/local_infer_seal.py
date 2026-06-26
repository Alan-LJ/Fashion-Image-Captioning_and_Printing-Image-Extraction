import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


DEFAULT_PROMPT = (
    "Describe the fashion items in the image concisely in English. "
    "Format: A simple plain paragraph. Do NOT output conversational filler."
)

INFER_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_ADAPTER = INFER_DIR / "fixed_reward_seed2026_20260515_184503"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _normalize_model_location(model_name_or_path: str) -> str:
    expanded = Path(model_name_or_path).expanduser()
    if model_name_or_path.startswith(("~", ".", "/")) or expanded.exists():
        return str(expanded)
    return model_name_or_path


def _resolve_base_model(base_model: str | None, adapter_path: Path) -> str:
    if base_model:
        return _normalize_model_location(base_model)

    run_config = _read_json(adapter_path / "run_config.json")
    model_id = run_config.get("model_id")
    if model_id:
        return _normalize_model_location(model_id)

    adapter_config = _read_json(adapter_path / "adapter_config.json")
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base:
        return _normalize_model_location(adapter_base)

    return DEFAULT_BASE_MODEL


def _select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return requested


def _select_dtype(requested: str, selected_device: str):
    if selected_device == "cpu":
        return torch.float32
    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[requested]


def load_model(
    base_model_path: str,
    adapter_path: str,
    device: str,
    dtype: str,
    min_pixels: int,
    max_pixels: int,
    local_files_only: bool,
):
    selected_device = _select_device(device)
    torch_dtype = _select_dtype(dtype, selected_device)

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if selected_device == "cuda":
        model_kwargs["device_map"] = "auto"

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        **model_kwargs,
    )
    if selected_device == "cpu":
        base_model = base_model.to("cpu")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
        local_files_only=local_files_only,
    )
    model.eval()
    return model, processor


@torch.inference_mode()
def infer(
    model,
    processor,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text_input],
        images=[image],
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = output_ids[:, prompt_len:]
    return processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Qwen2-VL image-to-text inference.")
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Local Qwen2-VL-2B-Instruct directory or Hugging Face model ID. "
            "Defaults to adapter run_config.json, then adapter_config.json, "
            f"then {DEFAULT_BASE_MODEL}."
        ),
    )
    parser.add_argument(
        "--adapter",
        default=str(DEFAULT_ADAPTER),
        help="SEAL LoRA adapter directory. Defaults to the adapter bundled in this inference folder.",
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=positive_int, default=128)
    parser.add_argument("--min-pixels", type=positive_int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=positive_int, default=1024 * 1024)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="CUDA dtype. CPU always uses float32.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load only local model/cache files and never download from Hugging Face.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = _normalize_path(args.adapter)
    image_path = _normalize_path(args.image)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter config not found: {adapter_path / 'adapter_config.json'}")
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Adapter weights not found: {adapter_path / 'adapter_model.safetensors'}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    base_model_path = _resolve_base_model(args.base_model, adapter_path)
    model, processor = load_model(
        base_model_path=base_model_path,
        adapter_path=str(adapter_path),
        device=args.device,
        dtype=args.dtype,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=args.local_files_only,
    )
    result = infer(
        model=model,
        processor=processor,
        image_path=str(image_path),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n=== SEAL Caption ===")
    print(result)


if __name__ == "__main__":
    main()
