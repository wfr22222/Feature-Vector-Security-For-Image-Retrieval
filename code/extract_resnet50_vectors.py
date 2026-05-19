from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use ResNet50 to extract image feature vectors into a TXT file."
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path(__file__).resolve().parent / "pic",
        help="Root directory containing image files (default: ./pic).",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=Path(__file__).resolve().parent / "resnet50_vectors.txt",
        help="Output TXT file path (default: ./resnet50_vectors.txt).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction.",
    )
    return parser.parse_args()


def collect_images(image_root: Path) -> list[Path]:
    images = [
        p
        for p in image_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
    images.sort(key=lambda p: str(p).lower())
    return images


def build_feature_extractor(device: torch.device) -> tuple[torch.nn.Module, callable]:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()
    preprocess = weights.transforms()
    return feature_extractor, preprocess


def batched(iterable: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def extract_features(
    image_paths: list[Path],
    image_root: Path,
    batch_size: int,
) -> list[tuple[str, str, list[float]]]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script is configured to run on GPU only."
        )
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    model, preprocess = build_feature_extractor(device)
    outputs: list[tuple[str, str, list[float]]] = []

    with torch.no_grad():
        for batch_paths in tqdm(list(batched(image_paths, batch_size)), desc="Extracting"):
            tensors = []
            valid_paths = []

            for image_path in batch_paths:
                try:
                    with Image.open(image_path) as img:
                        img = img.convert("RGB")
                        tensors.append(preprocess(img))
                        valid_paths.append(image_path)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Skip unreadable file: {image_path} ({exc})")

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(device)
            features = model(batch_tensor).squeeze(-1).squeeze(-1).cpu()

            for p, vec in zip(valid_paths, features):
                rel_path = p.relative_to(image_root).as_posix()
                filename = p.name
                vector = vec.tolist()
                outputs.append((filename, rel_path, vector))

    return outputs


def save_txt(records: list[tuple[str, str, list[float]]], output_txt: Path) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as f:
        for filename, rel_path, vector in records:
            vector_text = ",".join(f"{x:.6f}" for x in vector)
            f.write(f"{filename}\t{rel_path}\t{vector_text}\n")


def main() -> None:
    args = parse_args()
    image_root = args.image_root.resolve()
    output_txt = args.output_txt.resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: {image_root}")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    image_paths = collect_images(image_root)
    if not image_paths:
        raise RuntimeError(f"No images found under: {image_root}")

    print(f"Found {len(image_paths)} images under {image_root}")
    records = extract_features(image_paths, image_root, args.batch_size)
    save_txt(records, output_txt)
    print(f"Saved {len(records)} vectors to: {output_txt}")


if __name__ == "__main__":
    main()
