from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from PIL import Image
from torch.hub import get_dir
from torchvision.models import ResNet50_Weights, resnet50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image search with ResNet50 + Gradio.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path(__file__).resolve().parent / "pic",
        help="Root directory for image dataset.",
    )
    parser.add_argument(
        "--vector-txt",
        type=Path,
        default=Path(__file__).resolve().parent / "resnet50_vectors.txt",
        help="TXT file containing filename, relative path and vectors.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of similar images to return.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Gradio host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio port.",
    )
    parser.add_argument(
        "--encrypted-index",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "secure_vectors.enc.txt",
        help="Encrypted vector index path (AES-GCM, TXT format).",
    )
    parser.add_argument(
        "--vector-sha256",
        type=Path,
        default=Path(__file__).resolve().parent / "resnet50_vectors.sha256",
        help="SHA-256 record file for vector txt integrity check.",
    )
    parser.add_argument(
        "--rebuild-secure-index",
        action="store_true",
        help="Rebuild encrypted index from vector txt.",
    )
    return parser.parse_args()


class ImageSearchEngine:
    def __init__(
        self,
        image_root: Path,
        vector_txt: Path,
        encrypted_index: Path,
        vector_sha256: Path,
        topk: int = 10,
        rebuild_secure_index: bool = False,
    ) -> None:
        self.image_root = image_root.resolve()
        self.vector_txt = vector_txt.resolve()
        self.encrypted_index = encrypted_index.resolve()
        self.vector_sha256 = vector_sha256.resolve()
        self.topk = topk

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root does not exist: {self.image_root}")
        if not self.vector_txt.exists():
            raise FileNotFoundError(f"Vector txt does not exist: {self.vector_txt}")

        self.aes_key = self._load_aes_key()
        self.device = torch.device("cpu")
        print("Using device: CPU")

        self.model, self.preprocess = self._build_model()
        self._verify_or_create_vector_sha256()
        self.relative_paths, self.features = self._load_vectors_secure(
            rebuild_secure_index=rebuild_secure_index
        )

    def _load_aes_key(self) -> bytes:
        key_str = os.getenv("IMAGE_SEARCH_AES_KEY", "").strip()
        if not key_str:
            raise RuntimeError(
                "Missing IMAGE_SEARCH_AES_KEY. "
                "Set a 32-byte key (plain 32-char, 64-char hex, or base64)."
            )

        try:
            if len(key_str) == 32:
                key = key_str.encode("utf-8")
            elif len(key_str) == 64:
                key = bytes.fromhex(key_str)
            else:
                try:
                    key = base64.b64decode(key_str, validate=True)
                except Exception:
                    key = key_str.encode("utf-8")
        except Exception as exc:
            raise RuntimeError("Failed to parse IMAGE_SEARCH_AES_KEY.") from exc

        if len(key) != 32:
            raise RuntimeError("IMAGE_SEARCH_AES_KEY must be exactly 32 bytes for AES-256.")
        return key

    def _sha256_file(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _verify_or_create_vector_sha256(self) -> None:
        digest = self._sha256_file(self.vector_txt)
        if not self.vector_sha256.exists():
            self.vector_sha256.write_text(digest + "\n", encoding="utf-8")
            print(f"[INFO] Created vector integrity file: {self.vector_sha256}")
            return

        stored = self.vector_sha256.read_text(encoding="utf-8").strip()
        if stored != digest:
            raise RuntimeError(
                "Vector txt integrity check failed (SHA-256 mismatch). "
                "If you intentionally changed vectors, update hash file or rebuild index."
            )
        print("[INFO] Vector txt integrity check passed (SHA-256).")

    def _encrypt_bytes(self, data: bytes) -> dict[str, str]:
        nonce = get_random_bytes(12)
        cipher = AES.new(self.aes_key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(data)
        return {
            "alg": "AES-256-GCM",
            "nonce_b64": base64.b64encode(nonce).decode("utf-8"),
            "tag_b64": base64.b64encode(tag).decode("utf-8"),
            "ciphertext_b64": base64.b64encode(ciphertext).decode("utf-8"),
        }

    def _decrypt_bytes(self, payload: dict[str, str]) -> bytes:
        nonce = base64.b64decode(payload["nonce_b64"])
        tag = base64.b64decode(payload["tag_b64"])
        ciphertext = base64.b64decode(payload["ciphertext_b64"])
        cipher = AES.new(self.aes_key, AES.MODE_GCM, nonce=nonce)
        return cipher.decrypt_and_verify(ciphertext, tag)

    def _build_model(self) -> tuple[torch.nn.Module, callable]:
        weights = ResNet50_Weights.IMAGENET1K_V2
        try:
            model = resnet50(weights=weights)
        except RuntimeError as exc:
            # Common on interrupted downloads: cached .pth is corrupted.
            cache_file = Path(get_dir()) / "checkpoints" / "resnet50-11ad3fa6.pth"
            if cache_file.exists():
                print(f"[WARN] Remove corrupted weight cache: {cache_file}")
                cache_file.unlink()
                model = resnet50(weights=weights)
            else:
                raise RuntimeError(f"Failed to load ResNet50 weights: {exc}") from exc
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(self.device)
        feature_extractor.eval()
        preprocess = weights.transforms()
        return feature_extractor, preprocess

    def _load_vectors(self) -> tuple[list[str], np.ndarray]:
        rel_paths: list[str] = []
        vectors: list[np.ndarray] = []
        expected_dim: Optional[int] = None

        with self.vector_txt.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    print(f"[WARN] Skip malformed line {line_idx}")
                    continue

                _, rel_path, vec_text = parts
                try:
                    vec = np.fromstring(vec_text, sep=",", dtype=np.float32)
                except Exception:
                    print(f"[WARN] Skip unparsable vector at line {line_idx}")
                    continue

                if expected_dim is None:
                    expected_dim = int(vec.shape[0])
                if vec.shape[0] != expected_dim:
                    print(f"[WARN] Skip inconsistent dim at line {line_idx}")
                    continue

                rel_paths.append(rel_path)
                vectors.append(vec)

        if not vectors:
            raise RuntimeError("No valid vectors loaded from txt.")

        feature_matrix = np.vstack(vectors).astype(np.float32)
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True) + 1e-12
        feature_matrix = feature_matrix / norms

        print(f"Loaded {len(rel_paths)} vectors, dim={feature_matrix.shape[1]}")
        return rel_paths, feature_matrix

    def _write_secure_index(self, rel_paths: list[str], feature_matrix: np.ndarray) -> None:
        memory_file = io.BytesIO()
        np.savez_compressed(memory_file, rel_paths=np.array(rel_paths), features=feature_matrix)
        payload = self._encrypt_bytes(memory_file.getvalue())
        self.encrypted_index.parent.mkdir(parents=True, exist_ok=True)
        self.encrypted_index.write_text(
            json.dumps(payload, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"[INFO] Encrypted index saved to: {self.encrypted_index}")

    def _read_secure_index(self) -> tuple[list[str], np.ndarray]:
        payload = json.loads(self.encrypted_index.read_text(encoding="utf-8"))
        plain_bytes = self._decrypt_bytes(payload)
        memory_file = io.BytesIO(plain_bytes)
        loaded = np.load(memory_file, allow_pickle=False)
        rel_paths = loaded["rel_paths"].astype(str).tolist()
        feature_matrix = loaded["features"].astype(np.float32)
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True) + 1e-12
        feature_matrix = feature_matrix / norms
        print(f"Loaded encrypted vectors: {len(rel_paths)} items, dim={feature_matrix.shape[1]}")
        return rel_paths, feature_matrix

    def _load_vectors_secure(self, rebuild_secure_index: bool) -> tuple[list[str], np.ndarray]:
        if self.encrypted_index.exists() and not rebuild_secure_index:
            return self._read_secure_index()

        rel_paths, feature_matrix = self._load_vectors()
        self._write_secure_index(rel_paths, feature_matrix)
        return rel_paths, feature_matrix

    def _embed_query(self, image: Image.Image) -> np.ndarray:
        image_bytes = self._pil_to_png_bytes(image)
        query_digest = hashlib.sha256(image_bytes).hexdigest()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if hashlib.sha256(image_bytes).hexdigest() != query_digest:
            raise RuntimeError("Query image integrity check failed.")
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(x).squeeze(-1).squeeze(-1).squeeze(0).cpu().numpy().astype(np.float32)
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        return feat

    def _pil_to_png_bytes(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()

    def search(self, image: Image.Image) -> list[tuple[Image.Image, str]]:
        if image is None:
            raise gr.Error("Please upload an image first.")

        q = self._embed_query(image)
        sims = self.features @ q

        topk = min(self.topk, len(self.relative_paths))
        top_idx = np.argpartition(-sims, kth=topk - 1)[:topk]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        gallery: list[tuple[Image.Image, str]] = []
        for rank, idx in enumerate(top_idx, start=1):
            rel_path = self.relative_paths[int(idx)]
            score = float(sims[int(idx)])
            image_path = self.image_root / Path(rel_path)

            if not image_path.exists():
                continue

            try:
                with Image.open(image_path) as img:
                    sample = img.convert("RGB").copy()
            except Exception:
                continue

            label = f"Top{rank} | sim={score:.4f} | {rel_path}"
            gallery.append((sample, label))
        if not gallery:
            raise gr.Error("Search failed: no readable matched images.")

        return gallery


def build_demo(engine: ImageSearchEngine) -> gr.Blocks:
    custom_css = """
    :root { --app-bg: #D1CABB; }
    html, body {
        background: var(--app-bg) !important;
        min-height: 100vh !important;
    }
    body {
        margin: 0 !important;
        background: var(--app-bg) !important;
    }
    /* Force the very outermost wrappers to match background */
    body > div,
    #root,
    #app,
    .app,
    .gradio-container,
    .wrap,
    .contain,
    .container {
        background: var(--app-bg) !important;
    }
    /* Ensure the app stretches full width (no white side gutters) */
    .gradio-container {
        max-width: none !important;
        width: 100% !important;
        min-height: 100vh !important;
    }
    /* Make Gradio blocks/panels match background (remove white boxes) */
    .gradio-container .block,
    .gradio-container .gr-box,
    .gradio-container .gr-panel,
    .gradio-container .gr-form,
    .gradio-container .form,
    .gradio-container .tabitem {
        background: var(--app-bg) !important;
    }
    /* Upload + Gallery background */
    .gradio-container .gr-image,
    .gradio-container .image-container,
    .gradio-container .image-frame,
    .gradio-container .gr-gallery,
    .gradio-container .gallery,
    .gradio-container .gallery-container,
    .gradio-container .gr-gallery .grid,
    .gradio-container .gr-gallery .grid-wrap,
    .gradio-container .gr-gallery .gallery-item,
    .gradio-container .gr-gallery .thumbnail-item,
    .gradio-container .gr-gallery .thumbnail {
        background: var(--app-bg) !important;
    }
    /* Keep inputs readable */
    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select {
        background: #ffffff !important;
    }
    #login_card {
        max-width: 520px;
        margin: 24px auto;
        padding: 18px 20px 12px 20px;
        border: 1px solid #d9d9d9;
        border-radius: 12px;
        background: var(--app-bg) !important;
    }
    #login_card .block,
    #login_card .gr-box,
    #login_card .gr-panel,
    #login_card .gr-form,
    #login_card .form {
        background: var(--app-bg) !important;
    }
    #main_container {
        max-width: 1280px;
        margin: 0 auto;
    }
    """
    with gr.Blocks(title="以图搜图系统", css=custom_css) as demo:
        login_panel = gr.Column(visible=True, elem_id="login_card")
        main_panel = gr.Column(visible=False)

        with login_panel:
            gr.Markdown("<h2 style='text-align:center; margin-bottom: 10px;'>图像检索系统</h2>")
            login_status = gr.Markdown("")
            username = gr.Textbox(label="用户名", placeholder="请输入用户名")
            password = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
            login_btn = gr.Button("登录", variant="primary")

        with main_panel:
            with gr.Column(elem_id="main_container"):
                gr.Markdown("<h2 style='text-align:center;'>图像检索系统</h2>")
                gr.Markdown(
                    "<div style='text-align:center;'>"
                    "安全模块说明：1) SHA-256 向量文件完整性校验；"
                    "2) AES-256-GCM 向量索引加密存储/解密加载；"
                    "3) 基于环境变量的密钥管理（IMAGE_SEARCH_AES_KEY）。"
                    "</div>"
                )
                query_image = gr.Image(type="pil", label="上传查询图片")
                search_btn = gr.Button("开始检索", variant="primary")
                gallery = gr.Gallery(
                    label="Top10 相似图片",
                    columns=5,
                    rows=2,
                    height=560,
                )

        def handle_login(input_user: str, input_pass: str) -> tuple[str, dict, dict]:
            expected_user = os.getenv("IMAGE_SEARCH_USERNAME", "admin")
            expected_pass = os.getenv("IMAGE_SEARCH_PASSWORD", "123456")
            if input_user == expected_user and input_pass == expected_pass:
                return "登录成功。", gr.update(visible=False), gr.update(visible=True)
            return "用户名或密码错误，请重试。", gr.update(visible=True), gr.update(visible=False)

        login_btn.click(
            fn=handle_login,
            inputs=[username, password],
            outputs=[login_status, login_panel, main_panel],
        )

        search_btn.click(fn=engine.search, inputs=[query_image], outputs=[gallery])

    return demo


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("topk must be > 0")

    engine = ImageSearchEngine(
        image_root=args.image_root,
        vector_txt=args.vector_txt,
        encrypted_index=args.encrypted_index,
        vector_sha256=args.vector_sha256,
        topk=args.topk,
        rebuild_secure_index=args.rebuild_secure_index,
    )
    demo = build_demo(engine)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
