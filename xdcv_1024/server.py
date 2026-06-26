import argparse
import base64
import cgi
import hmac
import json
import os
import shutil
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runs"
SAMPLE_DIR = ROOT_DIR / "demo"
MAX_UPLOAD_BYTES = int(os.environ.get("XDCV_MAX_UPLOAD_MB", "25")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SAMPLE_IMAGES = ["demo14.jpg", "demo15.jpg", "demo_hf.jpg"]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def auth_credentials() -> tuple[str, str] | None:
    username = os.environ.get("XDCV_AUTH_USER", "").strip()
    password = os.environ.get("XDCV_AUTH_PASSWORD", "")
    if not username or not password:
        return None
    return username, password


def safe_relative_path(base_dir: Path, relative_url_path: str) -> Path:
    relative_path = Path(unquote(relative_url_path).lstrip("/"))
    resolved = (base_dir / relative_path).resolve()
    if base_dir.resolve() not in resolved.parents and resolved != base_dir.resolve():
        raise ValueError("Unsafe path")
    return resolved


def content_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".js":
        return "application/javascript; charset=utf-8"
    return "application/octet-stream"


def serve_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.exists() or not path.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND, "File not found")
        return

    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type_for(path))
    handler.send_header("Content-Length", str(path.stat().st_size))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    with path.open("rb") as handle:
        shutil.copyfileobj(handle, handler.wfile)


def save_upload(file_item, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("wb") as output:
        while True:
            chunk = file_item.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise ValueError(f"图片超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 限制")
            output.write(chunk)

    if written == 0:
        raise ValueError("上传文件为空")


class CaptionEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._processor = None

    def _load(self):
        from image_caption_infer.local_infer_seal import (
            DEFAULT_ADAPTER,
            _resolve_base_model,
            load_model,
        )

        adapter_path = Path(os.environ.get("XDCV_CAPTION_ADAPTER", str(DEFAULT_ADAPTER))).expanduser()
        base_model = os.environ.get("XDCV_CAPTION_BASE_MODEL")
        base_model_path = _resolve_base_model(base_model, adapter_path)
        device = os.environ.get("XDCV_CAPTION_DEVICE", os.environ.get("XDCV_DEVICE", "auto"))
        dtype = os.environ.get("XDCV_CAPTION_DTYPE", "auto")
        min_pixels = int(os.environ.get("XDCV_CAPTION_MIN_PIXELS", str(256 * 28 * 28)))
        max_pixels = int(os.environ.get("XDCV_CAPTION_MAX_PIXELS", str(1024 * 1024)))
        local_files_only = env_bool("XDCV_CAPTION_LOCAL_FILES_ONLY", False)

        self._model, self._processor = load_model(
            base_model_path=base_model_path,
            adapter_path=str(adapter_path),
            device=device,
            dtype=dtype,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            local_files_only=local_files_only,
        )

    def infer(self, image_path: Path, prompt: str, max_new_tokens: int) -> str:
        from image_caption_infer.local_infer_seal import DEFAULT_PROMPT, infer

        with self._lock:
            if self._model is None or self._processor is None:
                self._load()
            return infer(
                model=self._model,
                processor=self._processor,
                image_path=str(image_path),
                prompt=prompt.strip() or DEFAULT_PROMPT,
                max_new_tokens=max_new_tokens,
            )


class PrintEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._device = None
        self._device_name = None

    def _load(self, device_name: str):
        from itr_pe.Inference import DEFAULT_MODEL_PATH, load_model, select_device

        model_path = Path(os.environ.get("XDCV_PRINT_MODEL", str(DEFAULT_MODEL_PATH))).expanduser()
        device = select_device(device_name)
        self._model = load_model(model_path, device)
        self._device = device
        self._device_name = device_name

    def infer(
        self,
        image_path: Path,
        mask_path: Path,
        cutout_path: Path,
        input_size: int,
        threshold: int,
        fill_holes: bool = True,
    ) -> tuple[Path, Path]:
        from itr_pe.Inference import infer_image
        from itr_pe.apply_mask import apply_alpha_mask

        device_name = os.environ.get("XDCV_PRINT_DEVICE", os.environ.get("XDCV_DEVICE", "auto"))
        with self._lock:
            if self._model is None or self._device_name != device_name:
                self._load(device_name)
            infer_image(self._model, image_path, mask_path, self._device, input_size)
            apply_alpha_mask(image_path, mask_path, cutout_path, threshold=threshold, fill_holes=fill_holes)
        return mask_path, cutout_path


CAPTION_ENGINE = CaptionEngine()
PRINT_ENGINE = PrintEngine()


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XDCV-1024</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f5f7;
      --ink: #171a1f;
      --muted: #667085;
      --line: #d9dee7;
      --panel: #ffffff;
      --teal: #117b74;
      --teal-dark: #0d5f5a;
      --amber: #c48211;
      --red: #b94b43;
      --shadow: 0 16px 50px rgba(17, 24, 39, 0.10);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
    }
    button, input, textarea, select { font: inherit; }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .topbar {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 20px;
      padding: 18px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      position: sticky;
      top: 0;
      z-index: 20;
      backdrop-filter: blur(14px);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      grid-column: 2;
      justify-self: center;
      min-width: 0;
    }
    .mark {
      width: 40px;
      height: 40px;
      display: grid;
      place-items: center;
      background: #111827;
      color: #fff;
      border-radius: 8px;
      font-weight: 800;
      letter-spacing: 0;
    }
    .brand h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.05;
      letter-spacing: 0;
    }
    .brand p {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .status-pill {
      display: flex;
      align-items: center;
      gap: 8px;
      grid-column: 3;
      justify-self: end;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--teal);
    }
    .workspace {
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 24px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .controls {
      padding: 18px;
      display: grid;
      gap: 18px;
    }
    .section-title {
      margin: 0 0 10px;
      font-size: 13px;
      color: var(--muted);
      font-weight: 700;
    }
    .mode-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .mode-btn {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 12px;
      text-align: left;
      cursor: pointer;
      min-height: 76px;
    }
    .mode-btn strong {
      display: block;
      font-size: 15px;
      margin-bottom: 6px;
    }
    .mode-btn span {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .mode-btn.active {
      border-color: var(--teal);
      box-shadow: inset 0 0 0 1px var(--teal);
      background: #f4fbfa;
    }
    .dropzone {
      border: 1px dashed #aab3c2;
      border-radius: 8px;
      background: #fbfcfd;
      min-height: 210px;
      display: grid;
      place-items: center;
      overflow: hidden;
      position: relative;
      cursor: pointer;
    }
    .dropzone.dragging {
      border-color: var(--teal);
      background: #eef8f7;
    }
    .drop-copy {
      display: grid;
      gap: 8px;
      text-align: center;
      color: var(--muted);
      padding: 24px;
    }
    .drop-copy strong {
      color: var(--ink);
      font-size: 15px;
    }
    .dropzone img {
      width: 100%;
      height: 100%;
      max-height: 280px;
      object-fit: contain;
      background: #eef1f5;
      display: none;
    }
    .dropzone.has-image .drop-copy { display: none; }
    .dropzone.has-image img { display: block; }
    input[type="file"] { display: none; }
    .sample-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    .sample-btn {
      padding: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      height: 70px;
      cursor: pointer;
      background: #fff;
    }
    .sample-btn img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    label.field {
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }
    textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 11px 12px;
      outline: none;
    }
    textarea {
      min-height: 92px;
      resize: vertical;
      line-height: 1.5;
      font-weight: 500;
    }
    textarea:focus, select:focus {
      border-color: var(--teal);
      box-shadow: 0 0 0 3px rgba(17, 123, 116, 0.15);
    }
    .control-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .primary {
      border: 0;
      border-radius: 8px;
      background: var(--teal);
      color: #fff;
      padding: 13px 16px;
      font-weight: 800;
      cursor: pointer;
      min-height: 46px;
    }
    .primary:hover { background: var(--teal-dark); }
    .primary:disabled {
      background: #95aaa8;
      cursor: not-allowed;
    }
    .results {
      min-height: calc(100vh - 122px);
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }
    .results-head {
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
    }
    .results-head h2 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .result-body {
      padding: 20px;
      display: grid;
      gap: 18px;
    }
    .empty {
      min-height: 520px;
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      color: var(--muted);
      text-align: center;
      padding: 28px;
    }
    .image-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    .image-grid.print-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .result-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    .result-card header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }
    .result-card header strong { font-size: 14px; }
    .result-card a {
      color: var(--teal);
      text-decoration: none;
      font-size: 13px;
      font-weight: 700;
    }
    .result-card img {
      width: 100%;
      height: min(58vh, 620px);
      object-fit: contain;
      display: block;
      background:
        linear-gradient(45deg, #e8ecf1 25%, transparent 25%),
        linear-gradient(-45deg, #e8ecf1 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #e8ecf1 75%),
        linear-gradient(-45deg, transparent 75%, #e8ecf1 75%);
      background-size: 22px 22px;
      background-position: 0 0, 0 11px, 11px -11px, -11px 0;
    }
    .caption-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 18px;
      font-size: 18px;
      line-height: 1.7;
      color: #1f2937;
    }
    .error {
      border-left: 4px solid var(--red);
      padding: 12px 14px;
      background: #fff5f4;
      color: #7a271f;
      border-radius: 8px;
      display: none;
      line-height: 1.5;
    }
    .busy {
      display: none;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .spinner {
      width: 18px;
      height: 18px;
      border: 2px solid #c7d7d5;
      border-top-color: var(--teal);
      border-radius: 999px;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (max-width: 980px) {
      .workspace {
        grid-template-columns: 1fr;
        padding: 16px;
      }
      .results { min-height: 560px; }
      .image-grid { grid-template-columns: 1fr; }
      .image-grid.print-grid { grid-template-columns: 1fr; }
      .topbar {
        grid-template-columns: 1fr;
        padding: 14px 16px;
      }
      .brand {
        grid-column: 1;
        justify-self: center;
      }
      .status-pill { display: none; }
    }
    @media (max-width: 560px) {
      .mode-grid, .control-row, .sample-row { grid-template-columns: 1fr; }
      .sample-btn { height: 90px; }
      .brand h1 { font-size: 19px; }
      .brand p { font-size: 12px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <div class="mark">XD</div>
        <div>
          <h1>XDCV-1024</h1>
          <p>本地视觉推理工作台</p>
        </div>
      </div>
      <div class="status-pill"><span class="dot"></span><span id="serverState">Local Ready</span></div>
    </header>

    <main class="workspace">
      <aside class="panel controls">
        <section>
          <p class="section-title">任务</p>
          <div class="mode-grid">
            <button class="mode-btn active" type="button" data-mode="caption">
              <strong>图生文</strong>
              <span>Qwen2-VL + SEAL Adapter</span>
            </button>
            <button class="mode-btn" type="button" data-mode="print">
              <strong>印花提取</strong>
              <span>IS-Net Mask + RGBA</span>
            </button>
          </div>
        </section>

        <section>
          <p class="section-title">图片</p>
          <label class="dropzone" id="dropzone">
            <input id="fileInput" type="file" accept="image/png,image/jpeg">
            <div class="drop-copy">
              <strong>选择图片</strong>
              <span>JPG / PNG</span>
            </div>
            <img id="preview" alt="preview">
          </label>
        </section>

        <section>
          <p class="section-title">样例</p>
          <div class="sample-row">
            <button class="sample-btn" type="button" data-sample="demo14.jpg"><img src="/samples/demo14.jpg" alt="demo14"></button>
            <button class="sample-btn" type="button" data-sample="demo15.jpg"><img src="/samples/demo15.jpg" alt="demo15"></button>
            <button class="sample-btn" type="button" data-sample="demo_hf.jpg"><img src="/samples/demo_hf.jpg" alt="demo_hf"></button>
          </div>
        </section>

        <section id="captionControls">
          <label class="field">提示词
            <textarea id="prompt">Describe the fashion items in the image concisely in English. Format: A simple plain paragraph. Do NOT output conversational filler.</textarea>
          </label>
        </section>

        <section>
          <div class="control-row">
            <label class="field" id="tokenControl">输出长度
              <select id="maxTokens">
                <option value="96">96</option>
                <option value="128" selected>128</option>
                <option value="192">192</option>
                <option value="256">256</option>
              </select>
            </label>
            <label class="field" id="sizeControl" style="display:none">推理尺寸
              <select id="inputSize">
                <option value="512">512</option>
                <option value="768">768</option>
                <option value="1024" selected>1024</option>
              </select>
            </label>
            <label class="field" id="thresholdControl" style="display:none">Mask 阈值
              <select id="threshold">
                <option value="96">96</option>
                <option value="127" selected>127</option>
                <option value="160">160</option>
              </select>
            </label>
          </div>
        </section>

        <button class="primary" id="runButton" type="button">开始推理</button>
        <div class="busy" id="busy"><span class="spinner"></span><span>处理中</span></div>
        <div class="error" id="errorBox"></div>
      </aside>

      <section class="panel results">
        <header class="results-head">
          <h2>结果</h2>
          <span class="meta" id="resultMeta">等待输入</span>
        </header>
        <div class="result-body" id="resultBody">
          <div class="empty">
            <div>
              <strong>选择任务并上传图片</strong>
              <p>结果会显示在这里。</p>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const state = { mode: 'caption', file: null, previewUrl: null };
    const modeButtons = document.querySelectorAll('.mode-btn');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const runButton = document.getElementById('runButton');
    const busy = document.getElementById('busy');
    const errorBox = document.getElementById('errorBox');
    const resultBody = document.getElementById('resultBody');
    const resultMeta = document.getElementById('resultMeta');
    const captionControls = document.getElementById('captionControls');
    const tokenControl = document.getElementById('tokenControl');
    const sizeControl = document.getElementById('sizeControl');
    const thresholdControl = document.getElementById('thresholdControl');

    function setMode(mode) {
      state.mode = mode;
      modeButtons.forEach((button) => button.classList.toggle('active', button.dataset.mode === mode));
      captionControls.style.display = mode === 'caption' ? 'block' : 'none';
      tokenControl.style.display = mode === 'caption' ? 'grid' : 'none';
      sizeControl.style.display = mode === 'print' ? 'grid' : 'none';
      thresholdControl.style.display = mode === 'print' ? 'grid' : 'none';
      resultMeta.textContent = mode === 'caption' ? '图生文' : '印花提取';
    }

    function showError(message) {
      errorBox.textContent = message;
      errorBox.style.display = 'block';
    }

    function clearError() {
      errorBox.textContent = '';
      errorBox.style.display = 'none';
    }

    function setBusy(active) {
      runButton.disabled = active;
      busy.style.display = active ? 'flex' : 'none';
    }

    function setPreview(file) {
      state.file = file;
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = URL.createObjectURL(file);
      preview.src = state.previewUrl;
      dropzone.classList.add('has-image');
      resultMeta.textContent = file.name;
    }

    function renderCaption(data) {
      resultBody.innerHTML = `
        <div class="image-grid">
          <article class="result-card">
            <header><strong>输入图像</strong><a href="${data.input_url}" target="_blank">打开</a></header>
            <img src="${data.input_url}" alt="input">
          </article>
          <article class="result-card">
            <header><strong>图生文结果</strong></header>
            <div class="caption-box">${escapeHtml(data.caption || '')}</div>
          </article>
        </div>
      `;
      resultMeta.textContent = `${data.elapsed_seconds.toFixed(2)}s`;
    }

    function renderPrint(data) {
      resultBody.innerHTML = `
        <div class="image-grid print-grid">
          <article class="result-card">
            <header><strong>原图</strong><a href="${data.input_url}" target="_blank">打开</a></header>
            <img src="${data.input_url}" alt="input">
          </article>
          <article class="result-card">
            <header><strong>灰度 Mask</strong><a href="${data.mask_url}" target="_blank">打开</a></header>
            <img src="${data.mask_url}" alt="mask">
          </article>
          <article class="result-card">
            <header><strong>印花抠图</strong><a href="${data.cutout_url}" target="_blank">打开</a></header>
            <img src="${data.cutout_url}" alt="cutout">
          </article>
        </div>
      `;
      resultMeta.textContent = `${data.elapsed_seconds.toFixed(2)}s`;
    }

    function escapeHtml(value) {
      return value
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }

    async function runInference() {
      clearError();
      if (!state.file) {
        showError('请先选择 JPG 或 PNG 图片。');
        return;
      }

      const form = new FormData();
      form.append('mode', state.mode);
      form.append('image', state.file, state.file.name);
      form.append('prompt', document.getElementById('prompt').value);
      form.append('max_new_tokens', document.getElementById('maxTokens').value);
      form.append('input_size', document.getElementById('inputSize').value);
      form.append('threshold', document.getElementById('threshold').value);

      setBusy(true);
      try {
        const response = await fetch('/api/infer', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok || data.status !== 'ok') {
          throw new Error(data.error || '推理失败');
        }
        if (data.mode === 'caption') renderCaption(data);
        if (data.mode === 'print') renderPrint(data);
      } catch (error) {
        showError(error.message || String(error));
      } finally {
        setBusy(false);
      }
    }

    modeButtons.forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
    fileInput.addEventListener('change', () => {
      const file = fileInput.files && fileInput.files[0];
      if (file) setPreview(file);
    });
    ['dragenter', 'dragover'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add('dragging');
      });
    });
    ['dragleave', 'drop'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove('dragging');
      });
    });
    dropzone.addEventListener('drop', (event) => {
      const file = event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) setPreview(file);
    });
    document.querySelectorAll('.sample-btn').forEach((button) => {
      button.addEventListener('click', async () => {
        clearError();
        const name = button.dataset.sample;
        const response = await fetch(`/samples/${name}`);
        const blob = await response.blob();
        setPreview(new File([blob], name, { type: blob.type || 'image/jpeg' }));
      });
    });
    runButton.addEventListener('click', runInference);
  </script>
</body>
</html>
"""


class XDCVHandler(BaseHTTPRequestHandler):
    server_version = "XDCV1024/0.1"

    def log_message(self, format: str, *args) -> None:
        print("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))

    def is_authorized(self) -> bool:
        credentials = auth_credentials()
        if credentials is None:
            return True

        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Basic "):
            return False

        try:
            decoded = base64.b64decode(auth_header.removeprefix("Basic ").strip()).decode("utf-8")
        except Exception:
            return False

        username, separator, password = decoded.partition(":")
        if not separator:
            return False

        expected_username, expected_password = credentials
        return hmac.compare_digest(username, expected_username) and hmac.compare_digest(password, expected_password)

    def require_authorization(self) -> bool:
        if self.is_authorized():
            return True

        body = b"Authentication required"
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", 'Basic realm="XDCV-1024"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return False

    def do_GET(self) -> None:
        if not self.require_authorization():
            return

        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_index()
            return
        if parsed.path == "/api/health":
            json_response(self, {"status": "ok", "name": "XDCV-1024"})
            return
        if parsed.path.startswith("/runs/"):
            try:
                serve_file(self, safe_relative_path(RUNS_DIR, parsed.path.removeprefix("/runs/")))
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if parsed.path.startswith("/samples/"):
            name = Path(parsed.path).name
            if name not in SAMPLE_IMAGES:
                self.send_error(HTTPStatus.NOT_FOUND, "Sample not found")
                return
            serve_file(self, SAMPLE_DIR / name)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if not self.require_authorization():
            return

        parsed = urlparse(self.path)
        if parsed.path == "/api/infer":
            self.handle_infer()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def serve_index(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_infer(self) -> None:
        started = time.time()
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )
            mode = form.getfirst("mode", "caption")
            if mode not in {"caption", "print"}:
                raise ValueError("未知任务类型")
            if "image" not in form:
                raise ValueError("没有收到图片")

            file_item = form["image"]
            filename = Path(file_item.filename or "upload.png").name
            extension = Path(filename).suffix.lower()
            if extension not in ALLOWED_EXTENSIONS:
                raise ValueError("仅支持 JPG 和 PNG 图片")

            job_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            job_dir = RUNS_DIR / job_id
            input_dir = job_dir / "input"
            result_dir = job_dir / "result"
            input_path = input_dir / f"source{extension}"
            save_upload(file_item, input_path)

            if mode == "caption":
                response = self.run_caption(form, input_path, job_id, started)
            else:
                response = self.run_print(form, input_path, result_dir, job_id, started)
            json_response(self, response)
        except Exception as exc:
            json_response(self, {"status": "error", "error": str(exc)}, status=500)

    def run_caption(self, form, input_path: Path, job_id: str, started: float) -> dict:
        prompt = form.getfirst("prompt", "")
        max_new_tokens = int(form.getfirst("max_new_tokens", "128"))
        max_new_tokens = max(1, min(max_new_tokens, 512))
        caption = CAPTION_ENGINE.infer(input_path, prompt, max_new_tokens)
        return {
            "status": "ok",
            "mode": "caption",
            "caption": caption,
            "input_url": f"/runs/{job_id}/input/{input_path.name}",
            "elapsed_seconds": time.time() - started,
        }

    def run_print(self, form, input_path: Path, result_dir: Path, job_id: str, started: float) -> dict:
        input_size = int(form.getfirst("input_size", "1024"))
        threshold = int(form.getfirst("threshold", "127"))
        input_size = max(256, min(input_size, 1536))
        threshold = max(0, min(threshold, 255))
        mask_path = result_dir / "mask.png"
        cutout_path = result_dir / "cutout.png"
        PRINT_ENGINE.infer(input_path, mask_path, cutout_path, input_size, threshold, fill_holes=True)
        return {
            "status": "ok",
            "mode": "print",
            "input_url": f"/runs/{job_id}/input/{input_path.name}",
            "mask_url": f"/runs/{job_id}/result/mask.png",
            "cutout_url": f"/runs/{job_id}/result/cutout.png",
            "elapsed_seconds": time.time() - started,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the XDCV-1024 local web demo.")
    parser.add_argument("--host", default=os.environ.get("XDCV_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("XDCV_PORT", "7860")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), XDCVHandler)
    print(f"XDCV-1024 running at http://{args.host}:{args.port}")
    if auth_credentials() is None:
        print("Warning: XDCV_AUTH_USER/XDCV_AUTH_PASSWORD is not set. The service has no login protection.")
    else:
        print("Basic authentication is enabled.")
    server.serve_forever()


if __name__ == "__main__":
    main()
