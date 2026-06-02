from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional

import openai

from m3exam.config.config_loader import cfg
from m3exam.global_methods import run_chatgpt

logger = logging.getLogger(__name__)


def _load_image_b64(path: str, max_side_px: int) -> tuple[str, str]:
    with open(path, "rb") as fh:
        raw = fh.read()

    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"

    if max_side_px <= 0:
        return mime, base64.b64encode(raw).decode("utf-8")

    try:
        from PIL import Image  # type: ignore
    except ImportError:
        logger.warning(
            "Pillow not installed; sending full-resolution images. "
            "Install with: pip install Pillow  (enables max_image_side_px)"
        )
        return mime, base64.b64encode(raw).decode("utf-8")

    try:
        im = Image.open(io.BytesIO(raw))
        try:
            resample = Image.Resampling.LANCZOS  # Pillow >= 9.1
        except AttributeError:
            resample = Image.LANCZOS  # type: ignore[attr-defined]
        im.thumbnail((max_side_px, max_side_px), resample)
        buf = io.BytesIO()
        if im.mode in ("RGBA", "P"):
            im = im.convert("RGBA")
            im.save(buf, format="PNG")
            mime = "image/png"
        else:
            im = im.convert("RGB")
            im.save(buf, format="JPEG", quality=85)
            mime = "image/jpeg"
        return mime, base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        logger.warning("Image resize failed for %s (%s); sending original bytes.", path, exc)
        return mime, base64.b64encode(raw).decode("utf-8")


def call_vision_llm(
    prompt: str,
    image_paths: List[str],
    max_tokens: int = 300,
    *,
    require_vision: Optional[bool] = None,
) -> str:
    if require_vision is None:
        require_vision = bool(cfg("api", "vision_llm", "require_vision", default=False))

    max_side = int(cfg("api", "vision_llm", "max_image_side_px", default=2048) or 0)

    api_key = cfg("api", "vision_llm", "api_key") or ""
    base_url = cfg("api", "vision_llm", "base_url") or ""
    model = cfg("api", "vision_llm", "model") or "gpt-4.1-mini"

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    loaded = 0

    for img_path in image_paths:
        if not os.path.exists(img_path):
            msg = f"Image not found: {img_path}"
            if require_vision:
                raise FileNotFoundError(msg)
            logger.warning(msg)
            continue
        try:
            mime, b64 = _load_image_b64(img_path, max_side)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
            loaded += 1
        except FileNotFoundError:
            raise
        except Exception as exc:
            if require_vision:
                raise RuntimeError(f"Cannot read image {img_path}: {exc}") from exc
            logger.warning("Cannot read image %s: %s", img_path, exc)

    if loaded == 0:
        if require_vision:
            raise ValueError("require_vision=True but no images could be loaded for the vision call.")
        logger.warning("No images loaded - falling back to text LLM.")
        return run_chatgpt(prompt, 1, max_tokens, "chatgpt").strip()

    if not api_key:
        if require_vision:
            raise ValueError("Vision LLM api_key not configured (require_vision=True).")
        logger.warning("Vision LLM API key not configured - falling back to text LLM.")
        return run_chatgpt(prompt, 1, max_tokens, "chatgpt").strip()

    try:
        client = (
            openai.OpenAI(api_key=api_key, base_url=base_url)
            if base_url
            else openai.OpenAI(api_key=api_key)
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            n=1,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        if require_vision:
            raise RuntimeError(f"Vision LLM call failed: {exc}") from exc
        logger.error("Vision LLM call failed (%s) - falling back to text LLM.", exc)
        return run_chatgpt(prompt, 1, max_tokens, "chatgpt").strip()
