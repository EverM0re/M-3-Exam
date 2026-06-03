from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional


_IMG_LITERAL_RE = re.compile(r"\bimg[_-]?\d+(?:\.[a-zA-Z0-9]+)?\b", re.IGNORECASE)


_CLASSIFY_PROMPT_TEMPLATE = (
    "You decide which modality a memory-QA question depends on the most.\n"
    "Reply ONLY with one valid JSON object in this exact form:\n"
    '{"needs_image": <true|false>, "needs_pdf": <true|false>, "needs_chart": <true|false>}\n'
    "Rules:\n"
    "- needs_image = true if the answer must come from inspecting a photo or screenshot.\n"
    "- needs_pdf   = true if the question asks about a paper/document/file/report content.\n"
    "- needs_chart = true if the question asks about numbers, percentages, axes, bars, trends, tables.\n"
    "- Yes/No questions about visible content also count as needs_image=true.\n"
    "- Multiple flags can be true simultaneously.\n"
    "- If purely about prior conversation text (no figures or files), all three are false.\n\n"
    "Question: <<QUESTION>>\n"
    "JSON:"
)


def _build_classify_prompt(question: str) -> str:
    return _CLASSIFY_PROMPT_TEMPLATE.replace("<<QUESTION>>", question or "")


def _hash_key(dataset: str, question: str) -> str:
    h = hashlib.md5(f"{dataset}||{question}".encode("utf-8")).hexdigest()[:16]
    return h


def _heuristic(question: str) -> Optional[Dict[str, bool]]:
    if _IMG_LITERAL_RE.search(question or ""):
        return {"needs_image": True, "needs_pdf": False, "needs_chart": False}
    return None


def _parse_json_like(raw: str) -> Dict[str, bool]:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {
                "needs_image": bool(obj.get("needs_image", False)),
                "needs_pdf": bool(obj.get("needs_pdf", False)),
                "needs_chart": bool(obj.get("needs_chart", False)),
            }
    except Exception:
        pass
    low = raw.lower()
    return {
        "needs_image": '"needs_image": true' in low or "needs_image: true" in low,
        "needs_pdf": '"needs_pdf": true' in low or "needs_pdf: true" in low,
        "needs_chart": '"needs_chart": true' in low or "needs_chart: true" in low,
    }


class ModalityClassifier:
    _DEFAULT = {"needs_image": False, "needs_pdf": False, "needs_chart": False}

    def __init__(self, llm, cache_path: Optional[Path] = None, dataset_name: str = ""):
        self.llm = llm
        self.dataset_name = dataset_name
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, bool]] = {}
        if cache_path and cache_path.is_file():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def classify(self, question: str) -> Dict[str, bool]:
        key = _hash_key(self.dataset_name, question)
        if key in self.cache:
            return self.cache[key]

        try:
            h = _heuristic(question)
            if h is not None:
                self.cache[key] = h
                return h

            prompt = _build_classify_prompt(question)
            try:
                text, _u = self.llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=40,
                )
            except Exception as e:
                logging.getLogger("m3proctor").warning(
                    "modality-classifier LLM call failed: %s", e
                )
                text = ""
            result = _parse_json_like(text)
        except Exception:
            logging.getLogger("m3proctor").exception(
                "modality-classifier crashed on question=%r - fallback to all-False",
                question[:80],
            )
            result = dict(self._DEFAULT)
        self.cache[key] = result
        return result

    def flush(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
