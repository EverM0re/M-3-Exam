#!/usr/bin/env python3

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import string
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MAX_IMAGES_PER_QUESTION = 10

import openai

from m3exam.config.config_loader import cfg
from m3exam.global_methods import run_chatgpt, set_openai_key

logger = logging.getLogger(__name__)


def _eval_cfg(*keys, default=None):
    return cfg("evaluation", *keys, default=default)


def _make_eval_client() -> openai.OpenAI:
    api_key  = _eval_cfg("evaluation_model", "api_key")  or ""
    base_url = _eval_cfg("evaluation_model", "base_url") or ""
    if not api_key:
        raise ValueError("evaluation.evaluation_model.api_key is not configured.")
    return (
        openai.OpenAI(api_key=api_key, base_url=base_url)
        if base_url
        else openai.OpenAI(api_key=api_key)
    )


def _eval_model_name() -> str:
    return _eval_cfg("evaluation_model", "model") or "gpt-4o"


_judge_client: Optional[openai.OpenAI] = None
_judge_model_name_cached: Optional[str] = None


def _make_judge_client() -> openai.OpenAI:
    api_key  = _eval_cfg("judge_model", "api_key")  or ""
    base_url = _eval_cfg("judge_model", "base_url") or ""
    if not api_key:
        raise ValueError("evaluation.judge_model.api_key is not configured.")
    return (
        openai.OpenAI(api_key=api_key, base_url=base_url)
        if base_url
        else openai.OpenAI(api_key=api_key)
    )


def _judge_model_name() -> str:
    return _eval_cfg("judge_model", "model") or "gpt-4o"


def _get_judge_client() -> Tuple[openai.OpenAI, str]:
    global _judge_client, _judge_model_name_cached
    if _judge_client is None:
        _judge_client = _make_judge_client()
        _judge_model_name_cached = _judge_model_name()
    return _judge_client, _judge_model_name_cached  # type: ignore[return-value]


def _find_sessions_json(data_dir: Path) -> Path:
    direct = data_dir / "sessions.json"
    if direct.is_file():
        return direct
    for child in sorted(data_dir.iterdir()):
        if child.is_dir():
            candidate = child / "sessions.json"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"sessions.json not found in {data_dir} or its immediate sub-directories."
    )


def load_sessions(data_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Path]:
    path = _find_sessions_json(data_dir)
    with open(path, encoding="utf-8") as f:
        sessions: List[Dict[str, Any]] = json.load(f)
    return {str(s["session_id"]): s for s in sessions if "session_id" in s}, path


def _find_question_json(question_path: Path) -> Path:
    if question_path.suffix.lower() == ".json":
        if not question_path.is_file():
            raise FileNotFoundError(f"question file not found: {question_path}")
        return question_path
    candidate = question_path / "question.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"question.json not found in: {question_path}")
    return candidate


def load_questions(question_dir: Path) -> List[Dict[str, Any]]:
    path = _find_question_json(question_dir)
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r",\s*([\]\}])", r"\1", raw)
    return json.loads(raw)


_ROUND_RE = re.compile(r"([A-Z]\d+):\d+")


def parse_session_ids(supporting_facts: str) -> List[str]:
    seen = set()
    ids: List[str] = []
    for sid in _ROUND_RE.findall(supporting_facts):
        if sid not in seen:
            seen.add(sid)
            ids.append(sid)
    return ids


def _format_session_text(session: Dict[str, Any]) -> str:
    sid  = session.get("session_id", "?")
    date = session.get("date", "")
    header = f"=== Session {sid}" + (f" ({date})" if date else "") + " ==="
    lines = [header]
    for dlg in session.get("dialogues", []):
        rnd = dlg.get("round", "")
        lines.append(f"\n[{rnd}]")
        lines.append(f"User      : {dlg.get('user', '')}")
        lines.append(f"Assistant : {dlg.get('assistant', '')}")
    return "\n".join(lines)


def build_context_text(sessions_needed: List[Dict[str, Any]]) -> str:
    return "\n\n".join(_format_session_text(s) for s in sessions_needed)


def build_round_blocks(sessions_needed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for sess in sessions_needed:
        sid = sess.get("session_id", "?")
        date = sess.get("date", "")
        header = f"=== Session {sid}" + (f" ({date})" if date else "") + " ==="
        first = True
        for dlg in sess.get("dialogues", []):
            rnd = dlg.get("round", "")
            body = (
                f"[{rnd}]\n"
                f"User      : {dlg.get('user', '')}\n"
                f"Assistant : {dlg.get('assistant', '')}"
            )
            blocks.append({
                "sid": sid,
                "round": rnd,
                "is_head": first,
                "header": header,
                "body": body,
            })
            first = False
    return blocks


def render_blocks(blocks: List[Dict[str, Any]]) -> str:
    pieces: List[str] = []
    last_sid: Optional[str] = None
    for b in blocks:
        if b["sid"] != last_sid:
            pieces.append(b["header"])
            last_sid = b["sid"]
        pieces.append(b["body"])
    return "\n\n".join(pieces)


def must_keep_round_ids(supporting_facts: str) -> set:
    return set(re.findall(r"[A-Za-z]\d+:\d+", supporting_facts or ""))


def drop_oldest_blocks(
    blocks: List[Dict[str, Any]],
    must_keep: set,
    n_drop: int,
) -> List[Dict[str, Any]]:
    if n_drop <= 0 or not blocks:
        return blocks
    optional_idx = [i for i, b in enumerate(blocks) if b["round"] not in must_keep]
    drop_set = set(optional_idx[:n_drop])
    remaining = n_drop - len(drop_set)
    if remaining > 0:
        # Force-drop oldest cited rounds - note this may degrade grounding.
        for i in range(len(blocks)):
            if remaining <= 0:
                break
            if i not in drop_set:
                drop_set.add(i)
                remaining -= 1
    return [b for i, b in enumerate(blocks) if i not in drop_set]


def collect_session_images(
    sessions_needed: List[Dict[str, Any]],
    images_dir: Path,
) -> List[str]:
    seen: set = set()
    paths: List[str] = []
    for sess in sessions_needed:
        for dlg in sess.get("dialogues", []):
            for fn in (dlg.get("image_file") or []) + (dlg.get("img_file") or []):
                fn = str(fn).strip()
                if fn and fn not in seen:
                    seen.add(fn)
                    full = str(images_dir / fn)
                    paths.append(full)
    return paths


def collect_session_pdfs(
    sessions_needed: List[Dict[str, Any]],
    pdfs_dir: Path,
) -> List[str]:
    seen: set = set()
    paths: List[str] = []
    for sess in sessions_needed:
        candidates: List[str] = []
        src = sess.get("source_pdf")
        if src:
            candidates.append(str(src).strip())
        for dlg in sess.get("dialogues", []):
            for fn in dlg.get("pdf_file") or []:
                candidates.append(str(fn).strip())
        for fn in candidates:
            if not fn or fn in seen:
                continue
            seen.add(fn)
            full = str(pdfs_dir / fn)
            paths.append(full)
    return paths


def collect_pdfs_from_supporting_facts(
    supporting_facts: str,
    pdfs_dir: Path,
) -> List[str]:
    if not supporting_facts:
        return []
    seen: set = set()
    paths: List[str] = []
    for fn in re.findall(r"[^\s,]+?\.pdf", supporting_facts, flags=re.IGNORECASE):
        fn = fn.strip()
        if not fn or fn in seen:
            continue
        seen.add(fn)
        full = str(pdfs_dir / fn)
        paths.append(full)
    return paths


def render_pdf_to_pages(pdf_path: str, dpi: int = 144, max_pages: int = 30) -> List[str]:
    if not os.path.isfile(pdf_path):
        return []

    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed; cannot attach PDF: %s", pdf_path)
        return []

    cache_root = Path(os.environ.get("M3EXAM_EVAL_PDF_CACHE", "/tmp/m3exam_eval_pdf_cache"))
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        mtime = int(os.path.getmtime(pdf_path))
    except OSError:
        mtime = 0
    abspath = os.path.abspath(pdf_path)
    digest = hashlib.sha1(f"{abspath}:{mtime}".encode("utf-8")).hexdigest()[:16]
    cache_dir = cache_root / digest

    if cache_dir.is_dir():
        cached = sorted(p for p in cache_dir.glob("page_*.png"))
        if cached:
            return [str(p) for p in cached[:max_pages]]

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:                # noqa: BLE001
        logger.warning("Failed to open PDF %s: %s", pdf_path, exc)
        return []

    out: List[str] = []
    try:
        n = min(doc.page_count, max(1, int(max_pages)))
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            page_path = cache_dir / f"page_{i + 1:03d}.png"
            pix.save(str(page_path))
            out.append(str(page_path))
    finally:
        doc.close()
    return out


def expand_pdfs_to_images(
    pdf_paths: List[str],
    *,
    dpi: int = 144,
    max_pages_per_pdf: int = 30,
) -> List[str]:
    out: List[str] = []
    for pdf_path in pdf_paths:
        out.extend(render_pdf_to_pages(pdf_path, dpi=dpi, max_pages=max_pages_per_pdf))
    return out


def build_qa_prompt(context_text: str, question: str, evaluation_type: str) -> str:
    img_note = (
        "\nRelevant images from the conversation are also provided alongside this text.\n"
        if evaluation_type == "multimodal"
        else ""
    )
    return (
        "You are an AI assistant with access to a conversation history.\n"
        "Answer the following question based ONLY on the conversation provided"
        " (and the attached images if any).\n"
        f"{img_note}"
        "\n"
        "=== CONVERSATION HISTORY ===\n"
        f"{context_text}\n"
        "=== END OF CONVERSATION ===\n"
        "\n"
        f"Question: {question}\n"
        "\n"
        "Give a short, direct answer (a few words or a brief phrase). "
        "Do not add explanations or repeat the question."
    )


def _load_image_b64(path: str) -> Tuple[str, str]:
    max_side = int(cfg("api", "vision_llm", "max_image_side_px", default=2048) or 0)
    with open(path, "rb") as fh:
        raw = fh.read()
    ext  = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"

    if max_side > 0:
        try:
            from PIL import Image  # type: ignore
            im = Image.open(io.BytesIO(raw))
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS  # type: ignore[attr-defined]
            im.thumbnail((max_side, max_side), resample)
            buf = io.BytesIO()
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGBA")
                im.save(buf, format="PNG")
                mime = "image/png"
            else:
                im = im.convert("RGB")
                im.save(buf, format="JPEG", quality=85)
                mime = "image/jpeg"
            raw = buf.getvalue()
        except Exception:
            pass

    return mime, base64.b64encode(raw).decode("utf-8")


class ContextLengthExceeded(Exception):
    pass


_CTX_LEN_PATTERNS = (
    "context length",
    "context_length_exceeded",
    "maximum context length",
    "exceeds model's maximum",
    "input length",
    "too long",
    "maximum_tokens",
    "too many tokens",
)


def _is_context_length_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    code = getattr(exc, "code", None)
    if code == "context_length_exceeded":
        return True
    return any(p in msg for p in _CTX_LEN_PATTERNS)


def call_eval_model(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    image_paths: Optional[List[str]] = None,
    max_tokens: int = 200,
    wait_time: int = 1,
    image_labels: Optional[List[str]] = None,
) -> Tuple[str, int]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    if image_paths:
        for idx, img_path in enumerate(image_paths):
            if not os.path.isfile(img_path):
                logger.warning("Image not found (skipped): %s", img_path)
                continue
            try:
                mime, b64 = _load_image_b64(img_path)
                # Prepend a text label so the model can map visual content to
                # its file identifier (e.g. "img_60") - required for FM-style
                # find-matching questions whose answer is an image filename.
                if image_labels and idx < len(image_labels) and image_labels[idx]:
                    content.append({
                        "type": "text",
                        "text": f"This is {image_labels[idx]}:",
                    })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            except Exception as exc:
                logger.warning("Could not load image %s: %s", img_path, exc)

    while True:
        wait_time = wait_time * 2
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                n=1,
            )
            answer = resp.choices[0].message.content.strip()
            prompt_tokens = 0
            usage = getattr(resp, "usage", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            return answer, prompt_tokens
        except openai.RateLimitError:
            print(f"    [rate limit] retrying in {wait_time}s ...")
            time.sleep(wait_time)
        except Exception as e:
            if _is_context_length_error(e):
                raise ContextLengthExceeded(str(e)) from e
            print(f"    [API error] {e} - retrying in {wait_time}s ...")
            time.sleep(wait_time)


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    return text


# SQuAD-style normalisation for F1 / BLEU-1: lowercase, strip ALL punctuation,
# drop English articles (a/an/the), collapse whitespace.
_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def _squad_normalise(text: str) -> str:
    text = (text or "").lower()
    text = text.translate(_PUNCT_TABLE)
    tokens = [t for t in text.split() if t and t not in _ARTICLES]
    return " ".join(tokens)


def _tokenise(text: str) -> List[str]:
    return _squad_normalise(text).split()


def score_em(model_answer: str, answer_list: List[str]) -> float:
    norm_model = _normalise(model_answer)
    for i, gold in enumerate(answer_list):
        if _normalise(gold) == norm_model:
            return 1.0 / (2 ** i)
    return 0.0


def score_em_any(model_answer: str, answer_list: List[str]) -> float:
    norm_model = _normalise(model_answer)
    for gold in answer_list:
        if _normalise(gold) == norm_model:
            return 1.0
    return 0.0


def _f1_one(pred_toks: List[str], gold_toks: List[str]) -> float:
    if not pred_toks or not gold_toks:
        return 1.0 if (not pred_toks and not gold_toks) else 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_toks)
    recall    = num_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def score_f1(model_answer: str, gold: str) -> float:
    return _f1_one(_tokenise(model_answer), _tokenise(gold))


def _bleu1_one(pred_toks: List[str], gold_toks: List[str]) -> float:
    if not pred_toks or not gold_toks:
        return 1.0 if (not pred_toks and not gold_toks) else 0.0
    pred_counter = Counter(pred_toks)
    gold_counter = Counter(gold_toks)
    clipped = sum(min(cnt, gold_counter.get(w, 0)) for w, cnt in pred_counter.items())
    if clipped == 0:
        return 0.0
    c = len(pred_toks)
    r = len(gold_toks)
    precision = clipped / c
    bp = 1.0 if c >= r else math.exp(1.0 - r / c)
    return bp * precision


def score_bleu1(model_answer: str, gold: str) -> float:
    return _bleu1_one(_tokenise(model_answer), _tokenise(gold))


_LLM_JUDGE_PROMPT = """\
You are an impartial judge evaluating the memory capabilities of an AI assistant with the question-answering task. Your task is to compare the Assistant's Answer against the Ground Truth and
assign a score of 0, 0.25, 0.5, 0.75, or 1.

Question     : {question}
Gold Answer  : {gold}
Model Answer : {model_answer}

Scoring Rubric
Score 0 (Incorrect / Miss):
- The answer contradicts the Ground Truth.
- For Yes/No questions: The answer has the wrong polarity (e.g., says "Yes" when Ground Truth is
"No").
- For Open-ended questions: The answer provides factually wrong information or hallucinations.
- The assistant fails to provide the required information.
Score 0.25 (Poor / Tangential):
- The answer touches on the topic but misses the core entity or key value required.
- The answer contains a mix of minor correct details and significant hallucinations or wrong
associations.
- The answer is excessively vague to the point of being useless (e.g., answering "a dog" instead of
"a golden retriever").
Score 0.5 (Partial / Vague / Excessive):
- The answer is technically correct, but lacks confidence or is incomplete.
- The answer captures the main entity or concept correctly but misses a part of the required
supporting details.
- The answer includes the correct information, but is over informative or excessive.
- For Yes/No questions: The polarity is correct, but the reasoning is flawed (if have), or the assistant
is uncertain (e.g., "I think it might be Yes").
- For Open-ended questions: The answer is too general or misses key adjectives/details present in
the Ground Truth.
Score 0.75 (Good / Minor Imperfection):
- The answer is largely accurate and captures the core information confidently.
- It misses only minor details (e.g., specific adjectives or secondary details) that do not alter the
main truth.
- The answer contains all the correct information but includes unnecessary "fluff" or slight conversational filler that reduces precision.
Score 1 (Correct / Exact):
- The answer is accurate, precise, and confident.
- For Yes/No questions: The polarity matches the Ground Truth perfectly.
- For Open-ended questions: The answer contains all the core information and necessary details
required by the Ground Truth without hallucinations.

Reply with ONLY the score digit. No other text."""


def _judge_chat(prompt: str, max_tokens: int = 10, temperature: float = 0.0) -> str:
    client, model = _get_judge_client()
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "10"))
    max_backoff = int(os.environ.get("OPENAI_MAX_BACKOFF_SECONDS", "120"))
    backoff = 1
    last_err: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:                                  # noqa: BLE001
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
    raise RuntimeError(f"judge LLM failed after {max_retries} attempts") from last_err


def score_llm_judge(question: str, gold_label: str, model_answer: str) -> float:
    prompt = _LLM_JUDGE_PROMPT.format(
        question=question,
        gold=gold_label,
        model_answer=model_answer,
    )
    try:
        raw = _judge_chat(prompt, max_tokens=10, temperature=0.0).strip()
        m = re.search(r"[01]", raw)
        return float(m.group()) if m else 0.0
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return 0.0


def _process_question(
    q_idx: int,
    q: Dict[str, Any],
    total: int,
    session_index: Dict[str, Dict[str, Any]],
    images_dir: Path,
    eval_client: openai.OpenAI,
    eval_model: str,
    eval_type: str,
    baseline: str = "session",
    pdfs_dir: Optional[Path] = None,
    all_sessions_in_order: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    log: List[str] = []

    def _log(line: str = "") -> None:
        log.append(line)

    question    = q.get("question", "")
    answer_list = q.get("answer", [])
    label       = q.get("label", answer_list[0] if answer_list else "")
    sf          = q.get("supporting_facts", "")
    q_type      = q.get("type", "")

    _log(f"\n[{q_idx:3d}/{total}] {question[:80]}")
    _log(f"         sf={sf!r}  label={label!r}  baseline={baseline}")

    # Start the wall-clock for retrieval + generation. This covers session
    # lookup, image/PDF collection, prompt build, and the model call(s) -
    # i.e. everything the user-facing pipeline spends per question.
    t_start = time.perf_counter()

    needed_sids = parse_session_ids(sf)
    sessions_needed = [
        session_index[sid]
        for sid in needed_sids
        if sid in session_index
    ]
    missing = [sid for sid in needed_sids if sid not in session_index]
    if missing:
        _log(f"         [WARN] session(s) not found: {missing}")

    if not sessions_needed:
        _log("         [SKIP] no sessions found for this question")
        rec = {
            "question": question, "answer": answer_list, "label": label,
            "supporting_facts": sf, "type": q_type,
            "model_answer": "",
            "em_score":   0.0,
            "f1_score":   0.0,
            "bleu1_score": 0.0,
            "llm_score":  0.0,
            "num_images_attached": 0,
            "prompt_tokens": 0,
            "elapsed_sec": round(time.perf_counter() - t_start, 4),
            "error": "no_sessions_found",
        }
        return rec, log

    if baseline == "full" and all_sessions_in_order:
        blocks = build_round_blocks(list(all_sessions_in_order))
        n_sf_sessions = len(sessions_needed)
        n_total_sessions = len(all_sessions_in_order)
        _log(
            f"         text context: {n_total_sessions} session(s) "
            f"({n_sf_sessions} SF)  rounds={len(blocks)}"
        )
    else:
        blocks = build_round_blocks(sessions_needed)
        _log(f"         text context: {len(sessions_needed)} SF session(s)  rounds={len(blocks)}")

    keep_set = must_keep_round_ids(sf)

    img_paths: Optional[List[str]] = None
    n_total_imgs = 0
    n_total_pdfs = 0
    if eval_type == "multimodal":
        sf_images = [p for p in collect_session_images(sessions_needed, images_dir) if os.path.isfile(p)]

        sf_pdfs: List[str] = []
        if pdfs_dir is not None:
            sf_pdfs.extend(collect_session_pdfs(sessions_needed, pdfs_dir))
            sf_pdfs.extend(collect_pdfs_from_supporting_facts(sf, pdfs_dir))
            seen: set = set()
            sf_pdfs = [p for p in sf_pdfs if not (p in seen or seen.add(p))]
            sf_pdfs = [p for p in sf_pdfs if os.path.isfile(p)]
        pdf_pages = expand_pdfs_to_images(sf_pdfs) if sf_pdfs else []

        # PDF pages come AFTER raw session images so genuine attachments are
        # prioritised when the cap kicks in.
        all_media = sf_images + pdf_pages
        n_total_imgs = len(sf_images)
        n_total_pdfs = len(sf_pdfs)

        if len(all_media) > MAX_IMAGES_PER_QUESTION:
            _log(
                f"         media: imgs={n_total_imgs}, pdf_pages={len(pdf_pages)} "
                f"(from {n_total_pdfs} PDF) -> capped to first {MAX_IMAGES_PER_QUESTION}"
            )
            img_paths = all_media[:MAX_IMAGES_PER_QUESTION]
        else:
            _log(
                f"         media: imgs={n_total_imgs}, pdf_pages={len(pdf_pages)} "
                f"(from {n_total_pdfs} PDF)"
            )
            img_paths = all_media if all_media else None

    current_imgs = list(img_paths) if img_paths else None
    attempts     = 0
    max_attempts = 10
    model_answer = ""
    prompt_tokens = 0
    ctx_error: Optional[str] = None

    # Truncation policy:
    #   1. Drop non-SF rounds first.
    #   2. Once only SF rounds remain, start shrinking the image set.
    #   3. If both are exhausted, drop SF rounds as a last resort.
    while True:
        context_text = render_blocks(blocks)
        prompt       = build_qa_prompt(context_text, question, eval_type)
        # Label each image with its file identifier (e.g. "img_60") so the
        # model can reference images by name - essential for FM questions
        # whose gold answer is an image filename.
        img_labels = (
            [os.path.splitext(os.path.basename(p))[0] for p in current_imgs]
            if current_imgs else None
        )
        try:
            model_answer, prompt_tokens = call_eval_model(
                eval_client, eval_model, prompt,
                image_paths=current_imgs,
                max_tokens=200,
                image_labels=img_labels,
            )
            break
        except ContextLengthExceeded as e:
            attempts += 1
            if attempts > max_attempts:
                ctx_error = f"context_length_exceeded_after_{attempts}_attempts"
                _log(f"         [GIVE UP] {ctx_error}: {e}")
                model_answer = ""
                break

            actions: List[str] = []
            progress = False

            non_sf_count = sum(1 for b in blocks if b["round"] not in keep_set)
            if non_sf_count > 0:
                n_drop = max(1, non_sf_count // 3)
                new_blocks = drop_oldest_blocks(blocks, keep_set, n_drop)
                dropped = len(blocks) - len(new_blocks)
                if dropped > 0:
                    actions.append(
                        f"non-SF rounds {len(blocks)}->{len(new_blocks)}"
                    )
                    blocks = new_blocks
                    progress = True

            if not progress and current_imgs:
                if len(current_imgs) > 1:
                    new_n = max(1, len(current_imgs) // 2)
                    actions.append(f"images {len(current_imgs)}->{new_n}")
                    current_imgs = current_imgs[:new_n]
                    progress = True
                elif len(current_imgs) == 1:
                    actions.append("images 1->0")
                    current_imgs = None
                    progress = True

            if not progress and blocks and len(blocks) > 1:
                blocks = blocks[1:]
                actions.append(f"force-rounds 2->{len(blocks)}")
                progress = True

            if not progress:
                ctx_error = f"context_length_exceeded_unrecoverable: {e}"
                _log(f"         [GIVE UP] {ctx_error}")
                model_answer = ""
                break

            _log(f"         [truncate {attempts}] " + "; ".join(actions))
            continue

    elapsed_sec = time.perf_counter() - t_start
    _log(f"         model -> {model_answer[:120]!r}")
    _log(f"         prompt_tokens={prompt_tokens}  elapsed={elapsed_sec:.2f}s")

    # Per-type metric policy:
    #   fm  -> EM-only, any-match
    #   fj  -> EM-only (single-letter MCQ)
    #   else -> full metric suite (EM + F1 + BLEU-1 + LLM judge)
    qt_lower = (q_type or "").lower()
    if qt_lower == "fm":
        em    = score_em_any(model_answer, answer_list)
        f1    = 0.0
        bleu1 = 0.0
        llm   = 0.0
        _log(f"         EM(any)={em:.2f}  [fm: EM-only, any-match]")
    elif qt_lower == "fj":
        em    = score_em(model_answer, answer_list)
        f1    = 0.0
        bleu1 = 0.0
        llm   = 0.0
        _log(f"         EM={em:.2f}  [fj: EM-only]")
    else:
        em    = score_em(model_answer, answer_list)
        f1    = score_f1(model_answer, label)
        bleu1 = score_bleu1(model_answer, label)
        llm   = score_llm_judge(question, label, model_answer)
        _log(f"         EM={em:.2f}  F1={f1:.2f}  BLEU-1={bleu1:.2f}  LLM={llm:.2f}")

    rec: Dict[str, Any] = {
        "question":        question,
        "answer":          answer_list,
        "label":           label,
        "supporting_facts": sf,
        "type":            q_type,
        "baseline":        baseline,
        "model_answer":    model_answer,
        "em_score":        em,
        "f1_score":        f1,
        "bleu1_score":     bleu1,
        "llm_score":       llm,
        "num_images_total":    n_total_imgs,
        "num_pdfs_total":      n_total_pdfs,
        "num_images_attached": len(current_imgs) if current_imgs else 0,
        "context_truncations": attempts,
        "prompt_tokens":   prompt_tokens,
        "elapsed_sec":     round(elapsed_sec, 4),
    }
    if ctx_error:
        rec["error"] = ctx_error
    return rec, log


def run_evaluation() -> None:
    data_dir_str     = _eval_cfg("input_data_dir")     or ""
    question_dir_str = _eval_cfg("input_question_dir") or ""
    result_dir_str   = _eval_cfg("result_dir")         or ""
    baseline         = _eval_cfg("baseline")           or "unknown"
    eval_type        = (_eval_cfg("evaluation_type") or "text").lower()

    data_dir     = Path(data_dir_str)
    question_dir = Path(question_dir_str)
    result_dir   = Path(result_dir_str)

    if not data_dir.is_dir():
        print(f"[ERROR] input_data_dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Initialise the text-LLM client for run_chatgpt (still used for the
    # legacy default path); the judge LLM is lazily initialised inside
    # score_llm_judge via its dedicated client.
    set_openai_key()
    eval_client = _make_eval_client()
    eval_model  = _eval_model_name()

    try:
        session_index, sessions_path = load_sessions(data_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    images_dir = sessions_path.parent / "images"
    pdfs_dir   = sessions_path.parent / "pdfs"
    if not pdfs_dir.is_dir():
        pdfs_dir_for_use: Optional[Path] = None
    else:
        pdfs_dir_for_use = pdfs_dir

    try:
        questions = load_questions(question_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    dataset_name = sessions_path.parent.name

    # Normalise baseline.  Recognised values:
    #   "session" -> SF session(s) only.
    #   "full"    -> SF session(s) drive media + ALL sessions' text included.
    # Anything else is treated as "session".
    baseline_norm = (baseline or "").strip().lower()
    if baseline_norm not in ("session", "full"):
        baseline_norm = "session"

    all_sessions_in_order = list(session_index.values())

    print("=" * 64)
    print("  Evaluation")
    print("=" * 64)
    print(f"  Dataset    : {dataset_name}")
    print(f"  Baseline   : {baseline}  (normalised: {baseline_norm})")
    print(f"  Type       : {eval_type}")
    print(f"  Eval model : {eval_model}")
    print(f"  Judge model: {_judge_model_name()}")
    print(f"  Sessions   : {len(session_index)}")
    print(f"  Questions  : {len(questions)}")
    print(f"  Images dir : {images_dir}")
    print(f"  PDFs dir   : {pdfs_dir if pdfs_dir_for_use else '(none)'}")
    print("=" * 64)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset_name}_{baseline}_{ts}"
    run_dir  = result_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        concurrency = int(_eval_cfg("concurrency") or 1)
    except (TypeError, ValueError):
        concurrency = 1
    concurrency = max(1, concurrency)
    print(f"  Concurrency: {concurrency}")

    total_q = len(questions)

    records: List[Optional[Dict[str, Any]]] = [None] * total_q
    save_lock  = threading.Lock()

    def _emit(idx: int, rec: Dict[str, Any], log_lines: List[str]) -> None:
        with save_lock:
            print("\n".join(log_lines))
            records[idx] = rec
            _save_results(
                [r for r in records if r is not None],
                run_dir,
            )

    if concurrency == 1:
        for q_idx, q in enumerate(questions, start=1):
            rec, lines = _process_question(
                q_idx, q, total_q,
                session_index, images_dir,
                eval_client, eval_model, eval_type,
                baseline=baseline_norm,
                pdfs_dir=pdfs_dir_for_use,
                all_sessions_in_order=all_sessions_in_order,
            )
            _emit(q_idx - 1, rec, lines)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_idx = {
                pool.submit(
                    _process_question,
                    q_idx, q, total_q,
                    session_index, images_dir,
                    eval_client, eval_model, eval_type,
                    baseline_norm,
                    pdfs_dir_for_use,
                    all_sessions_in_order,
                ): q_idx - 1
                for q_idx, q in enumerate(questions, start=1)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    rec, lines = fut.result()
                except Exception as exc:                        # noqa: BLE001
                    rec = {
                        "question": questions[idx].get("question", ""),
                        "answer":   questions[idx].get("answer", []),
                        "label":    questions[idx].get("label", ""),
                        "supporting_facts": questions[idx].get("supporting_facts", ""),
                        "type":     questions[idx].get("type", ""),
                        "baseline": baseline_norm,
                        "model_answer": "",
                        "em_score": 0.0, "f1_score": 0.0,
                        "bleu1_score": 0.0, "llm_score": 0.0,
                        "num_images_attached": 0,
                        "num_pdfs_total": 0,
                        "prompt_tokens": 0,
                        "elapsed_sec": 0.0,
                        "error": f"worker_exception: {exc!r}",
                    }
                    lines = [f"\n[{idx + 1:3d}/{total_q}] worker exception: {exc!r}"]
                _emit(idx, rec, lines)

    records = [r for r in records if r is not None]                # type: ignore[assignment]

    n = len(records)
    answered = [r for r in records if not r.get("error")]
    n_ans = len(answered)

    rubric = build_rubric(records)
    total_row = rubric.get("total", {})
    em_avg   = float(total_row.get("em_score")    or 0.0)
    f1_avg   = float(total_row.get("f1_score")    or 0.0)
    bleu_avg = float(total_row.get("bleu1_score") or 0.0)
    llm_avg  = float(total_row.get("llm_score")   or 0.0)

    token_vals = [int(r.get("prompt_tokens") or 0) for r in records]
    token_vals_nonzero = [v for v in token_vals if v > 0]
    avg_prompt_tokens = (
        sum(token_vals_nonzero) / len(token_vals_nonzero)
        if token_vals_nonzero else 0.0
    )
    time_vals = [float(r.get("elapsed_sec") or 0.0) for r in records]
    avg_elapsed_sec = sum(time_vals) / len(time_vals) if time_vals else 0.0
    total_elapsed_sec = sum(time_vals)

    summary = {
        "dataset":          dataset_name,
        "baseline":         baseline,
        "evaluation_type":  eval_type,
        "model":            eval_model,
        "judge_model":      _judge_model_name(),
        "num_questions":    n,
        "num_answered":     n_ans,
        "em_score":         round(em_avg,   4),
        "f1_score":         round(f1_avg,   4),
        "bleu1_score":      round(bleu_avg, 4),
        "llm_score":        round(llm_avg,  4),
        "avg_prompt_tokens":   round(avg_prompt_tokens, 2),
        "avg_elapsed_sec":     round(avg_elapsed_sec, 4),
        "total_elapsed_sec":   round(total_elapsed_sec, 4),
        "prompt_tokens_samples": len(token_vals_nonzero),
        "max_images_per_question": MAX_IMAGES_PER_QUESTION,
        "rubric":           rubric,
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    rubric_path = run_dir / "rubric.json"
    with open(rubric_path, "w", encoding="utf-8") as f:
        json.dump(rubric, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print("  Evaluation Summary")
    print("=" * 64)
    print(f"  Dataset         : {dataset_name}")
    print(f"  Baseline        : {baseline}")
    print(f"  Type            : {eval_type}")
    print(f"  Eval model      : {eval_model}")
    print(f"  Judge model     : {_judge_model_name()}")
    print(f"  Questions       : {n}  (answered: {n_ans})")
    print(f"  EM     (avg)    : {em_avg:.4f}")
    print(f"  F1     (avg)    : {f1_avg:.4f}")
    print(f"  BLEU-1 (avg)    : {bleu_avg:.4f}")
    print(f"  LLM J. (avg)    : {llm_avg:.4f}")
    print(
        f"  Prompt tokens   : avg {avg_prompt_tokens:.1f} "
        f"(over {len(token_vals_nonzero)}/{n} questions w/ usage)"
    )
    print(
        f"  Retr.+Gen. time : avg {avg_elapsed_sec:.2f}s / question  "
        f"(total {total_elapsed_sec:.1f}s over {n})"
    )
    print_rubric_table(rubric)
    print(f"  Results dir     : {run_dir}")
    print("=" * 64)


_METRIC_KEYS = ("em_score", "f1_score", "bleu1_score", "llm_score")

# Question types whose only meaningful metric is EM. F1 / BLEU-1 / LLM are
# reported as "-" for these types and are excluded from the global averages.
_EM_ONLY_TYPES = {"fm", "fj"}

EXPECTED_TYPES: Tuple[str, ...] = (
    "ss",
    "ms",
    "tr",
    "mr",
    "th",
    "fm",
    "fj",
    "ii",
)


def build_rubric(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_type: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        t = (rec.get("type") or "unknown").lower()
        per_type.setdefault(t, []).append(rec)

    for t in EXPECTED_TYPES:
        per_type.setdefault(t, [])

    def _agg(group: List[Dict[str, Any]], type_name: Optional[str] = None) -> Dict[str, Any]:
        c = len(group)
        ans = sum(1 for r in group if not r.get("error"))
        out: Dict[str, Any] = {"count": c, "answered": ans}
        em_only = (type_name or "").lower() in _EM_ONLY_TYPES
        for k in _METRIC_KEYS:
            if c == 0:
                out[k] = 0.0
                continue
            if em_only and k != "em_score":
                out[k] = None
            else:
                out[k] = round(sum(float(r.get(k, 0.0)) for r in group) / c, 4)
        return out

    def _agg_total(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        c = len(records)
        ans = sum(1 for r in records if not r.get("error"))
        out: Dict[str, Any] = {"count": c, "answered": ans}
        for k in _METRIC_KEYS:
            if c == 0:
                out[k] = 0.0
                continue
            if k == "em_score":
                vals = [float(r.get(k, 0.0)) for r in records]
            else:
                vals = [
                    float(r.get(k, 0.0))
                    for r in records
                    if (r.get("type") or "").lower() not in _EM_ONLY_TYPES
                ]
            out[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
        return out

    ordered: List[str] = list(EXPECTED_TYPES)
    extras = sorted(t for t in per_type.keys() if t not in EXPECTED_TYPES)
    ordered.extend(extras)

    rubric: Dict[str, Any] = {
        "per_type": {t: _agg(per_type.get(t, []), t) for t in ordered},
        "total":    _agg_total(records),
    }
    return rubric


def _fmt_metric(v: Any) -> str:
    if v is None:
        return f"{'-':>10}"
    try:
        return f"{float(v):>10.4f}"
    except (TypeError, ValueError):
        return f"{'-':>10}"


def print_rubric_table(rubric: Dict[str, Any]) -> None:
    print("-" * 64)
    print("  Per-type Rubric")
    print("-" * 64)
    header = f"  {'type':<8}{'N':>5}{'EM':>10}{'F1':>10}{'BLEU-1':>10}{'LLM':>10}"
    print(header)
    for t, agg in rubric.get("per_type", {}).items():
        print(
            f"  {t:<8}"
            f"{agg.get('count', 0):>5}"
            f"{_fmt_metric(agg.get('em_score'))}"
            f"{_fmt_metric(agg.get('f1_score'))}"
            f"{_fmt_metric(agg.get('bleu1_score'))}"
            f"{_fmt_metric(agg.get('llm_score'))}"
        )
    tot = rubric.get("total", {})
    if tot:
        print(
            f"  {'TOTAL':<8}"
            f"{tot.get('count', 0):>5}"
            f"{_fmt_metric(tot.get('em_score'))}"
            f"{_fmt_metric(tot.get('f1_score'))}"
            f"{_fmt_metric(tot.get('bleu1_score'))}"
            f"{_fmt_metric(tot.get('llm_score'))}"
        )
    print("-" * 64)


def _save_results(records: List[Dict[str, Any]], run_dir: Path) -> None:
    path = run_dir / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, rec in enumerate(records):
            comma = "," if i < len(records) - 1 else ""
            f.write("  " + json.dumps(rec, ensure_ascii=False) + comma + "\n")
        f.write("]\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run_evaluation()
