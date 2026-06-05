from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

# and question.json supporting_facts  round-id formatconsistent, such as D1:2, D12:35
_ROUND_ID_RE = re.compile(r"\b([A-Z]\d+:\d+)\b")

# retrieve context  round-id marker: [D1:2]
_CONTEXT_ROUND_RE = re.compile(r"\[([A-Z]\d+:\d+)\]")
# match「img」+ optional _/- + number + optionalextension; **will not**onlymatchthe 3-letter `img`, alsowill notmatchwithout a number  img_. 
# literal img_<number> placeholderno \d+, the whole string doesn't match. 
_IMAGE_FILE_RE = re.compile(r"\bimg[_-]?(\d+)(?:\.[a-zA-Z0-9]+)?\b", re.IGNORECASE)

# model copiesin the hint placeholderexample (non-real img_60 number) 
_IMG_PLACEHOLDER_LITERAL_RE = re.compile(r"img_\s*<\s*number\s*>", re.IGNORECASE)

EXTRA_METRIC_KEYS = ("build_index_time", "retrieve_time", "recall_at_k", "tokens")


def stable_mmkg_doc_id(json_stem: str, round_id: Any, line_index: int) -> str:
    rid = re.sub(r"[^0-9a-zA-Z_-]+", "_", str(round_id).strip())
    rid = rid.strip("_")[:96]
    if not rid:
        rid = f"row{line_index}"
    return f"doc-mmkg-{json_stem}-{rid}"


def stable_lightrag_doc_id_from_file_path(
    file_path: str,
    *,
    prefix: str = "doc-text",
) -> str:
    stem = Path(str(file_path)).stem
    safe = re.sub(r"[^0-9a-zA-Z_-]+", "_", stem).strip("_")[:96]
    return f"{prefix}-{safe}" if safe else f"{prefix}-unknown"


def normalize_round_id_for_trace(round_id: Any) -> str:
    return str(round_id or "").strip().replace(" ", "")


def _leading_trace_headers_match(body: str, desired: List[str]) -> bool:
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    if len(lines) < len(desired):
        return False
    return lines[: len(desired)] == desired


def prepend_eval_round_trace(
    text: str,
    round_id: Any = None,
    *,
    chunk_id: Optional[str] = None,
    image_basenames: Optional[Sequence[str]] = None,
) -> str:
    body = text if isinstance(text, str) else str(text)
    if not body.strip():
        return body
    rid = normalize_round_id_for_trace(round_id)
    cid = (chunk_id or "").strip()
    desired: List[str] = []
    if rid:
        desired.extend([f"[{rid}] Round {rid}", f"[TURN_ID] {rid}"])
    if cid:
        desired.append(f"[CHUNK_ID] {cid}")
    img_line: Optional[str] = None
    if image_basenames:
        names = [
            Path(str(x)).name.strip()
            for x in image_basenames
            if str(x).strip()
        ]
        if names:
            img_line = "[IMAGE_FILES] " + ", ".join(names)
            desired.append(img_line)
    if not desired:
        return body
    if _leading_trace_headers_match(body, desired):
        return body
    # legacy header already present but missing [IMAGE_FILES]: only on the first round/ chunk id header-match, and afterwardsinsert a line
    if img_line and "[IMAGE_FILES]" not in body[:1600]:
        n_need = 0
        if rid and cid:
            n_need = 3
        elif rid:
            n_need = 2
        if n_need:
            raw_lines = body.split("\n")
            nonempty_idx: List[int] = []
            for i, ln in enumerate(raw_lines):
                if ln.strip():
                    nonempty_idx.append(i)
                    if len(nonempty_idx) >= n_need:
                        break
            if len(nonempty_idx) >= n_need:
                stripped = [raw_lines[i].strip() for i in nonempty_idx[:n_need]]
                head_ok = stripped[0].startswith(f"[{rid}]") and stripped[1] == f"[TURN_ID] {rid}"
                if cid:
                    head_ok = head_ok and len(stripped) >= 3 and stripped[2] == f"[CHUNK_ID] {cid}"
                if head_ok:
                    ins = nonempty_idx[n_need - 1]
                    merged = raw_lines[: ins + 1] + [img_line, ""] + raw_lines[ins + 1 :]
                    return "\n".join(merged)
    return "\n".join(desired) + "\n\n" + body.lstrip("\n")


def parsing_gold_rounds(supporting_facts: str) -> Set[str]:
    if not supporting_facts or not str(supporting_facts).strip():
        return set()
    return set(_ROUND_ID_RE.findall(str(supporting_facts)))


# model annotates the source round at the tail of the answer, such as ``Medial saphenous vein [rounds: D12:4]``
_ANSWER_ROUND_CITATION_SUFFIX_RE = re.compile(
    r"\s*\[(?:rounds?|source|turns?)\s*:\s*([^\]]+)\]\s*$",
    re.IGNORECASE,
)


def strip_answer_round_citation(answer: str) -> tuple[str, List[str]]:
    text = str(answer or "")
    m = _ANSWER_ROUND_CITATION_SUFFIX_RE.search(text)
    if not m:
        return text.strip(), []
    body = text[: m.start()].strip()
    cited = _ROUND_ID_RE.findall(m.group(1))
    seen: set[str] = set()
    ordered: List[str] = []
    for rid in cited:
        rr = rid.replace(" ", "")
        if rr and rr not in seen:
            seen.add(rr)
            ordered.append(rr)
    return body, ordered


def append_answer_round_citation(answer: str, round_ids: Sequence[str]) -> str:
    body, cited = strip_answer_round_citation(answer)
    if cited:
        return str(answer or "").strip()
    hints: List[str] = []
    seen: set[str] = set()
    for rid in round_ids:
        rr = normalize_round_id_for_trace(rid)
        if rr and rr not in seen:
            seen.add(rr)
            hints.append(rr)
        if len(hints) >= 3:
            break
    if not hints:
        return body
    if not body.strip():
        return ""
    return f"{body} [rounds: {', '.join(hints)}]"


ROUND_CITATION_PROMPT = (
    "After your answer, append one mandatory suffix listing the 1–3 conversation round "
    "ids you relied on, using markers from the memories (e.g. [D12:4] or Round D12:4). "
    "Format exactly: [rounds: D12:4] or [rounds: D12:4, D3:5]. "
    "Put the direct answer first, then a space, then the bracket suffix. "
    "Never output only the [rounds: ...] suffix without an answer phrase before it."
)


def parsing_rounds_in_context(context: str) -> Set[str]:
    if not context:
        return set()
    bracketed = set(_CONTEXT_ROUND_RE.findall(context))
    bare = set(_ROUND_ID_RE.findall(str(context)))
    return bracketed | bare


def recall_at_top_k(
    supporting_facts: str,
    context: str,
) -> Optional[float]:
    gold = parsing_gold_rounds(supporting_facts)
    if not gold:
        return None
    retrieved = parsing_rounds_in_context(context)
    hits = len(gold & retrieved)
    return hits / len(gold)


def normalize_image_file(value: str) -> Optional[str]:
    if not value:
        return None
    name = Path(str(value).strip()).name
    m = _IMAGE_FILE_RE.search(name)
    if not m:
        return None
    return f"img_{int(m.group(1))}"


def extract_image_files(value: str) -> Set[str]:
    if not value:
        return set()
    return {f"img_{int(m.group(1))}" for m in _IMAGE_FILE_RE.finditer(str(value))}


def fm_answer_has_placeholder_literal(text: str) -> bool:
    if not text or not str(text).strip():
        return False
    return bool(_IMG_PLACEHOLDER_LITERAL_RE.search(str(text)))


def gold_image_id_set(answers: Any) -> Set[str]:
    if answers is None:
        return set()
    if isinstance(answers, str):
        vals = [answers]
    elif isinstance(answers, list):
        vals = answers
    else:
        vals = []
    out: Set[str] = set()
    for a in vals:
        s = str(a)
        out |= extract_image_files(s)
        n = normalize_image_file(s)
        if n:
            out.add(n)
    return out


def fm_filename_image_metrics(prediction: str, gold_ids: Set[str]) -> Dict[str, Any]:
    pred_ids = extract_image_files(prediction)
    gold_g = set(gold_ids)
    if not gold_g:
        return {
            "pred_image_ids": sorted(pred_ids),
            "gold_image_ids": [],
            "image_any_hit": 0.0,
            "image_strict_set_em": 0.0,
            "image_set_em": 0.0,
            "image_precision": 0.0,
            "image_recall": 0.0,
            "image_f1": 0.0,
        }
    inter = pred_ids & gold_g
    prec = len(inter) / len(pred_ids) if pred_ids else 0.0
    rec = len(inter) / len(gold_g)
    if prec + rec <= 0:
        f1 = 0.0
    else:
        f1 = 2.0 * prec * rec / (prec + rec)
    any_hit = 1.0 if inter else 0.0
    strict = 1.0 if pred_ids == gold_g else 0.0
    return {
        "pred_image_ids": sorted(pred_ids),
        "gold_image_ids": sorted(gold_g),
        "image_any_hit": any_hit,
        "image_strict_set_em": strict,
        "image_set_em": any_hit,
        "image_precision": prec,
        "image_recall": rec,
        "image_f1": f1,
    }


def image_exact_match(prediction: str, answers: Any) -> Optional[float]:
    preds = extract_image_files(prediction)
    if not preds:
        return 0.0
    if isinstance(answers, str):
        answer_values = [answers]
    elif isinstance(answers, list):
        answer_values = answers
    else:
        answer_values = []
    gold = {x for x in (normalize_image_file(str(a)) for a in answer_values) if x}
    if not gold:
        return None
    return 1.0 if preds & gold else 0.0


def _coerce_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    if isinstance(value, dict):
        return sum(_coerce_token_count(v) for v in value.values())
    return 0


def openai_usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif isinstance(usage, dict):
        data = dict(usage)
    else:
        data = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        ):
            value = getattr(usage, key, None)
            if value is not None:
                data[key] = value
    prompt = _coerce_token_count(
        data.get("prompt_tokens", data.get("input_tokens"))
    )
    completion = _coerce_token_count(
        data.get("completion_tokens", data.get("output_tokens"))
    )
    total = _coerce_token_count(data.get("total_tokens"))
    if not total and (prompt or completion):
        total = prompt + completion
    out: Dict[str, Any] = {}
    if prompt:
        out["prompt_tokens"] = prompt
    if completion:
        out["completion_tokens"] = completion
    if total:
        out["total_tokens"] = total
    return out


def load_metrics_flags_from_cfg(cfg: Dict[str, Any]) -> Dict[str, bool]:
    out = {k: False for k in EXTRA_METRIC_KEYS}
    m = cfg.get("eval", {}).get("metrics") if isinstance(cfg.get("eval"), dict) else None
    if not isinstance(m, dict):
        return out
    for k in EXTRA_METRIC_KEYS:
        if k in m:
            out[k] = bool(m[k])
    return out


def merge_metrics_cli(
    base: Dict[str, bool],
    cli_spec: Optional[str],
) -> Dict[str, bool]:
    """
    cli_spec:
      None  → keep base
      '' / 'none' → all False
      'all' → all True
      comma-separated; aliases: build_index → build_index_time, retrieve → retrieve_time, 
                    recall → recall_at_k
    """
    if cli_spec is None:
        return dict(base)
    s = cli_spec.strip().lower()
    if s in ("", "none", "off"):
        return {k: False for k in EXTRA_METRIC_KEYS}
    if s in ("all", "full"):
        return {k: True for k in EXTRA_METRIC_KEYS}

    alias = {
        "build_index": "build_index_time",
        "build": "build_index_time",
        "index": "build_index_time",
        "retrieve": "retrieve_time",
        "latency": "retrieve_time",
        "recall": "recall_at_k",
        "recall_at_k": "recall_at_k",
        "token": "tokens",
        "tokens": "tokens",
    }
    out = {k: False for k in EXTRA_METRIC_KEYS}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        key = alias.get(part, part)
        if key in out:
            out[key] = True
    return out

