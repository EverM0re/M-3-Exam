from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any, Dict, List, Tuple

_WS = re.compile(r"\s+")
_PUNC = str.maketrans("", "", string.punctuation)
_IMG_RE = re.compile(r"\bimg[_-]?(\d+)(?:\.[a-zA-Z0-9]+)?\b", re.IGNORECASE)


def _norm(s: str) -> str:
    s = (s or "").lower().translate(_PUNC)
    s = _WS.sub(" ", s).strip()
    return s


def _tokenize(s: str) -> List[str]:
    return _norm(s).split()


def em_text(pred: str, gold_list: List[str]) -> float:
    pn = _norm(pred)
    for g in gold_list:
        if pn == _norm(g):
            return 1.0
    return 0.0


def f1_text(pred: str, label: str) -> float:
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(label)
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pred_toks)
    r = n / len(gold_toks)
    return 2 * p * r / (p + r)


def bleu1_text(pred: str, label: str) -> float:
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(label)
    if not pred_toks or not gold_toks:
        return 0.0
    gold_counts = Counter(gold_toks)
    overlap = 0
    pred_counts: Counter = Counter()
    for t in pred_toks:
        if pred_counts[t] < gold_counts[t]:
            overlap += 1
        pred_counts[t] += 1
    precision = overlap / len(pred_toks)
    bp = 1.0 if len(pred_toks) >= len(gold_toks) else math.exp(
        1 - len(gold_toks) / max(1, len(pred_toks))
    )
    return precision * bp


def _img_ids(s: str) -> set:
    return set(int(m) for m in _IMG_RE.findall(s or ""))


def fm_em_image(pred: str, gold_list: List[str]) -> Tuple[float, Dict[str, float]]:
    gold_ids: set = set()
    for g in gold_list:
        gold_ids |= _img_ids(g)
    if not gold_ids:
        return em_text(pred, gold_list), {}
    pred_ids = _img_ids(pred)
    em = 1.0 if pred_ids & gold_ids else 0.0
    return em, {}


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_type: Dict[str, Dict[str, float]] = {}
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        t = rec.get("type", "")
        by_type.setdefault(t, []).append(rec)

    def avg(key: str, rs: List[Dict[str, Any]]) -> float:
        if not rs:
            return 0.0
        vals = [float(r.get(key, 0.0) or 0.0) for r in rs]
        return round(sum(vals) / len(vals), 4)

    for t, rs in by_type.items():
        per_type[t] = {
            "count": len(rs),
            "answered": sum(1 for r in rs if (r.get("model_answer") or "").strip()),
            "em_score": avg("em_score", rs),
            "f1_score": avg("f1_score", rs),
            "bleu1_score": avg("bleu1_score", rs),
            "llm_score": avg("llm_score", rs),
        }

    total_em = avg("em_score", records)
    soft_rs = [r for r in records if r.get("type") not in ("fm", "fj")]
    return {
        "per_type": per_type,
        "total": {
            "count": len(records),
            "answered": sum(1 for r in records if (r.get("model_answer") or "").strip()),
            "em_score": total_em,
            "f1_score": avg("f1_score", soft_rs),
            "bleu1_score": avg("bleu1_score", soft_rs),
            "llm_score": avg("llm_score", soft_rs),
        },
    }
